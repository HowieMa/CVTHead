# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from typing import Union



def apply_imagenet_normalization(input):
    r"""Normalize using ImageNet mean and std.
    Args:
        input (4D tensor NxCxHxW): The input images, assuming to be [0, 1].
    Returns:
        Normalized inputs using the ImageNet normalization.
    """
    # normalize the input using the ImageNet mean and std
    mean = input.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = input.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    output = (input - mean) / std
    return output


class PerceptualLossWrapper(object):
    def __init__(self, num_scales, use_gpu, use_fp16):
        self.loss = PerceptualLoss(
            num_scales=num_scales,
            use_fp16=use_fp16)

        if use_gpu:
            self.loss = self.loss.cuda()

        self.forward = self.loss.forward


class PerceptualLoss(nn.Module):
    r"""Perceptual loss initialization.
    Args:
        network (str) : The name of the loss network: 'vgg16' | 'vgg19'.
        layers (str or list of str) : The layers used to compute the loss.
        weights (float or list of float : The loss weights of each layer.
        criterion (str): The type of distance function: 'l1' | 'l2'.
        resize (bool) : If ``True``, resize the inputsut images to 224x224.
        resize_mode (str): Algorithm used for resizing.
        instance_normalized (bool): If ``True``, applies instance normalization
            to the feature maps before computing the distance.
        num_scales (int): The loss will be evaluated at original size and
            this many times downsampled sizes.
        use_fp16 (bool) : If ``True``, use cast networks and inputs to FP16
    """

    def __init__(
            self, 
            network='vgg19', 
            layers=('relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1', 'relu_5_1'), 
            weights=(0.03125, 0.0625, 0.125, 0.25, 1.0),
            criterion='l1', 
            resize=False, 
            resize_mode='bilinear',
            instance_normalized=False,
            replace_maxpool_with_avgpool=False,
            num_scales=1,
            use_fp16=False
        ) -> None:
        super(PerceptualLoss, self).__init__()
        if isinstance(layers, str):
            layers = [layers]
        if weights is None:
            weights = [1.] * len(layers)
        elif isinstance(layers, float) or isinstance(layers, int):
            weights = [weights]

        assert len(layers) == len(weights), \
            'The number of layers (%s) must be equal to ' \
            'the number of weights (%s).' % (len(layers), len(weights))
        if network == 'vgg19':
            self.model = _vgg19(layers)
        elif network == 'vgg16':
            self.model = _vgg16(layers)
        elif network == 'alexnet':
            self.model = _alexnet(layers)
        elif network == 'inception_v3':
            self.model = _inception_v3(layers)
        elif network == 'resnet50':
            self.model = _resnet50(layers)
        elif network == 'robust_resnet50':
            self.model = _robust_resnet50(layers)
        elif network == 'vgg_face_dag':
            self.model = _vgg_face_dag(layers)
        else:
            raise ValueError('Network %s is not recognized' % network)

        if replace_maxpool_with_avgpool:
            for k, v in self.model.network._modules.items():
                if isinstance(v, nn.MaxPool2d):
                    self.model.network._modules[k] = nn.AvgPool2d(2)

        self.num_scales = num_scales
        self.layers = layers
        self.weights = weights
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = nn.MSEloss()
        else:
            raise ValueError('Criterion %s is not recognized' % criterion)
        self.resize = resize
        self.resize_mode = resize_mode
        self.instance_normalized = instance_normalized
        self.fp16 = use_fp16
        if self.fp16:
            self.model.half()

    @torch.cuda.amp.autocast(True)
    def forward(self, 
                inputs: Union[torch.Tensor, list], 
                target: torch.Tensor) -> Union[torch.Tensor, list]:
        r"""Perceptual loss forward.
        Args:
           inputs (4D tensor or list of 4D tensors) : inputsut tensor.
           target (4D tensor) : Ground truth tensor, same shape as the inputsut.
        Returns:
           (scalar tensor or list of tensors) : The perceptual loss.
        """
        if isinstance(inputs, list):
            # Concat alongside the batch axis
            input_is_a_list = True
            num_chunks = len(inputs)
            inputs = torch.cat(inputs)
        else:
            input_is_a_list = False

        # Perceptual loss should operate in eval mode by default.
        self.model.eval()
        inputs, target = \
            apply_imagenet_normalization(inputs), \
            apply_imagenet_normalization(target)
        if self.resize:
            inputs = F.interpolate(
                inputs, mode=self.resize_mode, size=(224, 224),
                align_corners=False)
            target = F.interpolate(
                target, mode=self.resize_mode, size=(224, 224),
                align_corners=False)

        # Evaluate perceptual loss at each scale.
        loss = 0

        for scale in range(self.num_scales):
            if self.fp16:
                input_features = self.model(inputs.half())
                with torch.no_grad():
                    target_features = self.model(target.half())
            else:
                input_features = self.model(inputs)
                with torch.no_grad():
                    target_features = self.model(target)

            for layer, weight in zip(self.layers, self.weights):
                # Example per-layer VGG19 loss values after applying
                # [0.03125, 0.0625, 0.125, 0.25, 1.0] weighting.
                # relu_1_1, 0.014698
                # relu_2_1, 0.085817
                # relu_3_1, 0.349977
                # relu_4_1, 0.544188
                # relu_5_1, 0.906261
                input_feature = input_features[layer]
                target_feature = target_features[layer].detach()
                if self.instance_normalized:
                    input_feature = F.instance_norm(input_feature)
                    target_feature = F.instance_norm(target_feature)

                if input_is_a_list:
                    target_feature = torch.cat([target_feature] * num_chunks)

                loss += weight * self.criterion(input_feature,
                                                target_feature)

            # Downsample the inputsut and target.
            if scale != self.num_scales - 1:
                inputs = F.interpolate(
                    inputs, mode=self.resize_mode, scale_factor=0.5,
                    align_corners=False, recompute_scale_factor=True)
                target = F.interpolate(
                    target, mode=self.resize_mode, scale_factor=0.5,
                    align_corners=False, recompute_scale_factor=True)

        loss /= self.num_scales

        return loss

    def train(self, mode: bool = True):
        return self


class _PerceptualNetwork(nn.Module):
    r"""The network that extracts features to compute the perceptual loss.
    Args:
        network (nn.Sequential) : The network that extracts features.
        layer_name_mapping (dict) : The dictionary that
            maps a layer's index to its name.
        layers (list of str): The list of layer names that we are using.
    """

    def __init__(self, network, layer_name_mapping, layers):
        super().__init__()
        assert isinstance(network, nn.Sequential), \
            'The network needs to be of type "nn.Sequential".'
        self.network = network
        self.layer_name_mapping = layer_name_mapping
        self.layers = layers

        for m in self.network.modules():
            names = [name for name, _ in m.named_parameters()]
            for name in names:
                if hasattr(m, name):
                    data = getattr(m, name).data
                    delattr(m, name)
                    m.register_buffer(name, data, persistent=False)

    def forward(self, x):
        r"""Extract perceptual features."""
        output = {}
        for i, layer in enumerate(self.network):
            x = layer(x)
            layer_name = self.layer_name_mapping.get(i, None)
            if layer_name in self.layers:
                # If the current layer is used by the perceptual loss.
                output[layer_name] = x
        return output


def _vgg19(layers):
    r"""Get vgg19 layers"""
    network = torchvision.models.vgg19(pretrained=True).features
    layer_name_mapping = {1: 'relu_1_1',
                          3: 'relu_1_2',
                          6: 'relu_2_1',
                          8: 'relu_2_2',
                          11: 'relu_3_1',
                          13: 'relu_3_2',
                          15: 'relu_3_3',
                          17: 'relu_3_4',
                          20: 'relu_4_1',
                          22: 'relu_4_2',
                          24: 'relu_4_3',
                          26: 'relu_4_4',
                          29: 'relu_5_1'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)



if __name__ == "__main__":
    vgg_loss = PerceptualLossWrapper(num_scales=1, use_gpu=True, use_fp16=False)

    x = torch.rand(1, 3, 256, 256).float().cuda()
    y = torch.rand(1, 3, 256, 256).float().cuda()

    loss = vgg_loss.forward(x, y)
    