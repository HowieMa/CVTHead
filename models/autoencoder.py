"""
Summary from ROME 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List



class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class BottleneckBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        mid_channels: int = -1,
        spade_channels: int = -1,
        num_layers: int = 3,
        expansion_factor: int = 4,
        kernel_size: Union[int, Tuple[int], List[int]] = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        conv_layer_type: str = 'conv',
        norm_layer_type: str = 'bn',
        activation_type: str = 'relu',
        skip_layer_type: str = 'conv',
        resize_layer_type: str = 'none'):
        """This is a base module for a residual bottleneck block"""
        super(BottleneckBlock, self).__init__()
        if stride > 1 and resize_layer_type in ['nearest', 'bilinear']:
            self.upsample = lambda inputs: F.interpolate(input=inputs, 
                                                         scale_factor=stride, 
                                                         mode=resize_layer_type, 
                                                         align_corners=None if resize_layer_type == 'nearest' else False)

        if mid_channels == -1:
            mid_channels = out_channels

        if norm_layer_type != 'none':
            norm_layer = norm_layers[norm_layer_type]
        activation = activations[activation_type]
        conv_layer = conv_layers[conv_layer_type]
        skip_layer = conv_layers[skip_layer_type]

        layers_ = []

        if norm_layer_type != 'none':
            if spade_channels != -1:
                layers_ += [norm_layer(in_channels * expansion_factor, spade_channels, affine=True)]
            else:
                layers_ += [norm_layer(in_channels * expansion_factor, affine=True)]

        layers_ += [
            activation(inplace=True),
            conv_layer(
                in_channels=in_channels * expansion_factor,
                out_channels=mid_channels,
                kernel_size=1,
                bias=norm_layer_type == 'none')]

        if norm_layer_type != 'none':
            if spade_channels != -1:
                layers_ += [norm_layer(mid_channels, spade_channels, affine=True)]
            else:
                layers_ += [norm_layer(mid_channels, affine=True)]
        layers_ += [activation(inplace=True)]

        assert num_layers > 2, 'Minimum number of layers is 3'
        for i in range(num_layers - 2):
            layers_ += [
                conv_layer(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=kernel_size, 
                    stride=stride if resize_layer_type == 'none' and i == num_layers - 3 else 1,
                    padding=padding, 
                    dilation=dilation, 
                    groups=groups,
                    bias=norm_layer_type == 'none')]

            if norm_layer_type != 'none':
                if spade_channels != -1:
                    layers_ += [norm_layer(mid_channels, spade_channels, affine=True)]
                else:
                    layers_ += [norm_layer(mid_channels, affine=True)]
            layers_ += [activation(inplace=True)]

        layers_ += [
            skip_layer(
                in_channels=mid_channels,
                out_channels=out_channels * expansion_factor,
                kernel_size=1,
                bias=norm_layer_type == 'none')]

        self.main = nn.Sequential(*layers_)

        if in_channels != out_channels:
            self.skip = skip_layer(
                in_channels=in_channels * expansion_factor,
                out_channels=out_channels * expansion_factor, 
                kernel_size=1,
                bias=norm_layer_type == 'none')
        else:
            self.skip = nn.Identity()

        if stride > 1 and resize_layer_type in downsampling_layers:
            self.downsample = downsampling_layers[resize_layer_type](stride)

    def forward(self, x):
        if hasattr(self, 'upsample'):
            x = self.upsample(x)

        x = self.main(x) + self.skip(x)

        if hasattr(self, 'downsample'):
            x = self.downsample(x)

        return x




############################################################
#                Definitions for the layers                #
############################################################

# Supported conv layers
conv_layers = {
    'conv': nn.Conv2d,
    'ws_conv': WSConv2d,
    # 'conv_3d': nn.Conv3d,
    # 'ada_conv': convs.AdaptiveConv,
    # 'ada_conv_3d': convs.AdaptiveConv3d

}

# Supported activations
activations = {
    'relu': nn.ReLU,
    # 'lrelu': functools.partial(nn.LeakyReLU, negative_slope=0.2)
}

# Supported normalization layers
norm_layers = {
    'in': nn.InstanceNorm2d,
    'in_3d': nn.InstanceNorm3d,
    'bn': nn.BatchNorm2d,
    'gn': lambda num_features, affine=True: nn.GroupNorm(num_groups=min(32, num_features), num_channels=num_features, affine=affine),
    # 'ada_in': norms.AdaptiveInstanceNorm,
    # 'ada_gn': lambda num_features, affine=True: norms.AdaptiveGroupNorm(num_groups=min(32, num_features), num_features=num_features, affine=affine),
}

# Supported downsampling layers
downsampling_layers = {
    'avgpool': nn.AvgPool2d,
    'maxpool': nn.MaxPool2d,
    'avgpool_3d': nn.AvgPool3d,
    'maxpool_3d': nn.MaxPool3d,
    # 'pixelunshuffle': PixelUnShuffle    
}




class Autoencoder(nn.Module):
    def __init__(self,
                 num_channels: int,
                 max_channels: int,
                 num_groups=4,
                 num_blocks=2,
                 num_layers=4,
                 input_channels=4):
        super(Autoencoder, self).__init__()
        # Encoder from inputs to latents
        expansion_factor = 4

        layers_ = [nn.Conv2d(input_channels, num_channels * expansion_factor, 7, 1, 3, bias=False)]
        in_channels = num_channels
        out_channels = num_channels

        for i in range(num_groups):     # 4
            for j in range(num_blocks): # 2
                layers_.append(BottleneckBlock(
                    in_channels=in_channels if j == 0 else out_channels,
                    out_channels=out_channels,
                    mid_channels=out_channels,
                    expansion_factor=expansion_factor,
                    num_layers=num_layers,
                    kernel_size=3,
                    stride=1,
                    norm_layer_type='gn',
                    activation_type='relu',
                    conv_layer_type='ws_conv'))

            in_channels = out_channels
            out_channels = min(num_channels * 2 ** (i + 1), max_channels)

            if i < num_groups - 1:
                layers_.append(nn.MaxPool2d(kernel_size=2))

        self.net = nn.Sequential(*layers_)
 
    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    x = torch.randn(1, 4, 256, 256)
    model = Autoencoder(64, 512)

    # >>>>>>>>>>>>>>>>>> Load pre-trained ROME weights >>>>>>>>>>>>>>>>>
    print("Load Pre-trained ROME Encoder weights ...")
    rome_dict = torch.load('data/rome.pth')
    rome_dict = {k: v for k, v in rome_dict.items() if 'autoencoder' in k}              # only keep autoencoder weights
    rome_dict = {k.replace('autoencoder.', ''): v for k, v in rome_dict.items()}        # remove autoencoder. prefix

    rome_encoder_dict = {}
    for k in rome_dict.keys():
        print("--- ROME Encoder weights: ", k, rome_dict[k].shape)
    
    model_keys = model.state_dict().keys()
    for k in model_keys:
        if k in rome_dict.keys():
            print("--- Load ROME Encoder weights: ", k, rome_dict[k].shape)
            rome_encoder_dict[k] = rome_dict[k]

    model.load_state_dict(rome_encoder_dict)

    y = model(x)

    print(model.state_dict().keys())

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    print(y.shape)
    # print(count_parameters(model) / )

