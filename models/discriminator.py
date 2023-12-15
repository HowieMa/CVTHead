
from torch import nn
import torch.nn.functional as F
import torch


class DownBlock2d(nn.Module):
    """
    Simple block for processing video (encoder).
    """

    def __init__(self, in_features, out_features, norm=False, kernel_size=4, pool=False, sn=False):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size)

        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

        if norm:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)
        else:
            self.norm = None
        self.pool = pool

    def forward(self, x):
        out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        if self.pool:
            out = F.avg_pool2d(out, (2, 2))
        return out


class Discriminator(nn.Module):
    """
    Discriminator similar to Pix2Pix
    """
    def __init__(self, num_channels=3, block_expansion=32, num_blocks=4, max_features=512, sn=False, use_kp=False):
        super(Discriminator, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(num_channels + 3 * use_kp if i == 0 else min(max_features, block_expansion * (2 ** i)),
                            min(max_features, block_expansion * (2 ** (i + 1))),
                            norm=(i != 0), kernel_size=4, pool=(i != num_blocks - 1), sn=sn))

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)
        self.use_kp = use_kp

    def forward(self, x, kp=None):
        feature_maps = []
        out = x
        if self.use_kp:
            bs, _, h1, w1 = kp.shape
            bs, C, h2, w2 = out.shape
            if h1 != h2 or w1 != w2:
                kp = F.interpolate(kp, size=(h2, w2), mode='bilinear')
            out = torch.cat([out, kp], dim=1)

        for down_block in self.down_blocks:
            feature_maps.append(down_block(out))
            out = feature_maps[-1]
        prediction_map = self.conv(out)

        return feature_maps, prediction_map


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale (scale) discriminator
    """
    def __init__(self, scales=([1]), **kwargs):
        super(MultiScaleDiscriminator, self).__init__()
        self.scales = scales        # [1]
        discs = {}
        for scale in scales:
            discs[str(scale).replace('.', '-')] = Discriminator(**kwargs)
        self.discs = nn.ModuleDict(discs)

    def forward(self, x, kp=None):
        # x, (B, 3, 256, 256)

        features = []
        scores = []
        for scale, disc in self.discs.items():
            scale = str(scale).replace('-', '.')
            feature_maps, prediction_map = disc(x, kp)
            features.append(feature_maps)           # [ (B, C, H, W), .., (B, C, H, W) ]
            scores.append(prediction_map)           # (B, 1, H, W)

        return features, scores


if __name__ == "__main__":
    x = torch.randn(2, 3, 256, 256)
    x_gt = torch.randn(2, 3, 256, 256)

    net_D = MultiScaleDiscriminator()

    features, scores = net_D(x)
    print(len(features), len(features[0]), len(scores), scores[0].shape)


