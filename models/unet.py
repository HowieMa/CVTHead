import torch
import torch.nn as nn
from torch.nn import functional as F


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        
        # two conv layers
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        else:
            self.pool = nn.Identity()

        self.activation = nn.ReLU(inplace=True)
        self.norm0 = nn.BatchNorm2d(self.out_channels)
        self.norm1 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        y = self.activation(self.norm0(self.conv1(x)))
        y = self.activation(self.norm1(self.conv2(y)))
        before_pool = y
        y = self.pool(y)
        return y, before_pool


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)


        self.conv1 = nn.Conv2d(2*self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.activation = nn.ReLU(inplace=True)

        self.norm0 = nn.BatchNorm2d(self.out_channels)
        self.norm1 = nn.BatchNorm2d(self.out_channels)
        self.norm2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, enc, dec):
        """ Forward pass
        Arguments:
            enc: Tensor from the encoder pathway
            dec: Tensor from the decoder pathway (to be upconv'd)
        """
        # up sample
        updec = self.upsample(dec)
        updec = self.activation(self.norm0(self.upconv(updec)))

        mrg = torch.cat((updec, enc), 1)

        y = self.activation(self.norm1(self.conv1(mrg)))
        y = self.activation(self.norm2(self.conv2(y)))
        return y


class UNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 2,
            n_blocks: int = 3,
            start_filts: int = 32,
    ):
        super().__init__()
        # Hyper Parameters
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.n_blocks = n_blocks

        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        
        # create the encoder pathway and add to a list
        for i in range(n_blocks):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2**i)
            pooling = True if i < n_blocks - 1 else False

            down_conv = DownConv(ins, outs,pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires n_blocks-1 blocks
        for i in range(n_blocks - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs)
            self.up_convs.append(up_conv)

        self.conv_final = nn.Conv2d(outs, self.out_channels, kernel_size=1)

        self.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            if getattr(m, 'bias') is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False):
        encoder_outs = []
        # Encoder pathway, save outputs for merging
        i = 0
        for module in self.down_convs:
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
            i += 1

        features = [x]
        # Decoding by UpConv and merging with saved outputs of encoder
        i = 0
        for module in self.up_convs:
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
            i += 1
            features.append(x)

        # >>>>>>>>>>>>>>> final output >>>>>>>>>>>>>
        x = self.conv_final(x)
        if return_features:
            return x, features
        else:
            return x
      

if __name__ == "__main__":
    net = UNet(in_channels=3, out_channels=20, n_blocks=5, start_filts=32)

    x = torch.randn(1, 3, 256, 256)
    y, features = net(x,return_features=True)
    
    for f in features:
        print(f.shape)

