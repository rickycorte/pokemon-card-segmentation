#
# Simple u-net like baseline model not pretrained
#

import torch
import torch.nn as nn

class DConvGroup(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.activation = nn.SiLU()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)

        return self.maxpool(x), x


class UConvGroup(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upscale=True):
        super().__init__()

        self.upscale = upscale

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.activation = nn.SiLU()

        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, skip, x):
        #print(f"{self.conv1.in_channels} =  s: {skip.shape} + x {x.shape}")
        x = torch.concat((skip, x), dim=1)

        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        if self.upscale:
            x = self.upsample(x)

        return x

class UNetBaseline(nn.Module):
    def __init__(self, in_depth, out_depth:int, depth_scale = 1):
        super().__init__()

        self.in_channels = in_depth
        self.out_channels = out_depth

        bsz = int(8 * depth_scale)

        self.d1 = DConvGroup(in_depth, bsz)
        self.d2 = DConvGroup(bsz * 1, bsz * 2)
        self.d3 = DConvGroup(bsz * 2, bsz * 4)
        self.d4 = DConvGroup(bsz * 4, bsz * 8)

        self.mconv = nn.Conv2d(bsz * 8, bsz * 16, kernel_size=3, padding=1)
        self.mconv2 = nn.Conv2d(bsz * 16, bsz * 8, kernel_size=3, padding=1)
        self.mconv_act = nn.SiLU()

        self.up = nn.Upsample(scale_factor=2)

        self.u1 = UConvGroup(bsz * 16, bsz * 4)
        self.u2 = UConvGroup(bsz * 8, bsz * 2)
        self.u3 = UConvGroup(bsz * 4, bsz * 1)
        self.u4 = UConvGroup(bsz * 2, bsz * 1, upscale=False)

        self.out = nn.Conv2d(bsz * 1, out_depth, kernel_size=3, padding=1)

    def forward(self, x):
        x, s1 = self.d1(x)
        x, s2 = self.d2(x)
        x, s3 = self.d3(x)
        x, s4 = self.d4(x)

        x = self.mconv(x)
        x = self.mconv_act(x)
        x = self.mconv2(x)
        x = self.mconv_act(x)
        x = self.up(x)

        x = self.u1(s4, x)
        x = self.u2(s3, x)
        x = self.u3(s2, x)
        x = self.u4(s1, x)

        return self.out(x)



