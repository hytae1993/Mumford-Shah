import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ===================================
#       Parts of the U-Net model
# ===================================
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConvDownStride(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConvLast(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels_last, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels_last
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels_last, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.double_conv(x)

class OneConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.one_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.one_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DownStride(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            DoubleConvDownStride(in_channels, out_channels)
        )

    def forward(self, x):
        return self.conv(x)



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, nearest=True):
        super().__init__()

        # if nearest, use the normal convolutions to reduce the number of channels
        if nearest:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            self.pad = nn.ReflectionPad2d(1)
            self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.pad(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpSkip(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, nearest=True):
        super().__init__()

        # if nearest, use the normal convolutions to reduce the number of channels
        if nearest:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            self.pad = nn.ReflectionPad2d(1)
            self.conv = DoubleConv(in_channels*2, out_channels, in_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels*2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels*2, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x1 = self.pad(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpSkipLast(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, nearest=True):
        super().__init__()

        # if nearest, use the normal convolutions to reduce the number of channels
        if nearest:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            self.pad = nn.ReflectionPad2d(1)
            self.conv = DoubleConvLast(in_channels*2, out_channels, in_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels*2, kernel_size=2, stride=2)
            self.conv = DoubleConvLast(in_channels*2, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x1 = self.pad(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class UpNoSkip(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, nearest=True):
        super().__init__()

        # if nearest, use the normal convolutions to reduce the number of channels
        if nearest:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            self.pad = nn.ReflectionPad2d(1)
            self.conv = DoubleConv(in_channels, out_channels, in_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels*2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x):
        x = self.up(x)
        x = self.pad(x)
        return self.conv(x)

class UpLast(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels_last, nearest=True):
        super().__init__()

        # if nearest, use the normal convolutions to reduce the number of channels
        if nearest:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            self.pad = nn.ReflectionPad2d(1)
            self.conv = DoubleConvLast(in_channels, out_channels_last, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvLast(in_channels, out_channels_last, in_channels // 2)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.pad(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UpNoSkipLast(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, nearest=True):
        super().__init__()

        # if nearest, use the normal convolutions to reduce the number of channels
        if nearest:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            self.pad = nn.ReflectionPad2d(1)
            self.conv = DoubleConvLast(in_channels, out_channels, in_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConvLast(in_channels, out_channels)


    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class LastSkip(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, nearest=True):
        super().__init__()

        # if nearest, use the normal convolutions to reduce the number of channels
        if nearest:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            self.pad = nn.ReflectionPad2d(1)
            self.conv = DoubleConvLast(in_channels*2, out_channels, in_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels*2, kernel_size=2, stride=2)
            self.conv = DoubleConvLast(in_channels*2, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.pad(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# =============================================================
#                     Two Branch Autoencoder
# =============================================================
class Encoder(nn.Module):
    def __init__(self, n_channels, n_classes, nearest=True):
        super(Encoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.nearest = nearest

        self.increase_depth = DoubleConv(n_channels, 64) # 64x64xn_ch -> 64x64X64
        self.down1 = DownStride(64, 128) # 64x64X64 -> 32x32x128

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.increase_depth(x)
        x2 = self.down1(x1)
        return x2

class DecoderSeg(nn.Module):
    def __init__(self, n_channels, n_classes, nearest=True):
        super(DecoderSeg, self).__init__()
        self.n_channels = n_channels
        self.n_classes = 3
        self.nearest = nearest

        self.decrease_depth = DoubleConv(128, 64) # 32x32x128 -> 32x32x64
        self.up1 = UpNoSkipLast(64, n_classes, nearest) # 64x64x64 + 64x64x64 -> 64x64x3
        self.adj_out_depth = OutConv(n_classes, 1) # 64x64x3 -> 64x64x1

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, latent):
        x = self.decrease_depth(latent)
        x = self.up1(x)
        logits = self.adj_out_depth(x)
        mask = torch.sigmoid(logits*1)
        return mask