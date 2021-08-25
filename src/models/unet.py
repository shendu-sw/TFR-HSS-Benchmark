# encoding: utf-8
import torch
import torch.nn.functional as F
from torch import nn

from src.utils.unet_initialize import initialize_weights


__all__ = ["UNet_VGG"]


class _EncoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, dropout=False, polling=True, bn=False
    ):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        self.encode = nn.Sequential(*layers)
        self.pool = None
        if polling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, bn=False):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels)
            if bn
            else nn.GroupNorm(32, middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels)
            if bn
            else nn.GroupNorm(32, middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class UNet_VGG(nn.Module):
    def __init__(self, out_channels=1, in_channels=1, bn=False):
        super(UNet_VGG, self).__init__()
        self.enc1 = _EncoderBlock(in_channels, 64, polling=False, bn=bn)
        self.enc2 = _EncoderBlock(64, 128, bn=bn)
        self.enc3 = _EncoderBlock(128, 256, bn=bn)
        self.enc4 = _EncoderBlock(256, 512, bn=bn)
        self.polling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlock(512, 1024, 512, bn=bn)
        self.dec4 = _DecoderBlock(1024, 512, 256, bn=bn)
        self.dec3 = _DecoderBlock(512, 256, 128, bn=bn)
        self.dec2 = _DecoderBlock(256, 128, 64, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc4))
        dec4 = self.dec4(
            torch.cat(
                [
                    F.interpolate(
                        center, enc4.size()[-2:], mode="bilinear", align_corners=True
                    ),
                    enc4,
                ],
                1,
            )
        )
        dec3 = self.dec3(torch.cat([dec4, enc3], 1))
        dec2 = self.dec2(torch.cat([dec3, enc2], 1))
        dec1 = self.dec1(torch.cat([dec2, enc1], 1))
        final = self.final(dec1)
        return final


if __name__ == "__main__":
    model = UNet_VGG(in_channels=1, out_channels=1)
    print(model)
    x = torch.randn(1, 1, 200, 200)
    with torch.no_grad():
        y = model(x)
        print(y.shape)
