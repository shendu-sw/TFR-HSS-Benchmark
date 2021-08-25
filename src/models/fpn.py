# encoding: utf-8
import torch.nn as nn
import torch.nn.functional as F

from src.utils.model_init import weights_init
from .util.backbone import *


__all__ = [
    "FPN_ResNet18",
    "FPN_ResNet34",
    "FPN_ResNet50",
    "FPN_ResNet101",
    "FPN_ResNet152",
]


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, size):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, size=skip.size()[-2:], mode="bilinear", align_corners=True)
        skip = self.skip_conv(skip)

        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        self.blocks = [
            Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))
        ]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                self.blocks.append(
                    Conv3x3GNReLU(out_channels, out_channels, upsample=True)
                )

        self.blocks_name = []
        for i, block in enumerate(self.blocks):
            self.add_module("Block_{}".format(i), block)
            self.blocks_name.append("Block_{}".format(i))

    def forward(self, x, sizes=[]):
        for i, block_name in enumerate(self.blocks_name):
            x = getattr(self, block_name)(x, sizes[i])
        return x


class FPN_ResNet(nn.Module):
    def __init__(
        self,
        backbone,
        encoder_channels,
        pyramid_channels=256,
        segmentation_channels=128,
        final_upsampling=4,
        final_channels=1,
        dropout=0.2,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone.apply(weights_init)
        self.final_upsampling = final_upsampling
        self.conv1 = nn.Conv2d(
            encoder_channels[0], pyramid_channels, kernel_size=(1, 1)
        )

        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.s5 = SegmentationBlock(
            pyramid_channels, segmentation_channels, n_upsamples=3
        )
        self.s4 = SegmentationBlock(
            pyramid_channels, segmentation_channels, n_upsamples=2
        )
        self.s3 = SegmentationBlock(
            pyramid_channels, segmentation_channels, n_upsamples=1
        )
        self.s2 = SegmentationBlock(
            pyramid_channels, segmentation_channels, n_upsamples=0
        )

        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.final_conv = nn.Conv2d(
            segmentation_channels, final_channels, kernel_size=1, padding=0
        )

    def forward(self, x):
        x = self.backbone(x)

        _, c2, c3, c4, c5 = x

        p5 = self.conv1(c5)
        p4 = self.p4([p5, c4])
        p3 = self.p3([p4, c3])
        p2 = self.p2([p3, c2])

        s5 = self.s5(p5, sizes=[c4.size()[-2:], c3.size()[-2:], c2.size()[-2:]])
        s4 = self.s4(p4, sizes=[c3.size()[-2:], c2.size()[-2:]])
        s3 = self.s3(p3, sizes=[c2.size()[-2:]])
        s2 = self.s2(p2, sizes=[c2.size()[-2:]])

        # x = torch.cat([s5, s4, s3, s2], dim=1)
        x = s5 + s4 + s3 + s2

        x = self.dropout(x)
        x = self.final_conv(x)

        if self.final_upsampling is not None and self.final_upsampling > 1:
            x = F.interpolate(
                x,
                scale_factor=self.final_upsampling,
                mode="bilinear",
                align_corners=True,
            )
        return x


def FPN_ResNet18(in_channels=1, **kwargs):
    """FPN with ResNet18 as backbone"""
    backbone = resnet18(in_channels=in_channels)
    model = FPN_ResNet(backbone, encoder_channels=[512, 256, 128, 64], **kwargs)
    return model


def FPN_ResNet34(in_channels=1, **kwargs):
    """FPN with ResNet18 as backbone"""
    backbone = resnet34(in_channels=in_channels)
    model = FPN_ResNet(backbone, encoder_channels=[512, 256, 128, 64], **kwargs)
    return model


def FPN_ResNet50(in_channels=1, **kwargs):
    """FPN with ResNet50 as backbone"""
    backbone = resnet50(in_channels=in_channels)
    model = FPN_ResNet(backbone, encoder_channels=[2048, 1024, 512, 256], **kwargs)
    return model


def FPN_ResNet101(in_channels=1, **kwargs):
    """FPN with ResNet101 as backbone"""
    backbone = resnet101(in_channels=in_channels)
    model = FPN_ResNet(backbone, encoder_channels=[2048, 1024, 512, 256], **kwargs)
    return model


def FPN_ResNet152(in_channels=1, **kwargs):
    """FPN with ResNet101 as backbone"""
    backbone = resnet152(in_channels=in_channels)
    model = FPN_ResNet(backbone, encoder_channels=[2048, 1024, 512, 256], **kwargs)
    return model
