# encoding: utf-8
import torch
from torch import nn
from torch.nn import functional as F

from .util.backbone import *


__all__ = [
    "FCN_VGG", "FCN_AlexNet", "FCN_ResNet18", "FCN_ResNet34",
    "FCN_ResNet50", "FCN_ResNet101", "FCN_ResNet152",
]


class Conv3x3GNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, size):
        if self.upsample:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        x = self.block(x)
        return x


class FCN_VGG(nn.Module):

    def __init__(self, inter_channels=256, in_channels=1, bn=False):
        super(FCN_VGG, self).__init__()
        vgg = vgg16()
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        if in_channels != 3:
            features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        features_temp = []
        if not bn:
            for i in range(len(features)):
                features_temp.append(features[i])
                if isinstance(features[i], nn.Conv2d):
                    features_temp.append(nn.GroupNorm(32, features[i].out_channels))

        self.features3 = nn.Sequential(*features[:17])
        self.features4 = nn.Sequential(*features[17: 24])
        self.features5 = nn.Sequential(*features[24:])

        self.score_pool3 = nn.Conv2d(256, inter_channels, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, inter_channels, kernel_size=1)

        fc6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        fc7 = nn.Conv2d(512, 512, kernel_size=1)
        score_fr = nn.Conv2d(512, inter_channels, kernel_size=1)

        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True), score_fr
        )
        self.upscore2 = Conv3x3GNReLU(inter_channels, inter_channels, upsample=True)
        self.upscore_pool4 = Conv3x3GNReLU(inter_channels, inter_channels, upsample=True)
        self.final_conv = nn.Conv2d(inter_channels, 1, kernel_size=1)

    def forward(self, x):
        pool3 = self.features3(x)
        pool4 = self.features4(pool3)
        pool5 = self.features5(pool4)

        score_fr = self.score_fr(pool5)
        upscore2 = self.upscore2(score_fr, pool4.size()[-2:])

        score_pool4 = self.score_pool4(pool4)
        upscore_pool4 = self.upscore_pool4(score_pool4 + upscore2, pool3.size()[-2:])

        score_pool3 = self.score_pool3(pool3)
        upscore8 = F.interpolate(self.final_conv(score_pool3 + upscore_pool4), x.size()[-2:], mode='bilinear', align_corners=True)
        return upscore8


class FCN_AlexNet(nn.Module):

    def __init__(self, inter_channels=256, in_channels=1):
        super(FCN_AlexNet, self).__init__()
        self.alexnet = AlexNet(in_channels=in_channels)

        self.score_pool3 = nn.Conv2d(64, inter_channels, kernel_size=1)
        self.score_pool4 = nn.Conv2d(192, inter_channels, kernel_size=1)

        fc6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        fc7 = nn.Conv2d(512, 512, kernel_size=1)
        score_fr = nn.Conv2d(512, inter_channels, kernel_size=1)

        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True), score_fr
        )
        self.upscore2 = Conv3x3GNReLU(inter_channels, inter_channels, upsample=True)
        self.upscore_pool4 = Conv3x3GNReLU(inter_channels, inter_channels, upsample=True)
        self.final_conv = nn.Conv2d(inter_channels, 1, kernel_size=1)

    def forward(self, x):
        pool3 = self.alexnet.features3(x)
        pool4 = self.alexnet.features4(pool3)
        pool5 = self.alexnet.features5(pool4)

        score_fr = self.score_fr(pool5)
        upscore2 = self.upscore2(score_fr, pool4.size()[-2:])

        score_pool4 = self.score_pool4(pool4)
        upscore_pool4 = self.upscore_pool4(score_pool4 + upscore2, pool3.size()[-2:])

        score_pool3 = self.score_pool3(pool3)
        upscore8 = F.interpolate(self.final_conv(score_pool3 + upscore_pool4), x.size()[-2:],
                                 mode='bilinear', align_corners=True)
        return upscore8


class FCN_ResNet(nn.Module):

    def __init__(self, backbone, inter_channels=256):
        super(FCN_ResNet, self).__init__()
        self.backbone = backbone

        self.score_pool3 = nn.Conv2d(backbone.layer2[0].downsample[1].num_features,
                                     inter_channels, kernel_size=1)
        self.score_pool4 = nn.Conv2d(backbone.layer3[0].downsample[1].num_features,
                                     inter_channels, kernel_size=1)

        fc6 = nn.Conv2d(backbone.layer4[0].downsample[1].num_features,
                        512, kernel_size=3, padding=1)
        fc7 = nn.Conv2d(512, 512, kernel_size=1)
        score_fr = nn.Conv2d(512, inter_channels, kernel_size=1)
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True), score_fr
        )
        self.upscore2 = Conv3x3GNReLU(inter_channels, inter_channels, upsample=True)
        self.upscore_pool4 = Conv3x3GNReLU(inter_channels, inter_channels, upsample=True)
        self.final_conv = nn.Conv2d(inter_channels, 1, kernel_size=1)

    def forward(self, x):
        _, _, pool3, pool4, pool5 = self.backbone(x)

        score_fr = self.score_fr(pool5)
        upscore2 = self.upscore2(score_fr, pool4.size()[-2:])

        score_pool4 = self.score_pool4(pool4)
        upscore_pool4 = self.upscore_pool4(score_pool4 + upscore2, pool3.size()[-2:])

        score_pool3 = self.score_pool3(pool3)
        upscore8 = F.interpolate(self.final_conv(score_pool3 + upscore_pool4), x.size()[-2:], mode='bilinear', align_corners=True)
        return upscore8


def FCN_ResNet18(in_channels=1, **kwargs):
    """
    Constructs FCN based on ResNet18 model.

    """
    backbone_net = resnet18(in_channels=in_channels)
    model = FCN_ResNet(backbone_net, **kwargs)
    return model


def FCN_ResNet34(in_channels=1, **kwargs):
    """
    Constructs FCN based on ResNet18 model.

    """
    backbone_net = resnet34(in_channels=in_channels)
    model = FCN_ResNet(backbone_net, **kwargs)
    return model


def FCN_ResNet50(in_channels=1, **kwargs):
    """
    Constructs FCN based on ResNet50 model.

    """
    backbone_net = resnet50(in_channels=in_channels)
    model = FCN_ResNet(backbone_net, **kwargs)
    return model


def FCN_ResNet101(in_channels=1, **kwargs):
    """
    Constructs FCN based on ResNet101 model.

    """
    backbone_net = resnet101(in_channels=in_channels)
    model = FCN_ResNet(backbone_net, **kwargs)
    return model


def FCN_ResNet152(in_channels=1, **kwargs):
    """
    Constructs FCN based on ResNet18 model.

    """
    backbone_net = resnet152(in_channels=in_channels)
    model = FCN_ResNet(backbone_net, **kwargs)
    return model


if __name__ == '__main__':
    model = FCN_AlexNet(in_channels=1, inter_channels=128)
    x = torch.randn(1, 1, 200, 200)
    with torch.no_grad():
        y = model(x)
        print(y.shape)