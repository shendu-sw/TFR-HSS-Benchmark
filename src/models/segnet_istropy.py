# encoding: utf-8
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from .util.backbone import *


__all__ = ["SegNet_VGG", "SegNet_VGG_GN", "SegNet_AlexNet", "SegNet_ResNet18",
           "SegNet_ResNet50", "SegNet_ResNet101", "SegNet_ResNet34", "SegNet_ResNet152"]


# required class for decoder of SegNet_ResNet
class DecoderBottleneck(nn.Module):

    def __init__(self, in_channels):
        super(DecoderBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.conv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4,
                                        kernel_size=2, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels // 4)
        self.conv3 = nn.Conv2d(in_channels // 4, in_channels // 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2,
                               kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(in_channels // 2))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# required class for decoder of SegNet_ResNet
class LastBottleneck(nn.Module):

    def __init__(self, in_channels):
        super(LastBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.conv2 = nn.Conv2d(in_channels // 4, in_channels // 4,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels // 4)
        self.conv3 = nn.Conv2d(in_channels // 4, in_channels // 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 4))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# required class for decoder of SegNet_ResNet
class DecoderBasicBlock(nn.Module):

    def __init__(self, in_channels):
        super(DecoderBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.conv2 = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                        kernel_size=2, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2,
                               kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(in_channels // 2))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class LastBasicBlock(nn.Module):

    def __init__(self, in_channels):
        super(LastBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class SegNet_VGG(nn.Module):

    def __init__(self, out_channels=1, in_channels=1, pretrained=False):
        super(SegNet_VGG, self).__init__()
        vgg_bn = vgg16_bn(pretrained=pretrained)
        encoder = list(vgg_bn.features.children())

        # Adjust the input size
        if in_channels != 3:
            encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)

        # Encoder, VGG without any maxpooling
        self.stage1_encoder = nn.Sequential(*encoder[:6])
        self.stage2_encoder = nn.Sequential(*encoder[7:13])
        self.stage3_encoder = nn.Sequential(*encoder[14:23])
        self.stage4_encoder = nn.Sequential(*encoder[24:33])
        self.stage5_encoder = nn.Sequential(*encoder[34:-1])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Decoder, same as the encoder but reversed, maxpool will not be used
        decoder = encoder
        decoder = [i for i in list(reversed(decoder)) if not isinstance(i, nn.MaxPool2d)]
        # Replace the last conv layer
        decoder[-1] = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # When reversing, we also reversed conv->batchN->relu, correct it
        decoder = [item for i in range(0, len(decoder), 3)
                   for item in decoder[i:i + 3][::-1]]
        # Replace some conv layers & batchN after them
        for i, module in enumerate(decoder):
            if isinstance(module, nn.Conv2d):
                if module.in_channels != module.out_channels:
                    decoder[i + 1] = nn.BatchNorm2d(module.in_channels)
                    decoder[i] = nn.Conv2d(module.out_channels, module.in_channels,
                                           kernel_size=3, stride=1, padding=1)

        self.stage1_decoder = nn.Sequential(*decoder[0:9])
        self.stage2_decoder = nn.Sequential(*decoder[9:18])
        self.stage3_decoder = nn.Sequential(*decoder[18:27])
        self.stage4_decoder = nn.Sequential(*decoder[27:33])
        self.stage5_decoder = nn.Sequential(*decoder[33:],
                                            nn.Conv2d(64, out_channels,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1)
                                            )
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self._initialize_weights(self.stage1_decoder, self.stage2_decoder, self.stage3_decoder,
                                 self.stage4_decoder, self.stage5_decoder)

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    def forward(self, x):
        # Encoder
        x = self.stage1_encoder(x)
        x1_size = x.size()
        x, indices1 = self.pool(x)

        x = self.stage2_encoder(x)
        x2_size = x.size()
        x, indices2 = self.pool(x)

        x = self.stage3_encoder(x)
        x3_size = x.size()
        x, indices3 = self.pool(x)

        x = self.stage4_encoder(x)
        x4_size = x.size()
        x, indices4 = self.pool(x)

        x = self.stage5_encoder(x)
        x5_size = x.size()
        x, indices5 = self.pool(x)

        # Decoder
        x = self.unpool(x, indices=indices5, output_size=x5_size)
        x = self.stage1_decoder(x)

        x = self.unpool(x, indices=indices4, output_size=x4_size)
        x = self.stage2_decoder(x)

        x = self.unpool(x, indices=indices3, output_size=x3_size)
        x = self.stage3_decoder(x)

        x = self.unpool(x, indices=indices2, output_size=x2_size)
        x = self.stage4_decoder(x)

        x = self.unpool(x, indices=indices1, output_size=x1_size)
        x = self.stage5_decoder(x)

        return x


class SegNet_VGG_GN(nn.Module):

    def __init__(self, out_channels=1, in_channels=3, pretrained=False):
        super(SegNet_VGG_GN, self).__init__()
        vgg_bn = vgg16_bn(pretrained=pretrained)
        encoder = list(vgg_bn.features.children())

        # Adjust the input size
        if in_channels != 3:
            encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)

        #
        for i in range(len(encoder)):
            if isinstance(encoder[i], nn.BatchNorm2d):
                encoder[i] = nn.GroupNorm(32, encoder[i].num_features)

        # Encoder, VGG without any maxpooling
        self.stage1_encoder = nn.Sequential(*encoder[:6])
        self.stage2_encoder = nn.Sequential(*encoder[7:13])
        self.stage3_encoder = nn.Sequential(*encoder[14:23])
        self.stage4_encoder = nn.Sequential(*encoder[24:33])
        self.stage5_encoder = nn.Sequential(*encoder[34:-1])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Decoder, same as the encoder but reversed, maxpool will not be used
        decoder = encoder
        decoder = [i for i in list(reversed(decoder)) if not isinstance(i, nn.MaxPool2d)]
        # Replace the last conv layer
        decoder[-1] = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # When reversing, we also reversed conv->batchN->relu, correct it
        decoder = [item for i in range(0, len(decoder), 3)
                   for item in decoder[i:i + 3][::-1]]
        # Replace some conv layers & batchN after them
        for i, module in enumerate(decoder):
            if isinstance(module, nn.Conv2d):
                if module.in_channels != module.out_channels:
                    decoder[i + 1] = nn.GroupNorm(32, module.in_channels)
                    decoder[i] = nn.Conv2d(module.out_channels, module.in_channels,
                                           kernel_size=3, stride=1, padding=1)

        self.stage1_decoder = nn.Sequential(*decoder[0:9])
        self.stage2_decoder = nn.Sequential(*decoder[9:18])
        self.stage3_decoder = nn.Sequential(*decoder[18:27])
        self.stage4_decoder = nn.Sequential(*decoder[27:33])
        self.stage5_decoder = nn.Sequential(*decoder[33:], nn.Conv2d(64,
                                                                     out_channels,
                                                                     kernel_size=3,
                                                                     stride=1,
                                                                     padding=1))
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self._initialize_weights(self.stage1_decoder, self.stage2_decoder, self.stage3_decoder,
                                 self.stage4_decoder, self.stage5_decoder)

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    def forward(self, x):
        # Encoder
        x = self.stage1_encoder(x)
        x1_size = x.size()
        x, indices1 = self.pool(x)

        x = self.stage2_encoder(x)
        x2_size = x.size()
        x, indices2 = self.pool(x)

        x = self.stage3_encoder(x)
        x3_size = x.size()
        x, indices3 = self.pool(x)

        x = self.stage4_encoder(x)
        x4_size = x.size()
        x, indices4 = self.pool(x)

        x = self.stage5_encoder(x)
        x5_size = x.size()
        x, indices5 = self.pool(x)

        # Decoder
        x = self.unpool(x, indices=indices5, output_size=x5_size)
        x = self.stage1_decoder(x)

        x = self.unpool(x, indices=indices4, output_size=x4_size)
        x = self.stage2_decoder(x)

        x = self.unpool(x, indices=indices3, output_size=x3_size)
        x = self.stage3_decoder(x)

        x = self.unpool(x, indices=indices2, output_size=x2_size)
        x = self.stage4_decoder(x)

        x = self.unpool(x, indices=indices1, output_size=x1_size)
        x = self.stage5_decoder(x)

        return x


class SegNet_AlexNet(nn.Module):

    def __init__(self, out_channels=1, in_channels=1, bn=False):
        super(SegNet_AlexNet, self).__init__()
        self.stage3_encoder = nn.Sequential(
            # kernel(11, 11) -> kernel(7, 7)
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            # padding=0 -> padding=1
        )
        self.stage4_encoder = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192) if bn else nn.GroupNorm(32, 192),
            nn.ReLU(inplace=True),
        )
        self.stage5_encoder = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384) if bn else nn.GroupNorm(32, 384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if bn else nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.stage5_decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if bn else nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384) if bn else nn.GroupNorm(32, 384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192) if bn else nn.GroupNorm(32, 192),
            nn.ReLU(inplace=True),
        )
        self.stage4_decoder = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )
        self.stage3_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, bias=False),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, bias=False),
            nn.Conv2d(64, out_channels, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x3 = self.stage3_encoder(x)
        x3_size = x3.size()
        x3, indices3 = self.maxpool(x3)
        x4 = self.stage4_encoder(x3)
        x4_size = x4.size()
        x4, indices4 = self.maxpool(x4)
        x5 = self.stage5_encoder(x4)
        x5_size = x5.size()
        x5, indices5 = self.maxpool(x5)

        out = self.unpool(x5, indices=indices5, output_size=x5_size)
        out = self.stage5_decoder(out)
        out = self.unpool(out, indices=indices4, output_size=x4_size)
        out = self.stage4_decoder(out)
        out = self.unpool(out, indices=indices3, output_size=x3_size)
        out = self.stage3_decoder(out)
        return out


class SegNet_ResNet(nn.Module):

    def __init__(self, backbone, out_channels=1, is_bottleneck=False, in_channels=1):
        super(SegNet_ResNet, self).__init__()
        resnet_backbone = backbone
        encoder = list(resnet_backbone.children())
        if in_channels != 3:
            encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        encoder[3].return_indices = True

        # Encoder
        self.first_conv = nn.Sequential(*encoder[:4])
        resnet_blocks = list(resnet_backbone.children())[4:]
        self.encoder = nn.Sequential(*resnet_blocks)

        # Decoder
        resnet_r_blocks = list(resnet_backbone.children())[4:][::-1]
        decoder = []
        if is_bottleneck:
            channels = (2048, 1024, 512)
        else:
            channels = (512, 256, 128)
        for i, block in enumerate(resnet_r_blocks[:-1]):
            new_block = list(block.children())[::-1][:-1]
            decoder.append(nn.Sequential(*new_block,
                                         DecoderBottleneck(channels[i])
                                         if is_bottleneck else DecoderBasicBlock(channels[i])))
        new_block = list(resnet_r_blocks[-1].children())[::-1][:-1]
        decoder.append(nn.Sequential(*new_block,
                                     LastBottleneck(256)
                                     if is_bottleneck else LastBasicBlock(64)))

        self.decoder = nn.Sequential(*decoder)
        self.last_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, bias=False),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        inputsize = x.size()

        # Encoder
        x, indices = self.first_conv(x)
        x = self.encoder(x)

        # Decoder
        x = self.decoder(x)
        h_diff = ceil((x.size()[2] - indices.size()[2]) / 2)
        w_diff = ceil((x.size()[3] - indices.size()[3]) / 2)
        if indices.size()[2] % 2 == 1:
            x = x[:, :, h_diff:x.size()[2] - (h_diff - 1),
                w_diff: x.size()[3] - (w_diff - 1)]
        else:
            x = x[:, :, h_diff:x.size()[2] - h_diff, w_diff: x.size()[3] - w_diff]

        x = F.max_unpool2d(x, indices, kernel_size=2, stride=2)
        x = self.last_conv(x)

        if inputsize != x.size():
            h_diff = (x.size()[2] - inputsize[2]) // 2
            w_diff = (x.size()[3] - inputsize[3]) // 2
            x = x[:, :, h_diff:x.size()[2] - h_diff, w_diff: x.size()[3] - w_diff]
            if h_diff % 2 != 0: x = x[:, :, :-1, :]
            if w_diff % 2 != 0: x = x[:, :, :, :-1]

        return x


def SegNet_ResNet18(in_channels=1, out_channels=1, **kwargs):
    """
    Construct SegNet based on ResNet18 model.

    """
    backbone_net = resnet18()
    model = SegNet_ResNet(backbone_net, out_channels=out_channels, is_bottleneck=False,
                          in_channels=in_channels, **kwargs)
    return model


def SegNet_ResNet34(in_channels=1, out_channels=1, **kwargs):
    """
    Construct SegNet based on ResNet18 model.

    """
    backbone_net = resnet34()
    model = SegNet_ResNet(backbone_net, out_channels=out_channels, is_bottleneck=False,
                          in_channels=in_channels, **kwargs)
    return model


def SegNet_ResNet50(in_channels=1, out_channels=1, **kwargs):
    """
    Construct SegNet based on ResNet50 model.

    """
    backbone_net = resnet50()
    model = SegNet_ResNet(backbone_net, out_channels=out_channels, is_bottleneck=True,
                          in_channels=in_channels, **kwargs)
    return model


def SegNet_ResNet101(in_channels=1, out_channels=1, **kwargs):
    """
    Construct SegNet based on ResNet101 model.

    """
    backbone_net = resnet101()
    model = SegNet_ResNet(backbone_net, out_channels=out_channels, is_bottleneck=True,
                          in_channels=in_channels, **kwargs)
    return model


def SegNet_ResNet152(in_channels=1, out_channels=1, **kwargs):
    """
    Construct SegNet based on ResNet101 model.

    """
    backbone_net = resnet101()
    model = SegNet_ResNet(backbone_net, out_channels=out_channels, is_bottleneck=True,
                          in_channels=in_channels, **kwargs)
    return model


if __name__ == '__main__':
    model = SegNet_AlexNet(in_channels=1, out_channels=1)
    print(model)
    x = torch.randn(1, 1, 200, 200)
    with torch.no_grad():
        y = model(x)
        print(y.shape)