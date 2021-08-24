# encoding: utf-8
"""
Alexnet backbone

"""
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ["AlexNet"]


class AlexNet(nn.Module):
    def __init__(self, in_channels=1, bn=False):
        super(AlexNet, self).__init__()
        self.features3 = nn.Sequential(
            # kernel(11, 11) -> kernel(7, 7)
            nn.Conv2d(in_channels=in_channels, out_channels=64,
                      kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )
        # padding=0 -> padding=1
        self.features4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192) if bn else nn.GroupNorm(32, 192),
            nn.ReLU(inplace=True),
        )
        self.features5 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features3(x)
        x, indices3 = self.maxpool(x)
        x = self.features4(x)
        x, indices4 = self.maxpool(x)
        x = self.features5(x)
        x, indices5 = self.maxpool(x)
        return x


if __name__ == "__main__":
    x = torch.zeros(8, 1, 200, 200)
    net = Alexnet()
    print(net)
    y = net(x)
    print()
