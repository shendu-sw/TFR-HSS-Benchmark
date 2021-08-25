# -*- encoding: utf-8 -*-
"""
Desc      :   Transforms.
"""
# File    :   np_transforms.py
# Time    :   2020/04/06 17:24:54
# Author  :   Zweien
# Contact :   278954153@qq.com

import torch
from torchvision import transforms
from torch.nn.functional import interpolate


class ToTensor:
    """Transform np.array to torch.tensor
    Args:
        add_dim (bool, optional): add first dim. Defaults to True.
        type_ (torch.dtype, optional): dtype of the tensor. Defaults to tensor.torch.float32.
    Returns:
        torch.tensor: tensor
    """

    def __init__(self, add_dim=True, type_=torch.float32):

        self.add_dim = add_dim
        self.type = type_

    def __call__(self, x):
        if self.add_dim:
            return torch.tensor(x, dtype=self.type).unsqueeze(0)
        return torch.tensor(x, dtype=self.type)


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        x_tensor = torch.tensor(x)
        x_dim = x_tensor.dim()
        for _ in range(4 - x_dim):
            x_tensor = x_tensor.unsqueeze(0)
        x_resize = interpolate(x_tensor, size=self.size)
        for _ in range(4 - x_dim):
            x_resize = x_resize.squeeze(0)
        return x_resize.numpy()


class Lambda(transforms.Lambda):
    pass


class Compose(transforms.Compose):
    pass


class Normalize(transforms.Normalize):
    pass
