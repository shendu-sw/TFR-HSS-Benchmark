# -*- encoding: utf-8 -*-
"""Load Response Dataset.
"""
import os

import torch
import scipy.io as sio
import numpy as np
from torchvision.datasets import VisionDataset


TOL = 1e-14


class LoadResponse(VisionDataset):
    """Some Information about LoadResponse dataset"""

    def __init__(
        self,
        root,
        loader,
        list_path,
        load_name="u_obs",
        resp_name="u",
        layout_name="F",
        extensions=None,
        transform=None,
        target_transform=None,
        is_valid_file=None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.list_path = list_path
        self.loader = loader
        self.load_name = load_name
        self.resp_name = resp_name
        self.layout_name = layout_name
        self.extensions = extensions
        self.sample_files = make_dataset_list(
            root, list_path, extensions, is_valid_file
        )

    def __getitem__(self, index):
        path = self.sample_files[index]
        load, resp, _ = self.loader(path, self.load_name, self.resp_name)

        load[np.where(load < TOL)] = 298

        if self.transform is not None:
            load = self.transform(load)
        if self.target_transform is not None:
            resp = self.target_transform(resp)

        return load, resp

    def _layout(self):
        path = self.sample_files[0]
        _, _, layout = self.loader(
            path, self.load_name, self.resp_name, self.layout_name
        )
        return layout

    def __len__(self):
        return len(self.sample_files)


class LoadPointResponse(VisionDataset):
    """Some Information about LoadResponse dataset"""

    def __init__(
        self,
        root,
        loader,
        list_path,
        load_name="u_obs",
        resp_name="u",
        layout_name="F",
        extensions=None,
        is_valid_file=None,
    ):
        super().__init__(root)
        self.list_path = list_path
        self.loader = loader
        self.load_name = load_name
        self.resp_name = resp_name
        self.layout_name = layout_name
        self.extensions = extensions
        self.sample_files = make_dataset_list(
            root, list_path, extensions, is_valid_file
        )

    def __getitem__(self, index):
        path = self.sample_files[index]
        load, resp, layout = self.loader(
            path, self.load_name, self.resp_name, self.layout_name
        )

        return load, resp, layout

    def __len__(self):
        return len(self.sample_files)


class LoadVecResponse(VisionDataset):
    def __init__(
        self,
        root,
        loader,
        list_path,
        load_name="u_obs",
        resp_name="u",
        layout_name="F",
        div_num=4,
        extensions=None,
        transform=None,
        target_transform=None,
        is_valid_file=None,
    ):
        super().__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.loader = loader
        self.list_path = list_path
        self.load_name = load_name
        self.resp_name = resp_name
        self.layout_name = layout_name
        self.div_num = div_num
        self.sample_files = make_dataset_list(
            root, list_path, extensions, is_valid_file
        )

    def __getitem__(self, index):

        path = self.sample_files[index]
        x_context, y_context, x_target, y_target, resp = self._loader(path)

        if self.transform is not None:
            y_context = (y_context - self.transform[0]) / self.transform[1]
        else:
            pass

        if self.target_transform is not None:
            y_target = (y_target - self.target_transform[0]) / self.transform[1]
            resp = (resp - self.target_transform[0]) / self.transform[1]
        else:
            pass

        return (
            x_context,
            y_context.type(torch.FloatTensor),
            x_target,
            y_target.type(torch.FloatTensor),
            resp.type(torch.FloatTensor),
        )

    def __len__(self):
        return len(self.sample_files)

    def _loader(self, path):

        load, resp, _ = self.loader(path, self.load_name, self.resp_name)

        monitor_x, monitor_y = np.where(load > TOL)
        y_context = torch.from_numpy(load[monitor_x, monitor_y].reshape(1, -1)).float()

        monitor_x, monitor_y = monitor_x / load.shape[0], monitor_y / load.shape[1]
        x_context = torch.from_numpy(
            np.concatenate([monitor_x.reshape(-1, 1), monitor_y.reshape(-1, 1)], axis=1)
        ).float()

        x = np.linspace(0, load.shape[0] - 1, load.shape[0]).astype(int)
        y = np.linspace(1, load.shape[1] - 1, load.shape[1]).astype(int)

        x_target = None
        y_target = None
        for i in range(self.div_num):
            for j in range(self.div_num):
                x1, y1 = (
                    x[0 + i : np.size(x) : self.div_num],
                    y[0 + j : np.size(y) : self.div_num],
                )
                x1, y1 = np.meshgrid(x1, y1)
                x_target0 = (
                    torch.from_numpy(
                        np.concatenate([x1.reshape(-1, 1), y1.reshape(-1, 1)], axis=1)
                        / np.max(load.shape)
                    )
                    .float()
                    .unsqueeze(0)
                )
                y_target0 = torch.from_numpy(resp[x1, y1].reshape(1, -1)).unsqueeze(0)
                if x_target is not None:
                    x_target = torch.cat((x_target, x_target0), 0)
                else:
                    x_target = x_target0

                if y_target is not None:
                    y_target = torch.cat((y_target, y_target0), 0)
                else:
                    y_target = y_target0

        return x_context, y_context, x_target, y_target, torch.from_numpy(resp).float()

    def _layout(self):
        path = self.sample_files[0]
        _, _, layout = self.loader(
            path, self.load_name, self.resp_name, self.layout_name
        )
        return layout

    def _inputdim(self):
        path = self.sample_files[0]
        load, _, _ = self.loader(path, self.load_name, self.resp_name, self.layout_name)
        monitor_x, _ = np.where(load > TOL)
        return np.size(monitor_x)


def make_dataset(root_dir, extensions=None, is_valid_file=None):
    """make_dataset() from torchvision."""
    files = []
    root_dir = os.path.expanduser(root_dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError(
            "Both extensions and is_valid_file \
                cannot be None or not None at the same time"
        )
    if extensions is not None:
        is_valid_file = lambda x: has_allowed_extension(x, extensions)

    assert os.path.isdir(root_dir), root_dir
    for root, _, fns in sorted(os.walk(root_dir, followlinks=True)):
        for fn in sorted(fns):
            path = os.path.join(root, fn)
            if is_valid_file(path):
                files.append(path)
    return files


def make_dataset_list(root_dir, list_path, extensions=None, is_valid_file=None):
    """make_dataset() from torchvision."""
    files = []
    root_dir = os.path.expanduser(root_dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError(
            "Both extensions and is_valid_file \
                cannot be None or not None at the same time"
        )
    if extensions is not None:
        is_valid_file = lambda x: has_allowed_extension(x, extensions)

    assert os.path.isdir(root_dir), root_dir
    with open(list_path, "r") as rf:
        for line in rf.readlines():
            data_path = line.strip()
            path = os.path.join(root_dir, data_path)
            if is_valid_file(path):
                files.append(path)
    return files


def has_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)


def mat_loader(path, load_name, resp_name=None, layout_name=None):
    mats = sio.loadmat(path)
    load = mats.get(load_name)
    resp = mats.get(resp_name) if resp_name is not None else None
    layout = mats.get(layout_name) if layout_name is not None else None
    return load, resp, layout


if __name__ == "__main__":
    total_num = 50000
    with open("train" + str(total_num) + ".txt", "w") as wf:
        for idx in range(int(total_num * 0.8)):
            wf.write("Example" + str(idx) + ".mat" + "\n")
    with open("val" + str(total_num) + ".txt", "w") as wf:
        for idx in range(int(total_num * 0.8), total_num):
            wf.write("Example" + str(idx) + ".mat" + "\n")
