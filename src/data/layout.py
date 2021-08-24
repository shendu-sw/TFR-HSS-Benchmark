# -*- encoding: utf-8 -*-
"""Layout dataset
"""
import os
from .loadresponse import LoadResponse, LoadPointResponse, LoadVecResponse, mat_loader


class LayoutDataset(LoadResponse):
    """Layout dataset (mutiple files) generated by 'layout-generator'.
    """

    def __init__(
        self,
        root,
        list_path=None,
        train=True,
        transform=None,
        target_transform=None,
        load_name="u_obs",
        resp_name="u",
    ):
        test_name = os.path.splitext(os.path.basename(list_path))[0]
        subdir = os.path.join("train", "train") \
            if train else os.path.join("test", test_name)

        # find the path of the list of train/test samples
        list_path = os.path.join(root, list_path)

        # find the root path of the samples
        root = os.path.join(root, subdir)

        super().__init__(
            root,
            mat_loader,
            list_path,
            load_name=load_name,
            resp_name=resp_name,
            extensions="mat",
            transform=transform,
            target_transform=target_transform,
        )


class LayoutPointDataset(LoadPointResponse):

    def __init__(
        self,
        root,
        list_path=None,
        train=True,
        load_name="u_obs",
        resp_name="u",
        layout_name="F",
    ):
        test_name = os.path.splitext(os.path.basename(list_path))[0]
        subdir = os.path.join("train", "train") \
            if train else os.path.join("test", test_name)

        # find the path of the list of train/test samples
        list_path = os.path.join(root, list_path)

        # find the root path of the samples
        root = os.path.join(root, subdir)

        super().__init__(
            root,
            mat_loader,
            list_path,
            load_name=load_name,
            resp_name=resp_name,
            layout_name=layout_name,
            extensions="mat",
        )


class LayoutVecDataset(LoadVecResponse):
    """Layout dataset (mutiple files) generated by 'layout-generator'.
    """

    def __init__(
        self,
        root,
        list_path=None,
        train=True,
        transform=None,
        div_num=4,
        target_transform=None,
        load_name="u_obs",
        resp_name="u",
    ):
        test_name = os.path.splitext(os.path.basename(list_path))[0]
        subdir = os.path.join("train", "train") \
            if train else os.path.join("test", test_name)

        # find the path of the list of train/test samples
        list_path = os.path.join(root, list_path)

        # find the root path of the samples
        root = os.path.join(root, subdir)

        super().__init__(
            root,
            mat_loader,
            list_path,
            load_name=load_name,
            resp_name=resp_name,
            extensions="mat",
            div_num=div_num,
            transform=transform,
            target_transform=target_transform,
        )