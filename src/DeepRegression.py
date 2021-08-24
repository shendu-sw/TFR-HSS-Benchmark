# encoding: utf-8
import math
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt
import numpy as np

from src.data.layout import LayoutDataset, LayoutVecDataset
import src.utils.np_transforms as transforms
import src.models as models


class Model(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.read_vec_data()
        self.input_dim = self.data_info._inputdim()
        self.output_dim = np.square(int(200 / self.hparams.div_num))
        self._build_model()
        self.criterion = nn.L1Loss()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.default_layout = None

    def _build_model(self):
        model_list = ["SegNet_AlexNet", "SegNet_VGG", "SegNet_ResNet18", "SegNet_ResNet50",
                      "SegNet_ResNet101", "SegNet_ResNet34", "SegNet_ResNet152",
                      "FPN_ResNet18", "FPN_ResNet50", "FPN_ResNet101", "FPN_ResNet34", "FPN_ResNet152",
                      "FCN_AlexNet", "FCN_VGG", "FCN_ResNet18", "FCN_ResNet50", "FCN_ResNet101",
                      "FCN_ResNet34", "FCN_ResNet152",
                      "UNet_VGG",
                      "MLP", "ConditionalNeuralProcess", "TransformerRecon",
                      "DenseDeepGCN"]
        layout_model = self.hparams.model_name + '_' + self.hparams.backbone
        assert (layout_model in model_list or self.hparams.model_name in model_list)
        self.layout_model = layout_model if layout_model in model_list else self.hparams.model_name

        self.vec = True if self.layout_model in ["MLP", "ConditionalNeuralProcess", "TransformerRecon", "DenseDeepGCN"] else False

        self.model = nn.ModuleList([getattr(models, self.layout_model)(input_dim=self.input_dim, output_dim=self.output_dim) for i in range(self.hparams.div_num*self.hparams.div_num)]) if self.vec else getattr(models, self.layout_model)(in_channels=1)

    def forward(self, x):

        if self.vec:
            if self.layout_model == "MLP":
                for num, submodel in enumerate(self.model):
                    if num == 0:
                        output = submodel(x[1]).unsqueeze(1)
                    else:
                        output = torch.cat((output, submodel(x[1]).unsqueeze(1)), axis=1)
            elif self.layout_model == "ConditionalNeuralProcess" or self.layout_model == "TransformerRecon":
                for num, submodel in enumerate(self.model):
                    if num == 0:
                        output = submodel(x[0], x[1], (x[2])[:,0,...], (x[3])[:,0,...]).unsqueeze(1)
                    else:
                        output = torch.cat((output, submodel(x[0], x[1], (x[2])[:,num,...], (x[3])[:,num,...]).unsqueeze(1)), axis=1)
            elif self.layout_model =="DenseDeepGCN":
                for num, submodel in enumerate(self.model):
                    if num == 0:
                        output = submodel(x[num, ...]).unsqueeze(1)
                    else:
                        output = torch.cat((output, submodel(x[num, ...]).unsqueeze(1)), axis=1)
        else:
            output = self.model(x)

        return output

    def __dataloader(self, dataset, shuffle=False):
        loader = DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
        return loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        return [optimizer], [scheduler]

    def read_image_data(self):

        size: int = self.hparams.input_size
        transform_layout = transforms.Compose(
            [
                transforms.Resize(size=(size, size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    torch.tensor([self.hparams.mean_layout]),
                    torch.tensor([self.hparams.std_layout]),
                ),
            ]
        )
        transform_heat = transforms.Compose(
            [
                transforms.Resize(size=(size, size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    torch.tensor([self.hparams.mean_heat]),
                    torch.tensor([self.hparams.std_heat]),
                ),
            ]
        )

        # here only support format "mat"
        assert self.hparams.data_format == "mat"
        trainval_dataset = LayoutDataset(
            self.hparams.data_root,
            list_path=self.hparams.train_list,
            train=True,
            transform=transform_layout,
            target_transform=transform_heat,
        )
        test_dataset = LayoutDataset(
            self.hparams.data_root,
            list_path=self.hparams.test_list,
            train=False,
            transform=transform_layout,
            target_transform=transform_heat,
        )
        return trainval_dataset, test_dataset

    def read_vec_data(self):
        assert self.hparams.data_format == "mat"
        trainval_dataset = LayoutVecDataset(
            self.hparams.data_root,
            list_path=self.hparams.train_list,
            train=True,
            div_num=self.hparams.div_num,
            transform=[self.hparams.mean_layout, self.hparams.std_layout],
            target_transform=[self.hparams.mean_heat, self.hparams.std_heat],
        )
        test_dataset = LayoutVecDataset(
            self.hparams.data_root,
            list_path=self.hparams.test_list,
            train=False,
            div_num=self.hparams.div_num,
            transform=[self.hparams.mean_layout, self.hparams.std_layout],
            target_transform=[self.hparams.mean_heat, self.hparams.std_heat],
        )
        self.data_info = test_dataset
        return trainval_dataset, test_dataset

    def prepare_data(self):
        """Prepare dataset
        """
        trainval_dataset, test_dataset = self.read_vec_data() if self.vec else self.read_image_data()

        # split train/val set
        train_length, val_length = int(len(trainval_dataset) * 0.8), len(trainval_dataset)-int(len(trainval_dataset) * 0.8)
        train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset,
                                                                   [train_length, val_length])
        print(
            f"Prepared dataset, train:{int(len(train_dataset))},\
                val:{int(len(val_dataset))}, test:{len(test_dataset)}"
        )

        # assign to use in dataloaders
        self.train_dataset = self.__dataloader(train_dataset, shuffle=True)
        self.val_dataset = self.__dataloader(val_dataset, shuffle=False)
        self.test_dataset = self.__dataloader(test_dataset, shuffle=False)

        self.default_layout = trainval_dataset._layout()
        
        self.default_layout = torch.from_numpy(self.default_layout).unsqueeze(0).unsqueeze(0)

    def train_dataloader(self):
        return self.train_dataset

    def val_dataloader(self):
        return self.val_dataset

    def test_dataloader(self):
        return self.test_dataset

    def training_step(self, batch, batch_idx):

        if self.vec:
            obs_index, heat_obs, pred_index, heat, _ = batch
            heat_info = [obs_index, heat_obs, pred_index, heat]
        else:
            heat_obs, heat = batch
            heat_info = heat_obs

        if self.layout_model=="ConditionalNeuralProcess" or self.layout_model=="TransformerRecon":
            heat_info[1] = heat_info[1].transpose(1,2)
            heat_info[3] = heat_info[3].transpose(2,3)
            heat = heat.transpose(2,3)
        elif self.layout_model=="DenseDeepGCN":
            heat_obs=heat_obs.squeeze()
            pseudo_heat = torch.zeros_like(heat[:,0,:]).squeeze()
            inputs = torch.cat((torch.cat((heat_obs, pseudo_heat), 1).unsqueeze(-1), torch.cat((obs_index, pred_index[:,0,...]), 1)), 2).transpose(1,2).unsqueeze(-1).unsqueeze(0)

            for i in range(self.hparams.div_num*self.hparams.div_num-1): 
                input_single = torch.cat((torch.cat((heat_obs, pseudo_heat), 1).unsqueeze(-1), torch.cat((obs_index, pred_index[:,i+1,...]), 1)), 2).transpose(1,2).unsqueeze(-1).unsqueeze(0)
                inputs = torch.cat((inputs, input_single), 0)

            heat_info = inputs

            labels = torch.cat((heat_obs,heat[:,0,:].squeeze()), 1).unsqueeze(1).unsqueeze(1)
            for i in range(self.hparams.div_num*self.hparams.div_num-1):
                label = torch.cat((heat_obs,heat[:,i,:].squeeze()), 1).unsqueeze(1).unsqueeze(1)
                labels = torch.cat((labels, label), 1)

            heat = labels
        else:
            pass

        heat_pred = self(heat_info)

        loss = self.criterion(heat, heat_pred) * self.hparams.std_heat
        self.log("train/training_mae", loss)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        if self.vec:
            obs_index, heat_obs, pred_index, heat, _ = batch
            heat_info = [obs_index, heat_obs, pred_index, heat]
        else:
            heat_obs, heat = batch
            heat_info = heat_obs

        if self.layout_model=="ConditionalNeuralProcess" or self.layout_model=="TransformerRecon":
            heat_info[1] = heat_info[1].transpose(1,2)
            heat_info[3] = heat_info[3].transpose(2,3)
            heat = heat.transpose(2,3)
        elif self.layout_model=="DenseDeepGCN":
            heat_obs=heat_obs.squeeze()

            pseudo_heat = torch.zeros_like(heat[:,0,:]).squeeze()

            inputs = torch.cat((torch.cat((heat_obs, pseudo_heat), 1).unsqueeze(-1), torch.cat((obs_index, pred_index[:,0,...]), 1)), 2).transpose(1,2).unsqueeze(-1).unsqueeze(0)

            for i in range(self.hparams.div_num*self.hparams.div_num-1): 
                input_single = torch.cat((torch.cat((heat_obs, pseudo_heat), 1).unsqueeze(-1), torch.cat((obs_index, pred_index[:,i+1,...]), 1)), 2).transpose(1,2).unsqueeze(-1).unsqueeze(0)
                inputs = torch.cat((inputs, input_single), 0)

            heat_info = inputs

            labels = torch.cat((heat_obs,heat[:,0,:].squeeze()), 1).unsqueeze(1).unsqueeze(1)
            for i in range(self.hparams.div_num*self.hparams.div_num-1):
                label = torch.cat((heat_obs,heat[:,i,:].squeeze()), 1).unsqueeze(1).unsqueeze(1)
                labels = torch.cat((labels, label), 1)

            heat = labels
        else:
            pass

        heat_pred = self(heat_info)

        loss = self.criterion(heat, heat_pred) * self.hparams.std_heat
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val/val_mae", val_loss_mean.item() * self.hparams.std_heat)

    def test_step(self, batch, batch_idx):

        if self.vec:
            obs_index, heat_obs, pred_index, heat0, heat = batch
            heat_info = [obs_index, heat_obs, pred_index, heat0]
        else:
            heat_obs, heat = batch
            heat_info = heat_obs
        
        if self.layout_model=="ConditionalNeuralProcess" or self.layout_model=="TransformerRecon":
            heat_info[1] = heat_info[1].transpose(1,2)
            heat_info[3] = heat_info[3].transpose(2,3)
        elif self.layout_model=="DenseDeepGCN":
            heat_obs = heat_obs.squeeze()
            pseudo_heat = torch.zeros_like(heat0[:,0,:]).squeeze()
            inputs = torch.cat((torch.cat((heat_obs, pseudo_heat), 1).unsqueeze(-1), torch.cat((obs_index, pred_index[:,0,...]), 1)), 2).transpose(1,2).unsqueeze(-1).unsqueeze(0)

            for i in range(self.hparams.div_num*self.hparams.div_num-1): 
                input_single = torch.cat((torch.cat((heat_obs, pseudo_heat), 1).unsqueeze(-1), torch.cat((obs_index, pred_index[:,i+1,...]), 1)), 2).transpose(1,2).unsqueeze(-1).unsqueeze(0)
                inputs = torch.cat((inputs, input_single), 0)

            heat_info = inputs

        heat_pred0 = self(heat_info)

        if self.vec:
            if self.layout_model=="DenseDeepGCN":
                heat_pred0 = heat_pred0[...,-self.output_dim:]
            else:
                pass
            
            heat_pred0 = heat_pred0.reshape((-1, self.hparams.div_num*self.hparams.div_num, int(200 / self.hparams.div_num), int(200 / self.hparams.div_num)))
            heat_pred = torch.zeros_like(heat_pred0).reshape((-1, 1, 200, 200))
            for i in range(self.hparams.div_num):
                for j in range(self.hparams.div_num):
                    heat_pred[..., 0+i:200:self.hparams.div_num, 0+j:200:self.hparams.div_num] = heat_pred0[:, self.hparams.div_num*i+j,...].unsqueeze(1)
            heat_pred = heat_pred.transpose(2,3)
            heat = heat.unsqueeze(1)

        else:
            heat_pred = heat_pred0

        loss = self.criterion(heat_pred, heat) * self.hparams.std_heat

        default_layout = torch.repeat_interleave(self.default_layout, repeats=heat_pred.size(0), dim=0).float().to(device=heat.device)
        ones = torch.ones_like(default_layout).to(device=heat.device)
        zeros = torch.zeros_like(default_layout).to(device=heat.device)
        layout_ind = torch.where(default_layout<1e-2,zeros,ones)
        loss_2 = torch.sum(torch.abs(torch.sub(heat, heat_pred)) *layout_ind )* self.hparams.std_heat/ torch.sum(layout_ind)
        #---------------------------------
        loss_1 = torch.sum(torch.max(torch.max(torch.max(torch.abs(torch.sub(heat,heat_pred)) * layout_ind, 3).values, 2).values * self.hparams.std_heat,1).values)/heat_pred.size(0)
        #---------------------------------
        boundary_ones = torch.zeros_like(default_layout).to(device=heat.device)
        boundary_ones[..., -2:, :] = ones[..., -2:, :]
        boundary_ones[..., :2, :] = ones[..., :2, :]
        boundary_ones[..., :, :2] = ones[..., :, :2]
        boundary_ones[..., :, -2:] = ones[..., :, -2:]
        loss_3 = torch.sum(torch.abs(torch.sub(heat, heat_pred)) *boundary_ones )* self.hparams.std_heat/ torch.sum(boundary_ones)
        #----------------------------------
        loss_4 = torch.sum(torch.max(torch.max(torch.max(torch.abs(torch.sub(heat,heat_pred)), 3).values, 2).values * self.hparams.std_heat,1).values)/heat_pred.size(0)
        
        return {"test_loss": loss, "test_loss_1": loss_1, "test_loss_2": loss_2, "test_loss_3": loss_3, "test_loss_4": loss_4}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x["test_loss"] for x in outputs]).mean()
        self.log("test_loss (" + "MAE" +")", test_loss_mean.item())
        #test_loss_max = torch.max(torch.stack([x["test_loss_1"] for x in outputs]))
        test_loss_max = torch.stack([x["test_loss_1"] for x in outputs]).mean()
        self.log("test_loss_1 (" + "M-CAE" +")", test_loss_max.item())
        test_loss_com_mean = torch.stack([x["test_loss_2"] for x in outputs]).mean()
        self.log("test_loss_2 (" + "CMAE" +")", test_loss_com_mean.item())
        test_loss_bc_mean = torch.stack([x["test_loss_3"] for x in outputs]).mean()
        self.log("test_loss_3 (" + "BMAE" +")", test_loss_bc_mean.item())
        test_loss_max_1 = torch.stack([x["test_loss_4"] for x in outputs]).mean()
        self.log("test_loss_4 (" + "MaxAE" + ")", test_loss_max_1.item())

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no-cover
        """Parameters you define here will be available to your model through `self.hparams`.
        """
        # dataset args
        parser.add_argument("--data_root", type=str, required=True, help="path of dataset")
        parser.add_argument("--train_list", type=str, required=True, help="path of train dataset list")
        parser.add_argument("--train_size", default=0.8, type=float, help="train_size in train_test_split")
        parser.add_argument("--test_list", type=str, required=True, help="path of test dataset list")
        #parser.add_argument("--boundary", type=str, default="rm_wall", help="boundary condition")
        parser.add_argument("--data_format", type=str, default="mat", choices=["mat", "h5"], help="dataset format")

        # Normalization params
        parser.add_argument("--mean_layout", default=0, type=float)
        parser.add_argument("--std_layout", default=1, type=float)
        parser.add_argument("--mean_heat", default=0, type=float)
        parser.add_argument("--std_heat", default=1, type=float)

        # Model params (opt)
        parser.add_argument("--input_size", default=200, type=int)
        parser.add_argument("--model_name", type=str, default='FCN', help="the name of chosen model")
        parser.add_argument("--backbone", type=str, default='AlexNet', help="the used backbone in the regression model")

        # div_num for vec (opt)
        parser.add_argument("--div_num", default=4, type=int, help="division of heat source systems")
        
        return parser
