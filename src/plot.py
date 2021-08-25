"""
Runs a model on a single node across multiple gpus.
"""
import os
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
import scipy.io as sio
import matplotlib.pyplot as plt
import configargparse

from src.DeepRegression import Model


TOL = 1e-14


def main(hparams):

    if hparams.gpu == 0:
        device = torch.device("cpu")
    else:
        ngpu = "cuda:" + str(hparams.gpu - 1)
        print(ngpu)
        device = torch.device(ngpu)
    model = Model(hparams).to(device)

    print(hparams)
    print()

    # Model loading
    model_path = os.path.join(
        f"lightning_logs/version_" + hparams.test_check_num, "checkpoints/"
    )
    ckpt = list(Path(model_path).glob("*.ckpt"))[0]
    print(ckpt)

    model = model.load_from_checkpoint(str(ckpt))

    model.eval()
    model.to(device)
    mae_test = []

    # Testing Set
    root = hparams.data_root
    test_list = hparams.test_list
    file_path = os.path.join(root, test_list)
    test_name = os.path.splitext(os.path.basename(test_list))[0]
    root_dir = os.path.join(root, "test", test_name)

    with open(file_path, "r") as fp:
        for line in fp.readlines():
            # Data Reading
            data_path = line.strip()
            path = os.path.join(root_dir, data_path)

            if model.vec:
                _, test_data = model.read_vec_data()
                obs_index, heat_obs, pred_index, heat0, heat = test_data._loader(path)
                u_true = heat.squeeze().squeeze().numpy()
                heat_obs = (heat_obs - hparams.mean_layout) / hparams.std_layout
                heat0 = (heat0 - hparams.mean_heat) / hparams.std_heat
                heat = (heat - hparams.mean_heat) / hparams.std_heat
                obs_index, heat_obs, pred_index, heat0, heat = (
                    obs_index.to(device),
                    heat_obs.to(device),
                    pred_index.to(device),
                    heat0.to(device),
                    heat.to(device),
                )
                heat_info = [obs_index, heat_obs, pred_index, heat0]

                if (
                    model.layout_model == "ConditionalNeuralProcess"
                    or model.layout_model == "TransformerRecon"
                ):
                    heat_info[1] = heat_info[1].transpose(1, 2)
                    heat_info[3] = heat_info[3].transpose(2, 3)
                elif model.layout_model == "DenseDeepGCN":
                    heat_obs = heat_obs.squeeze()
                    pseudo_heat = torch.zeros_like(heat0[:, 0, :]).squeeze()
                    inputs = (
                        torch.cat(
                            (
                                torch.cat((heat_obs, pseudo_heat), 1).unsqueeze(-1),
                                torch.cat((obs_index, pred_index[:, 0, ...]), 1),
                            ),
                            2,
                        )
                        .transpose(1, 2)
                        .unsqueeze(-1)
                        .unsqueeze(0)
                    )

                    for i in range(hparams.div_num * hparams.div_num - 1):
                        input_single = (
                            torch.cat(
                                (
                                    torch.cat((heat_obs, pseudo_heat), 1).unsqueeze(-1),
                                    torch.cat(
                                        (obs_index, pred_index[:, i + 1, ...]), 1
                                    ),
                                ),
                                2,
                            )
                            .transpose(1, 2)
                            .unsqueeze(-1)
                            .unsqueeze(0)
                        )
                        inputs = torch.cat((inputs, input_single), 0)

                    heat_info = inputs
            else:
                data = sio.loadmat(path)
                u_true, u_obs = data["u"], data["u_obs"]

                u_obs[np.where(u_obs < TOL)] = hparams.mean_layout
                u_obs = (
                    torch.Tensor((u_obs - hparams.mean_layout) / hparams.std_layout)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                )
                heat = (
                    torch.Tensor((u_true - hparams.mean_heat) / hparams.std_heat)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                )
                heat_info = u_obs

            hs_F = sio.loadmat(path)["F"]

            # Plot u_obs and Real Temperature Field
            fig = plt.figure(figsize=(22.5, 5))

            grid_x = np.linspace(0, 0.1, num=200)
            grid_y = np.linspace(0, 0.1, num=200)
            X, Y = np.meshgrid(grid_x, grid_y)

            plt.subplot(141)
            plt.title("Real Time Power")

            im = plt.pcolormesh(X, Y, hs_F)
            plt.colorbar(im)
            fig.tight_layout(pad=2.0, w_pad=3.0, h_pad=2.0)

            with torch.no_grad():

                heat_pred0 = model(heat_info)
                if model.vec:
                    if model.layout_model == "DenseDeepGCN":
                        heat_pred0 = heat_pred0[..., -model.output_dim :]
                    else:
                        pass

                    heat_pred0 = heat_pred0.reshape(
                        (
                            -1,
                            hparams.div_num * hparams.div_num,
                            int(200 / hparams.div_num),
                            int(200 / hparams.div_num),
                        )
                    )
                    heat_pre = torch.zeros_like(heat_pred0).reshape((-1, 1, 200, 200))
                    for i in range(hparams.div_num):
                        for j in range(hparams.div_num):
                            heat_pre[
                                ...,
                                0 + i : 200 : hparams.div_num,
                                0 + j : 200 : hparams.div_num,
                            ] = heat_pred0[:, hparams.div_num * i + j, ...].unsqueeze(1)
                    heat_pre = heat_pre.transpose(2, 3)
                    heat = heat.unsqueeze(1)
                else:
                    heat_pre = heat_pred0
                    heat = heat

                mae = F.l1_loss(heat, heat_pre) * hparams.std_heat
                print("sample:", data_path)
                print("MAE:", mae)
            mae_test.append(mae.item())
            heat_pre = (
                heat_pre.squeeze(0).squeeze(0).cpu().numpy() * hparams.std_heat
                + hparams.mean_heat
            )
            # heat_pre = np.transpose(heat_pre, (1,0))
            hmax = max(np.max(heat_pre), np.max(u_true))
            hmin = min(np.min(heat_pre), np.min(u_true))

            plt.subplot(142)
            plt.title("Real Temperature Field")

            im = plt.contourf(X, Y, u_true, levels=150, cmap="jet")
            plt.colorbar(im)

            plt.subplot(143)
            plt.title("Reconstructed Temperature Field")

            im = plt.contourf(X, Y, heat_pre, levels=150, cmap="jet")
            plt.colorbar(im)

            plt.subplot(144)
            plt.title("Absolute Error")

            im = plt.contourf(X, Y, np.abs(heat_pre - u_true), levels=150, cmap="jet")

            plt.colorbar(im)

            save_name = os.path.join(
                "outputs/predict_plot",
                os.path.splitext(os.path.basename(path))[0] + ".png",
            )
            mat_name = os.path.join(
                "outputs/predict_plot",
                os.path.splitext(os.path.basename(path))[0] + ".mat",
            )
            sio.savemat(mat_name, {"pre": heat_pre, "u_true": u_true})
            fig.savefig(save_name, dpi=300)
            plt.close()

    mae_test = np.array(mae_test)
    print(mae_test.mean())
    np.savetxt("outputs/mae_test.csv", mae_test, fmt="%f", delimiter=",")
