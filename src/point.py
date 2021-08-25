import os
import time
import tqdm
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import configargparse
from sklearn.metrics import mean_absolute_error

import src
from src.data.layout import LayoutPointDataset
from src.models.instance import Point


TOL = 1e-14


def main(hparams):

    model = PointModel(hparams)
    model.testing_step()


class PointModel:
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.data = self.read_data()
        self.metric = Metric()

    def read_data(self):

        test_dataset = LayoutPointDataset(
            self.hparams.data_root,
            list_path=self.hparams.test_list,
            train=False,
        )
        return test_dataset

    def testing_step(self):

        trange = tqdm.tqdm(self.data)
        all_mae, all_maxae, all_cmae, all_mcae, all_bmae = 0, 0, 0, 0, 0

        mae_test = []
        for num, data in enumerate(trange):

            path = self.data.sample_files[num]
            u_obs = data[0]
            u = data[1]
            F = data[2]
            sample = Point(self.hparams.model_name, u_obs, u)

            u_pred = sample.predict()
            if self.hparams.plot:
                self.plot(path, u_pred, u, F)

            all_mae += self.metric.mae(u_pred, u)
            all_maxae += self.metric.maxae(u_pred, u)
            all_cmae += self.metric.cmae(u_pred, u, F)
            all_mcae += self.metric.mcae(u_pred, u, F)
            all_bmae += self.metric.bmae(u_pred, u)

            mae_test.append(self.metric.mae(u_pred, u))

            trange.set_description("Testing")

            trange.set_postfix(
                MAE=all_mae / (num + 1),
                MaxAE=all_maxae / (num + 1),
                CMAE=all_cmae / (num + 1),
                MCAE=all_mcae / (num + 1),
                BMAE=all_bmae / (num + 1),
            )

        mae_test = np.array(mae_test)
        print(mae_test.mean())
        np.savetxt("outputs/mae_test.csv", mae_test, fmt="%f", delimiter=",")

    def plot(self, path, heat_pre, u_true, hs_F):
        fig = plt.figure(figsize=(22.5, 5))

        grid_x = np.linspace(0, 0.1, num=200)
        grid_y = np.linspace(0, 0.1, num=200)
        X, Y = np.meshgrid(grid_x, grid_y)

        plt.subplot(141)
        plt.title("Real Time Power")

        im = plt.pcolormesh(X, Y, hs_F)
        plt.colorbar(im)
        fig.tight_layout(pad=2.0, w_pad=3.0, h_pad=2.0)

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
            "outputs/predict_plot", os.path.splitext(os.path.basename(path))[0] + ".png"
        )
        mat_name = os.path.join(
            "outputs/predict_plot", os.path.splitext(os.path.basename(path))[0] + ".mat"
        )
        sio.savemat(mat_name, {"pre": heat_pre, "u_true": u_true})
        fig.savefig(save_name, dpi=300)
        plt.close()


class Metric:
    def __init__(self):
        super().__init__()

    def mae(self, u_pred, u):
        return mean_absolute_error(u_pred, u)

    def maxae(self, u_pred, u):
        return np.max(np.max(abs(u_pred - u)))

    def mcae(self, u_pred, u, F):
        F[np.where(F > TOL)] = 1
        return np.max(np.max(abs(u_pred - u) * F))

    def cmae(self, u_pred, u, F):
        F[np.where(F > TOL)] = 1
        return np.sum(np.sum(abs(u_pred - u) * F)) / np.sum(F)

    def bmae(self, u_pred, u):
        ind = np.zeros_like(u_pred)
        ind[:2, ...] = 1
        ind[-2:, ...] = 1
        ind[..., :2] = 1
        ind[..., -2:] = 1
        return np.sum(np.sum(abs(u_pred - u) * ind)) / np.sum(ind)
