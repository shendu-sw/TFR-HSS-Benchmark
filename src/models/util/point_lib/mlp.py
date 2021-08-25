import numpy as np
import math
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from .util.base import Base


class MLPP(Base):
    def __init__(self, u_obs, u, layers=[100, 50], constant=298):
        super().__init__(u_obs, u)
        self.layers = layers

    def predict(self):

        self.pred_init()
        X, Y = self.train_samples()
        test_samples = self.test_samples()

        regressor = MLPRegressor(
            hidden_layer_sizes=self.layers,
            alpha=2,
            solver="adam",
            random_state=1,
            max_iter=20000,
        )
        regressor.fit(X, Y)

        self.u_pred = regressor.predict(test_samples).reshape(
            self.u.shape[0], self.u.shape[1]
        )
        return self.u_pred


if __name__ == "__main__":
    m = sio.loadmat("Example0.mat")
    u_obs = m["u_obs"]
    u = m["u"]
    sample = MLPP(u_obs, u)
    u_pred = sample.predict()
    print("mae:", mae(u_pred, u))

    # u_pred=u_pred*50+298

    from sklearn.metrics import mean_absolute_error as mae

    print("mae:", mae(u_pred, u))

    fig = plt.figure(figsize=(22.5, 5))

    grid_x = np.linspace(0, 0.1, num=200)
    grid_y = np.linspace(0, 0.1, num=200)
    X, Y = np.meshgrid(grid_x, grid_y)

    fig = plt.figure(figsize=(22.5, 5))

    plt.subplot(141)
    plt.title("Absolute Error")
    im = plt.pcolormesh(X, Y, abs(u - u_pred))
    plt.colorbar(im)
    fig.tight_layout(pad=2.0, w_pad=3.0, h_pad=2.0)

    plt.subplot(142)
    plt.title("Real Temperature Field")
    im = plt.contourf(X, Y, u, levels=150, cmap="jet")
    plt.colorbar(im)

    plt.subplot(143)
    plt.title("Reconstructed Temperature Field")
    im = plt.contourf(X, Y, u_pred, levels=150, cmap="jet")
    plt.colorbar(im)

    plt.subplot(144)
    plt.title("Absolute Error")
    im = plt.contourf(X, Y, abs(u - u_pred), levels=150, cmap="jet")
    plt.colorbar(im)

    # save_name = os.path.join('outputs/predict_plot', '1.png')
    # fig.savefig(save_name, dpi=300)

    # fig = plt.figure(figsize=(5,5))
    # im = plt.imshow(u,cmap='jet')
    # plt.colorbar(im)
    fig.savefig("prediction.png", dpi=300)
