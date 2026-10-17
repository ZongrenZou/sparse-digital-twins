import numpy as np
import jax.random as jr
import scipy.io as sio
import numpy as np


from ckpt_io import save_model
import models


if __name__ == "__main__":
    M = 95_000

    data = sio.loadmat("./data/pendulum_data.mat")
    theta = data["theta"].reshape([500, 200])
    theta_next = data["theta_next"].reshape([500, 200])
    u = data["u"].reshape([500, 200])
    y_data = theta_next - theta

    x_data = np.stack(
        [
            np.sin(theta[:, 9:]),
            np.sin(theta[:, 6:-3]),
            np.cos(theta[:, 9:]),
            np.cos(theta[:, 6:-3]),
            u[:, 9:],
            u[:, 6:-3],
        ],
        axis=-1,
    )
    y_data = y_data[:, 9:]

    x_data = x_data.reshape([-1, 6])[:M]
    y_data = y_data.reshape([-1, 1])[:M]

    seed = 81763263  # for reproducibility
    np.random.seed(seed)

    ## normalize the data
    x_mu = np.mean(x_data, axis=0)
    x_sd = np.std(x_data, axis=0)
    y_mu = np.mean(y_data, axis=0)
    y_sd = np.std(y_data, axis=0)
    X_train = (x_data - x_mu) / x_sd
    Y_train = (y_data - y_mu) / y_sd

    key = np.random.randint(0, 1_000_000_000)
    key = jr.PRNGKey(key)

    ############ Training ############
    params, model = models.train_cfm_flax(
        key,
        X_train,
        Y_train,
        model_cfg=models.ModelCfg(
            hidden_dims=[64, 64, 64, 64],
        ),
        train_cfg=models.TrainCfg(
            lr=1e-4,
            batch_size=1_000,
            steps=1_000,
        ),
    )

    save_model(
        ckpt_dir="./checkpoints/cfm_pruned_2",
        params=params,
        model=model,
        model_cfg=models.ModelCfg(hidden_dims=[64, 64, 64, 64]),
        extras={
            "x_mu": x_mu,
            "x_sd": x_sd,
            "y_mu": y_mu,
            "y_sd": y_sd,
        },
    )

    print("End main.")
