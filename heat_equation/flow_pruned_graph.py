import numpy as np
import jax.random as jr
import scipy.io as sio


from ckpt_io import save_model
import models


if __name__ == "__main__":
    data = sio.loadmat("./data/data.mat")
    x_data = data["X"]
    u_data = data["U"]
    y_data = data["X_next"] - data["X"]

    for index in [1, 100]:
        seed = 81763263  # for reproducibility
        np.random.seed(seed)

        if index == 1:
            _x_data = np.concatenate([x_data[:, 0:2], u_data[:, 0:1]], axis=-1)
        elif index == 100:
            _x_data = np.concatenate([x_data[:, 98:100], u_data[:, 1:2]], axis=-1)
        _y_data = y_data[:, index-1: index]
            
        x_mu = np.mean(_x_data, axis=0)
        x_sd = np.std(_x_data, axis=0)
        y_mu = np.mean(_y_data, axis=0)
        y_sd = np.std(_y_data, axis=0)
        X_train = (_x_data - x_mu) / x_sd
        Y_train = (_y_data - y_mu) / y_sd

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
            ckpt_dir=f"./checkpoints/cfm_{index}_pruned",
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
