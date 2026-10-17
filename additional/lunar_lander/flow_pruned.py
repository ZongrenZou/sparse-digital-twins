import numpy as np
import jax.random as jr
import scipy.io as sio
import numpy as np


from ckpt_io import save_model
import models



if __name__ == "__main__":
    data = sio.loadmat("./data/lunar_lander_impulse_data_0.mat")
    actions = data["actions"]
    states = data["states"]
    delta_vx = data["delta_vx"].reshape([-1, 1])
    delta_vy = data["delta_vy"].reshape([-1, 1])
    delta_va = data["delta_angvel"].reshape([-1, 1])
    print(states.shape)

    x_data = np.concatenate(
        [states[:, 4:5], actions], axis=-1,
    )
    y_data = np.concatenate(
        [delta_vx, delta_vy, delta_va],
        axis=-1,
    )

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
            steps=10_000,
        ),
    )

    save_model(
        ckpt_dir="./checkpoints/cfm_pruned",
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
