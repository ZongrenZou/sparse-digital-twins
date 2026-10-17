from dataclasses import dataclass
from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flax.training import train_state
import scipy.io as sio
import numpy as np


from ckpt_io import save_model
import models


@dataclass
class ModelCfg:
    hidden_dims: Tuple[int, ...] = (64, 64)


@dataclass
class TrainCfg:
    lr: float = 2e-4
    batch_size: int = 256
    steps: int = 50_000
    log_every: int = 500
    seed: int = 0


def make_cfm_batch(key, Xb, Yb):
    key_t, key_z0 = jr.split(key)
    B, d = Yb.shape
    t = jr.uniform(key_t, (B,))
    z0 = jr.normal(key_z0, (B, d))
    z_t = (1.0 - t)[:, None] * z0 + t[:, None] * Yb
    u_t = Yb - z0
    return t, z_t, Xb, u_t


def loss_fn(params, model, key, Xb, Yb):
    t, z_t, xb, u_t = make_cfm_batch(key, Xb, Yb)
    v_pred = model.apply({"params": params}, t, z_t, xb)  # (B,d)
    mse = jnp.mean(jnp.sum((v_pred - u_t) ** 2, axis=-1))
    return mse


def create_train_state(rng, model, d_in: int, k_in: int, cfg: TrainCfg):
    # dummy shapes for init
    t0 = jnp.array(0.0)
    z0 = jnp.zeros((1, d_in))
    x0 = jnp.zeros((1, k_in))
    params = model.init(rng, t0, z0, x0)["params"]
    tx = optax.adam(cfg.lr)
    return train_state.TrainState(
        step=0, apply_fn=model.apply, params=params, tx=tx, opt_state=tx.init(params)
    )


def train_cfm_flax(
    key,
    X_train,
    Y_train,
    model_cfg,
    train_cfg,
):

    N, k = X_train.shape
    d = Y_train.shape[1]

    model = models.VelocityMLP(hidden_dims=model_cfg.hidden_dims, dim_out=d)
    key, init_key = jr.split(key)
    state = create_train_state(init_key, model, d_in=d, k_in=k, cfg=train_cfg)

    @jax.jit
    def step(state, key, Xb, Yb):
        l, grads = jax.value_and_grad(loss_fn)(state.params, model, key, Xb, Yb)
        state = state.apply_gradients(grads=grads)
        return state, l

    for it in range(1, train_cfg.steps + 1):
        idx = np.random.choice(N, N, replace=False)
        loss = 0

        for i in range(N // train_cfg.batch_size):
            idx_batch = idx[i * train_cfg.batch_size : (i + 1) * train_cfg.batch_size]
            X_batch = jnp.array(X_train[idx_batch])
            Y_batch = jnp.array(Y_train[idx_batch])
            key, _ = jr.split(key, 2)
            state, l = step(state, key, X_batch, Y_batch)
            loss = loss + l

        print(it, loss / (i + 1), flush=True)

    return state.params, model


if __name__ == "__main__":
    M = 1_000_000
    data = sio.loadmat("./data/data_train.mat")
    x_data = data["x_data"][:1000, 0:3]
    # x_data = data["x_data"][:1000, 2:3]
    y_data = data["y_data"][:1000, 1:2]

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
    hidden_dims = [64, 64, 64, 64]
    params, model = train_cfm_flax(
        key,
        X_train,
        Y_train,
        model_cfg=ModelCfg(
            hidden_dims=hidden_dims,
        ),
        train_cfg=TrainCfg(
            lr=1e-4,
            batch_size=1_000,
            steps=50_000,
        ),
    )

    save_model(
        ckpt_dir="./checkpoints/cfm_2_compare",
        # ckpt_dir="./checkpoints/cfm_2_pruned_compare",
        params=params,
        model=model,
        model_cfg=ModelCfg(hidden_dims=hidden_dims),
        extras={
            "x_mu": x_mu,
            "x_sd": x_sd,
            "y_mu": y_mu,
            "y_sd": y_sd,
        },
    )

    print("End main.")
