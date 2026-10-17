from dataclasses import dataclass
from typing import Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
import numpy as np
import optax
from flax.training import train_state



class VelocityMLP(nn.Module):
    """v_theta(t, z, x): concat [z, t, x] -> MLP -> R^d."""

    hidden_dims: Tuple[int, ...]
    dim_out: int

    @nn.compact
    def __call__(self, t, z, x):
        # t: () or (B,), z: (d,) or (B,d), x: (k,) or (B,k)
        # Broadcast t to match batch of z/x if needed
        if t.ndim == 0:
            t = t[None] if z.ndim == 1 else jnp.broadcast_to(t, (z.shape[0],))
        t = t[..., None]  # (B,1) or (1,1)

        # Ensure rank alignment
        if z.ndim == 1:
            z = z[None, :]  # (1,d)
        if x.ndim == 1:
            x = x[None, :]  # (1,k)

        h = jnp.concatenate([z, t, x], axis=-1)
        for width in self.hidden_dims:
            h = nn.silu(nn.Dense(width)(h))
        out = nn.Dense(self.dim_out)(h)  # (B, d)
        return out if out.shape[0] > 1 else out[0]


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

    model = VelocityMLP(hidden_dims=model_cfg.hidden_dims, dim_out=d)
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
