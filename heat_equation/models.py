from dataclasses import dataclass
from typing import Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flax.training import train_state
from flax import linen as nn
import numpy as np


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


# ── Conditional GAN: learn p(y | x) via G(z, x) ─────────────────────────────


class GeneratorMLP(nn.Module):
    """G(z, x) -> y: concat [z, x] -> MLP -> R^{dim_out}."""

    hidden_dims: Tuple[int, ...]
    dim_out: int

    @nn.compact
    def __call__(self, z, x):
        if z.ndim == 1:
            z = z[None, :]
        if x.ndim == 1:
            x = x[None, :]

        h = jnp.concatenate([z, x], axis=-1)
        for width in self.hidden_dims:
            h = nn.silu(nn.Dense(width)(h))
        out = nn.Dense(self.dim_out)(h)
        return out if out.shape[0] > 1 else out[0]


class DiscriminatorMLP(nn.Module):
    """D(y, x) -> logit: concat [y, x] -> MLP -> R."""

    hidden_dims: Tuple[int, ...]

    @nn.compact
    def __call__(self, y, x):
        if y.ndim == 1:
            y = y[None, :]
        if x.ndim == 1:
            x = x[None, :]

        h = jnp.concatenate([y, x], axis=-1)
        for width in self.hidden_dims:
            h = nn.leaky_relu(nn.Dense(width)(h), negative_slope=0.2)
        logit = nn.Dense(1)(h)[..., 0]
        return logit if logit.shape[0] > 1 else logit[0]


@dataclass
class GanModelCfg:
    hidden_dims: Tuple[int, ...] = (64, 64, 64, 64)
    z_dim: int = 1
    d_hidden_dims: Tuple[int, ...] | None = None  # defaults to hidden_dims


def create_gan_states(rng, generator, discriminator, d_out, k_in, z_dim, cfg: TrainCfg):
    rng_g, rng_d = jr.split(rng)
    z0 = jnp.zeros((1, z_dim))
    y0 = jnp.zeros((1, d_out))
    x0 = jnp.zeros((1, k_in))

    g_params = generator.init(rng_g, z0, x0)["params"]
    d_params = discriminator.init(rng_d, y0, x0)["params"]

    tx_g = optax.adam(cfg.lr, b1=0.5, b2=0.999)
    tx_d = optax.adam(cfg.lr, b1=0.5, b2=0.999)

    g_state = train_state.TrainState.create(
        apply_fn=generator.apply, params=g_params, tx=tx_g
    )
    d_state = train_state.TrainState.create(
        apply_fn=discriminator.apply, params=d_params, tx=tx_d
    )
    return g_state, d_state


def train_cgan_flax(
    key,
    X_train,
    Y_train,
    model_cfg: GanModelCfg,
    train_cfg: TrainCfg,
):
    """Train a conditional GAN for p(y | x). Returns generator params and module."""

    N, k = X_train.shape
    d = Y_train.shape[1]
    z_dim = model_cfg.z_dim
    d_hidden = model_cfg.d_hidden_dims or model_cfg.hidden_dims

    generator = GeneratorMLP(hidden_dims=model_cfg.hidden_dims, dim_out=d)
    discriminator = DiscriminatorMLP(hidden_dims=d_hidden)

    key, init_key = jr.split(key)
    g_state, d_state = create_gan_states(
        init_key, generator, discriminator, d_out=d, k_in=k, z_dim=z_dim, cfg=train_cfg
    )

    def discriminator_loss(d_params, g_params, key, Xb, Yb):
        z = jr.normal(key, (Xb.shape[0], z_dim))
        y_fake = generator.apply({"params": g_params}, z, Xb)
        logits_real = discriminator.apply({"params": d_params}, Yb, Xb)
        logits_fake = discriminator.apply({"params": d_params}, y_fake, Xb)
        # non-saturating BCE
        loss_real = optax.sigmoid_binary_cross_entropy(
            logits_real, jnp.ones_like(logits_real)
        ).mean()
        loss_fake = optax.sigmoid_binary_cross_entropy(
            logits_fake, jnp.zeros_like(logits_fake)
        ).mean()
        return loss_real + loss_fake

    def generator_loss(g_params, d_params, key, Xb):
        z = jr.normal(key, (Xb.shape[0], z_dim))
        y_fake = generator.apply({"params": g_params}, z, Xb)
        logits_fake = discriminator.apply({"params": d_params}, y_fake, Xb)
        return optax.sigmoid_binary_cross_entropy(
            logits_fake, jnp.ones_like(logits_fake)
        ).mean()

    @jax.jit
    def step(g_state, d_state, key, Xb, Yb):
        key_d, key_g = jr.split(key)

        d_loss, d_grads = jax.value_and_grad(discriminator_loss)(
            d_state.params, g_state.params, key_d, Xb, Yb
        )
        d_state = d_state.apply_gradients(grads=d_grads)

        g_loss, g_grads = jax.value_and_grad(generator_loss)(
            g_state.params, d_state.params, key_g, Xb
        )
        g_state = g_state.apply_gradients(grads=g_grads)
        return g_state, d_state, g_loss, d_loss

    for it in range(1, train_cfg.steps + 1):
        idx = np.random.choice(N, N, replace=False)
        g_loss_acc = 0.0
        d_loss_acc = 0.0
        n_batches = N // train_cfg.batch_size

        for i in range(n_batches):
            idx_batch = idx[i * train_cfg.batch_size : (i + 1) * train_cfg.batch_size]
            X_batch = jnp.array(X_train[idx_batch])
            Y_batch = jnp.array(Y_train[idx_batch])
            key, step_key = jr.split(key)
            g_state, d_state, g_l, d_l = step(
                g_state, d_state, step_key, X_batch, Y_batch
            )
            g_loss_acc = g_loss_acc + g_l
            d_loss_acc = d_loss_acc + d_l

        print(
            it,
            float(g_loss_acc / n_batches),
            float(d_loss_acc / n_batches),
            flush=True,
        )

    return g_state.params, generator

