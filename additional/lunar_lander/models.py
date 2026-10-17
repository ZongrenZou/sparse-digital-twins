from dataclasses import dataclass
from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flax.training import train_state
from flax import linen as nn


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
        return out


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


# ---------------------------------------------------------------------------
# Conditional GAN: model p(y | x) via G(z, x) -> y
# ---------------------------------------------------------------------------


class GeneratorMLP(nn.Module):
    """G(z, x): concat [z, x] -> MLP -> R^d."""

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
        return nn.Dense(self.dim_out)(h)


class DiscriminatorMLP(nn.Module):
    """D(y, x): concat [y, x] -> MLP -> logit."""

    hidden_dims: Tuple[int, ...]

    @nn.compact
    def __call__(self, y, x):
        if y.ndim == 1:
            y = y[None, :]
        if x.ndim == 1:
            x = x[None, :]
        h = jnp.concatenate([y, x], axis=-1)
        for width in self.hidden_dims:
            h = nn.silu(nn.Dense(width)(h))
        return nn.Dense(1)(h).squeeze(-1)


@dataclass
class GanModelCfg:
    hidden_dims: Tuple[int, ...] = (64, 64, 64, 64)
    noise_dim: int = 8


@dataclass
class GanTrainCfg:
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    batch_size: int = 1_000
    steps: int = 5_000
    n_critic: int = 1
    r1_gamma: float = 10.0
    log_every: int = 1


def _create_gan_states(rng, gen, disc, d_noise: int, d_y: int, k_x: int, cfg: GanTrainCfg):
    rng_g, rng_d = jr.split(rng)
    z0 = jnp.zeros((1, d_noise))
    y0 = jnp.zeros((1, d_y))
    x0 = jnp.zeros((1, k_x))

    g_params = gen.init(rng_g, z0, x0)["params"]
    d_params = disc.init(rng_d, y0, x0)["params"]

    tx_g = optax.adam(cfg.lr_g, b1=0.5, b2=0.999)
    tx_d = optax.adam(cfg.lr_d, b1=0.5, b2=0.999)

    g_state = train_state.TrainState(
        step=0,
        apply_fn=gen.apply,
        params=g_params,
        tx=tx_g,
        opt_state=tx_g.init(g_params),
    )
    d_state = train_state.TrainState(
        step=0,
        apply_fn=disc.apply,
        params=d_params,
        tx=tx_d,
        opt_state=tx_d.init(d_params),
    )
    return g_state, d_state


def train_cgan_flax(
    key,
    X_train,
    Y_train,
    model_cfg: GanModelCfg,
    train_cfg: GanTrainCfg,
):
    """Train a conditional GAN for p(y | x). Returns generator params and module."""

    N, k = X_train.shape
    d = Y_train.shape[1]
    noise_dim = model_cfg.noise_dim

    gen = GeneratorMLP(hidden_dims=model_cfg.hidden_dims, dim_out=d)
    disc = DiscriminatorMLP(hidden_dims=model_cfg.hidden_dims)

    key, init_key = jr.split(key)
    g_state, d_state = _create_gan_states(
        init_key, gen, disc, d_noise=noise_dim, d_y=d, k_x=k, cfg=train_cfg
    )

    def disc_loss_fn(d_params, g_params, key, Xb, Yb):
        key_z, key_r1 = jr.split(key)
        z = jr.normal(key_z, (Xb.shape[0], noise_dim))
        y_fake = gen.apply({"params": g_params}, z, Xb)

        logits_real = disc.apply({"params": d_params}, Yb, Xb)
        logits_fake = disc.apply({"params": d_params}, y_fake, Xb)

        # Non-saturating logistic loss
        loss_real = optax.sigmoid_binary_cross_entropy(
            logits_real, jnp.ones_like(logits_real)
        ).mean()
        loss_fake = optax.sigmoid_binary_cross_entropy(
            logits_fake, jnp.zeros_like(logits_fake)
        ).mean()
        loss = loss_real + loss_fake

        # R1 gradient penalty on real samples (stabilizes training)
        if train_cfg.r1_gamma > 0.0:

            def d_real(y):
                return disc.apply({"params": d_params}, y, Xb).sum()

            grads_y = jax.grad(d_real)(Yb)
            r1 = 0.5 * train_cfg.r1_gamma * jnp.mean(jnp.sum(grads_y**2, axis=-1))
            loss = loss + r1
        else:
            r1 = 0.0

        return loss, (loss_real, loss_fake, r1)

    def gen_loss_fn(g_params, d_params, key, Xb):
        z = jr.normal(key, (Xb.shape[0], noise_dim))
        y_fake = gen.apply({"params": g_params}, z, Xb)
        logits_fake = disc.apply({"params": d_params}, y_fake, Xb)
        # Non-saturating generator loss: maximize log D(G(z,x))
        loss = optax.sigmoid_binary_cross_entropy(
            logits_fake, jnp.ones_like(logits_fake)
        ).mean()
        return loss

    @jax.jit
    def d_step(d_state, g_state, key, Xb, Yb):
        (loss, aux), grads = jax.value_and_grad(disc_loss_fn, has_aux=True)(
            d_state.params, g_state.params, key, Xb, Yb
        )
        d_state = d_state.apply_gradients(grads=grads)
        return d_state, loss, aux

    @jax.jit
    def g_step(g_state, d_state, key, Xb):
        loss, grads = jax.value_and_grad(gen_loss_fn)(
            g_state.params, d_state.params, key, Xb
        )
        g_state = g_state.apply_gradients(grads=grads)
        return g_state, loss

    for it in range(1, train_cfg.steps + 1):
        idx = np.random.choice(N, N, replace=False)
        d_loss_acc = 0.0
        g_loss_acc = 0.0
        n_batches = N // train_cfg.batch_size

        for i in range(n_batches):
            idx_batch = idx[i * train_cfg.batch_size : (i + 1) * train_cfg.batch_size]
            X_batch = jnp.array(X_train[idx_batch])
            Y_batch = jnp.array(Y_train[idx_batch])

            for _ in range(train_cfg.n_critic):
                key, d_key = jr.split(key)
                d_state, d_loss, _ = d_step(d_state, g_state, d_key, X_batch, Y_batch)

            key, g_key = jr.split(key)
            g_state, g_loss = g_step(g_state, d_state, g_key, X_batch)

            d_loss_acc = d_loss_acc + d_loss
            g_loss_acc = g_loss_acc + g_loss

        if it % train_cfg.log_every == 0:
            print(
                it,
                float(d_loss_acc / n_batches),
                float(g_loss_acc / n_batches),
                flush=True,
            )

    return g_state.params, gen


def sample_cgan(params, gen, x, noise_dim: int, key):
    """Draw one y ~ p_G(· | x) for each row of x."""
    z = jr.normal(key, (x.shape[0], noise_dim))
    return gen.apply({"params": params}, z, x)