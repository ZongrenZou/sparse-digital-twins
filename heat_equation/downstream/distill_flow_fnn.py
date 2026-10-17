import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence
import diffrax as dfx
from flax import linen as nn
from flax import serialization
from flax.training import train_state
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax


from ckpt_io import load_model
import models

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class StudentCfg:
    hidden_dims: tuple[int, ...] = (128, 128, 128)
    latent_dim: int = 1
    out_dim: int = 1


@dataclass(frozen=True)
class TrainCfg:
    lr: float = 1e-3
    batch_size: int = 512
    steps: int = 20_000
    log_every: int = 200
    seed: int = 0


@dataclass(frozen=True)
class DistillCfg:
    teacher_ckpt: str = "./checkpoints/cfm_1"
    output_ckpt: str = "./checkpoints/cfm_1_student"
    student_hidden_dims: tuple[int, ...] = (128, 128, 128)
    lr: float = 1e-3
    batch_size: int = 512
    steps: int = 20_000
    log_every: int = 200
    seed: int = 0


DISTILL_CFG = DistillCfg(
    teacher_ckpt="./checkpoints/cfm_1",
    output_ckpt="./checkpoints/cfm_1_student",
    student_hidden_dims=(128, 128, 128),
    lr=1e-3,
    batch_size=512,
    steps=20_000,
    log_every=200,
    seed=0,
)


class StudentMLP(nn.Module):
    hidden_dims: Sequence[int]
    dim_out: int

    @nn.compact
    def __call__(self, x, z):
        if x.ndim == 1:
            x = x[None, :]
        if z.ndim == 1:
            z = z[None, :]
        h = jnp.concatenate([x, z], axis=-1)
        for width in self.hidden_dims:
            h = nn.silu(nn.Dense(width)(h))
        out = nn.Dense(self.dim_out)(h)
        return out if out.shape[0] > 1 else out[0]


def make_teacher_sampler(ckpt_dir: Path):
    params, model, extras = load_model(ckpt_dir, models.VelocityMLP)
    x_mu = jnp.asarray(extras["x_mu"])
    x_sd = jnp.asarray(extras["x_sd"])
    y_mu = jnp.asarray(extras["y_mu"])
    y_sd = jnp.asarray(extras["y_sd"])

    term = dfx.ODETerm(lambda t, y, args: model.apply({"params": params}, t, y, args))
    solver = dfx.Tsit5()
    saveat = dfx.SaveAt(t1=True)
    controller = dfx.PIDController(rtol=1e-5, atol=1e-5)

    def solve_single(x_cond_norm, z0):
        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=None,
            y0=z0,
            args=x_cond_norm,
            saveat=saveat,
            stepsize_controller=controller,
            max_steps=1_000_000,
        )
        return sol.ys.reshape(y_mu.shape)

    teacher_batch = jax.jit(jax.vmap(solve_single, in_axes=(0, 0)))
    return teacher_batch, x_mu, x_sd, y_mu, y_sd


def create_state(rng, model: StudentMLP, x_dim: int, z_dim: int, cfg: TrainCfg):
    x0 = jnp.zeros((1, x_dim))
    z0 = jnp.zeros((1, z_dim))
    params = model.init(rng, x0, z0)["params"]
    tx = optax.adam(cfg.lr)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def save_student(
    ckpt_dir: Path,
    params,
    student_cfg: StudentCfg,
    teacher_stats: dict[str, np.ndarray],
):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "params.msgpack").write_bytes(serialization.to_bytes(params))
    (ckpt_dir / "student_cfg.json").write_text(json.dumps(asdict(student_cfg)))
    np.savez(ckpt_dir / "extras.npz", **teacher_stats)


def build_configs(cfg: DistillCfg, latent_dim: int) -> tuple[StudentCfg, TrainCfg]:
    student_cfg = StudentCfg(
        hidden_dims=cfg.student_hidden_dims,
        latent_dim=latent_dim,
        out_dim=latent_dim,
    )
    train_cfg = TrainCfg(
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        steps=cfg.steps,
        log_every=cfg.log_every,
        seed=cfg.seed,
    )
    return student_cfg, train_cfg


def main():
    cfg = DISTILL_CFG
    teacher_ckpt = (SCRIPT_DIR / cfg.teacher_ckpt).resolve()
    output_ckpt = (SCRIPT_DIR / cfg.output_ckpt).resolve()

    teacher_batch, x_mu, x_sd, y_mu, y_sd = make_teacher_sampler(teacher_ckpt)
    latent_dim = int(np.asarray(y_mu).size)
    x_dim = int(np.asarray(x_mu).size)

    student_cfg, train_cfg = build_configs(cfg, latent_dim)
    model = StudentMLP(
        hidden_dims=student_cfg.hidden_dims,
        dim_out=student_cfg.out_dim,
    )

    key = jr.PRNGKey(train_cfg.seed)
    key, init_key = jr.split(key)
    state = create_state(
        init_key,
        model,
        x_dim=x_dim,
        z_dim=latent_dim,
        cfg=train_cfg,
    )

    @jax.jit
    def train_step(state, x_batch, z_batch, y_batch):
        def loss_fn(params):
            pred = model.apply({"params": params}, x_batch, z_batch)
            return jnp.mean(jnp.sum((pred - y_batch) ** 2, axis=-1))

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    print(f"Teacher checkpoint : {teacher_ckpt}")
    print(f"Student checkpoint : {output_ckpt}")
    print(f"Condition dim      : {x_dim}")
    print(f"Latent dim         : {latent_dim}")
    print(f"Batch size         : {train_cfg.batch_size}")

    for step in range(1, train_cfg.steps + 1):
        key, x_key, z_key = jr.split(key, 3)
        x_batch_norm = jr.normal(x_key, shape=(train_cfg.batch_size, x_dim))
        x_batch = x_mu + x_sd * x_batch_norm
        z_batch = jr.normal(z_key, shape=(train_cfg.batch_size, latent_dim))
        y_teacher_norm = teacher_batch(x_batch_norm, z_batch)

        state, loss = train_step(state, x_batch, z_batch, y_teacher_norm)

        if step % train_cfg.log_every == 0 or step == 1 or step == train_cfg.steps:
            print(f"step={step:6d} loss={float(loss):.6e}", flush=True)

    teacher_stats = {
        "x_mu": np.asarray(x_mu),
        "x_sd": np.asarray(x_sd),
        "y_mu": np.asarray(y_mu),
        "y_sd": np.asarray(y_sd),
    }
    save_student(output_ckpt, state.params, student_cfg, teacher_stats)
    print(f"Saved distilled student to {output_ckpt}")


if __name__ == "__main__":
    main()
