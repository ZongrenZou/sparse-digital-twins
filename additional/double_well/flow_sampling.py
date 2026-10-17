# train_cfm_conditional_flax.py
# pip install jax jaxlib flax optax
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import os, json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from flax import serialization

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flax import linen as nn
from flax.training import train_state
import scipy.io as sio
import numpy as np
import diffrax as dfx

from ckpt_io import save_model, load_model
from scipy.integrate import solve_ivp
import models


def sample(
    key,
    params,
    x,
    z0=None,
    M=None,
):
    # Build once
    term = dfx.ODETerm(lambda t, y, args: velocity(params, model, t, y, args))

    def solve_single(z0i, x):
        sol = dfx.diffeqsolve(
            term,
            dfx.Tsit5(),
            t0=0.0,
            t1=1.0,
            dt0=None,
            y0=z0i,
            args=x,  # pass conditioning vector for this traj
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
            max_steps=1_000_000,
        )
        return sol.ys

    # If you want n samples for the SAME x:
    # z0 = jr.normal(key, [x.shape[0], 4])
    if z0 is None:
        z0 = np.random.normal(size=[x.shape[0], 4])
        z0 = jnp.array(z0)
    else:
        # z0 = jnp.tile(z0, [M, 1])
        z0 = jnp.tile(z0[None, :, :], [M, 1, 1])
        z0 = jnp.transpose(z0, [1, 0, 2])
        z0 = z0.reshape([-1, z0.shape[-1]])
    x_fixed = x  # (k,)
    zT = jax.vmap(solve_single, in_axes=(0, 0))(z0, x_fixed)
    return zT, z0


def velocity(params, model, t, z, x):
    """Single-trajectory: t:(), z:(d,), x:(k,) -> (d,)"""
    return model.apply({"params": params}, t, z, x)
    # return model.apply(params, t, z, x)


if __name__ == "__main__":

    params, model, extras = load_model("./checkpoints/cfm_m_10", models.VelocityMLP)
    x_mu = extras["x_mu"]
    x_sd = extras["x_sd"]
    y_mu = extras["y_mu"]
    y_sd = extras["y_sd"]

    data = sio.loadmat("./data/data_m_10.mat")

    m = data["m"].reshape([]).astype(np.int32)
    _x_data = data["Xs"].reshape([-1, m])
    _dx_data = data["dXs"].reshape([-1, 1])

    M = 1_000_000
    np.random.seed(77499981)
    idx = np.random.choice(_x_data.shape[0], _x_data.shape[0], replace=False)[:M]
    x_data = _x_data[idx]
    dx_data = _dx_data[idx]

    #### start sampling ####
    M = 10000
    idx = np.random.choice(x_data.shape[0], x_data.shape[0], replace=False)[:M]
    x_samples = x_data[idx, :]

    N = 1000
    z0 = np.random.normal(size=[N, 1])
    x = np.tile(x_samples, [N, 1]).reshape([N * M, x_samples.shape[1]])
    key = jr.PRNGKey(72819)
    z_samples, z0_samples = sample(
        key,
        params,
        x=(x - x_mu) / x_sd,
        z0=z0,
        M=M,
    )
    y_samples = z_samples[:, 0, :] * y_sd + y_mu

    sio.savemat(
        "./outputs/samples.mat",
        {
            "y_samples": y_samples,
            "z_samples": z0_samples,
            "x_samples": x,
            "x_mu": x_mu,
            "x_sd": x_sd,
            "y_mu": y_mu,
            "y_sd": y_sd,
        },
    )

    print("End main.")
