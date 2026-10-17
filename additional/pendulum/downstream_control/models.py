from typing import Tuple
import jax.numpy as jnp
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
