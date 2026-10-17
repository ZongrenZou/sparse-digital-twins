"""Input standardization (matches TensorStandardScaler behavior)."""

from __future__ import annotations

import numpy as np
import torch


class StandardScaler:
    def __init__(self, x_dim: int):
        self.x_dim = x_dim
        self.mu = np.zeros((1, x_dim), dtype=np.float64)
        self.sigma = np.ones((1, x_dim), dtype=np.float64)
        self.fitted = False
        self._torch_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def fit(self, data: np.ndarray) -> None:
        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0
        self.mu = mu
        self.sigma = sigma
        self.fitted = True
        self._torch_cache.clear()

    def transform_numpy(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mu) / self.sigma

    def transform_torch(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        key = (str(device), str(x.dtype))
        cached = self._torch_cache.get(key)
        if cached is None:
            cached = (
                torch.as_tensor(self.mu, dtype=x.dtype, device=device),
                torch.as_tensor(self.sigma, dtype=x.dtype, device=device),
            )
            self._torch_cache[key] = cached
        mu, sig = cached
        return (x - mu) / sig
