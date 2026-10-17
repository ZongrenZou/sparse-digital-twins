"""Probabilistic ensemble of dynamics networks (PE model from PETS)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from dmbrl.torch_pets.scaler import StandardScaler


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class _DynamicsNet(nn.Module):
    """Single network: outputs mean (delta) and heteroskedastic log-variance."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: tuple[int, ...] = (500, 500, 500),
        weight_decays: tuple[float, ...] | None = None,
    ):
        super().__init__()
        self.out_dim = out_dim
        hds = tuple(hidden_dims)
        if weight_decays is None:
            if len(hds) == 3 and hds[0] == 500:
                weight_decays = (0.0001, 0.00025, 0.00025, 0.0005)
            else:
                raise ValueError("Provide weight_decays when hidden_dims differ from default (500,500,500).")
        self._weight_decays = tuple(weight_decays)
        if len(self._weight_decays) != len(hds) + 1:
            raise ValueError(
                "weight_decays must have len(hidden_dims)+1 entries (one per linear layer)."
            )

        layers: list[nn.Linear] = []
        prev = in_dim
        for h in hds:
            layers.append(nn.Linear(prev, h))
            prev = h
        layers.append(nn.Linear(prev, 2 * out_dim))
        self.linears = nn.ModuleList(layers)

    def weight_decay_loss(self) -> torch.Tensor:
        loss = torch.zeros((), device=self.linears[0].weight.device)
        for lin, wd in zip(self.linears, self._weight_decays):
            loss = loss + wd * 0.5 * (lin.weight**2).sum()
        return loss

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean (delta) and log-variance raw logits.

        The mean head is **linear** (no activation), matching `BNN._compile_outputs` when the
        last `FC` has `activation=None` (e.g. cartpole: only hidden layers use swish).
        """
        h = x
        for lin in self.linears[:-1]:
            h = swish(lin(h))
        raw = self.linears[-1](h)
        mean = raw[..., : self.out_dim]
        logvar_raw = raw[..., self.out_dim :]
        return mean, logvar_raw


class ProbabilisticEnsemble(nn.Module):
    """B bootstrap networks with Gaussian NLL training (matches BNN semantics)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_networks: int = 5,
        device: torch.device | None = None,
        learning_rate: float = 1e-3,
        hidden_dims: tuple[int, ...] = (500, 500, 500),
        weight_decays: tuple[float, ...] | None = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_networks = num_networks
        self.device = device or torch.device("cpu")
        self.scaler = StandardScaler(in_dim)

        self.nets = nn.ModuleList(
            [
                _DynamicsNet(
                    in_dim,
                    out_dim,
                    hidden_dims=hidden_dims,
                    weight_decays=weight_decays,
                )
                for _ in range(num_networks)
            ]
        )
        self.max_logvar = nn.Parameter(torch.ones(1, out_dim) * 0.5)
        self.min_logvar = nn.Parameter(-torch.ones(1, out_dim) * 10.0)

        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def _bounded_logvar(self, logvar_raw: torch.Tensor) -> torch.Tensor:
        logv = self.max_logvar - torch.nn.functional.softplus(
            self.max_logvar - logvar_raw
        )
        logv = self.min_logvar + torch.nn.functional.softplus(logv - self.min_logvar)
        return logv

    def forward_net(
        self, net_idx: int, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns mean and variance for one ensemble member. x is raw model input."""
        x = self.scaler.transform_torch(x, self.device)
        mean, logvar_raw = self.nets[net_idx](x)
        logvar = self._bounded_logvar(logvar_raw)
        var = torch.exp(logvar)
        return mean, var

    def forward_all_nets(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized ensemble forward pass.

        x: raw model input with shape (E, N, in_dim), where E == num_networks.
        Returns mean/var with shape (E, N, out_dim).
        """
        if x.ndim != 3 or x.shape[0] != self.num_networks:
            raise ValueError(
                "Expected x with shape (num_networks, batch, in_dim); got %r" % (tuple(x.shape),)
            )

        h = self.scaler.transform_torch(x, self.device)
        num_layers = len(self.nets[0].linears)
        for layer_idx in range(num_layers):
            weights = torch.stack(
                [net.linears[layer_idx].weight for net in self.nets], dim=0
            )
            biases = torch.stack(
                [net.linears[layer_idx].bias for net in self.nets], dim=0
            )
            h = torch.einsum("eni,eoi->eno", h, weights) + biases[:, None, :]
            if layer_idx < num_layers - 1:
                h = swish(h)

        mean = h[..., : self.out_dim]
        logvar = self._bounded_logvar(h[..., self.out_dim :])
        var = torch.exp(logvar)
        return mean, var

    def train_from_data(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 32,
        epochs: int = 5,
        holdout_ratio: float = 0.0,
        hide_progress: bool = False,
    ) -> None:
        """Train on (N, in_dim) / (N, out_dim) with bootstrap resampling per network."""
        self.train()
        inputs = np.asarray(inputs, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        n = inputs.shape[0]

        num_holdout = min(int(n * holdout_ratio), 5000) if holdout_ratio > 0 else 0
        perm = np.random.permutation(n)
        tr_sl = perm[num_holdout:]
        train_inputs = inputs[tr_sl]
        self.scaler.fit(train_inputs)
        in_tr = torch.as_tensor(
            self.scaler.transform_numpy(train_inputs),
            dtype=torch.float32,
            device=self.device,
        )
        targ_tr = torch.as_tensor(targets[tr_sl], dtype=torch.float32, device=self.device)
        n_train = tr_sl.size

        idxs = np.random.randint(0, n_train, size=(self.num_networks, n_train))

        it = range(epochs) if hide_progress else trange(epochs, desc="dynamics fit", unit="epoch")

        for _ in it:
            for batch_start in range(0, n_train, batch_size):
                batch_end = min(batch_start + batch_size, n_train)
                bcols = slice(batch_start, batch_end)
                loss_total = torch.zeros((), device=self.device)

                for e in range(self.num_networks):
                    rows = idxs[e, bcols]
                    xb = in_tr[rows]
                    yb = targ_tr[rows]
                    mean, logvar_raw = self.nets[e](xb)
                    logvar = self._bounded_logvar(logvar_raw)
                    inv_var = torch.exp(-logvar)
                    mse = ((mean - yb) ** 2 * inv_var).mean()
                    var_loss = logvar.mean()
                    nll = mse + var_loss
                    nll = nll + self.nets[e].weight_decay_loss()
                    loss_total = loss_total + nll

                reg = 0.01 * self.max_logvar.sum() - 0.01 * self.min_logvar.sum()
                loss_total = loss_total + reg
                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()

            idxs = self._shuffle_rows(idxs)

    @staticmethod
    def _shuffle_rows(arr: np.ndarray) -> np.ndarray:
        idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
        return arr[np.arange(arr.shape[0])[:, None], idxs]

    @torch.no_grad()
    def predict_factored(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (N, in_dim) raw inputs -> mean (E, N, out_dim), var (E, N, out_dim)."""
        return self.forward_all_nets(
            x.unsqueeze(0).expand(self.num_networks, -1, -1).contiguous()
        )
