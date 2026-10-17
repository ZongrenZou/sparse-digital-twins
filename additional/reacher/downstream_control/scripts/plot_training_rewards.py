#!/usr/bin/env python3
"""
Plot rewards/returns versus training iteration from a PETS logs.mat file.

Example:
  python scripts/plot_training_rewards.py --logmat log_reacher_test/2026-04-06--12:34:56/logs.mat
"""

from __future__ import annotations

import argparse
import os


def _squeeze_numeric(arr):
    import numpy as np

    arr = np.asarray(arr)
    return np.squeeze(arr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logmat", type=str, required=True, help="Path to logs.mat")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output image path. Default: <logmat_dir>/returns_vs_training.png",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Returns vs Training Iteration",
        help="Plot title",
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.io import loadmat

    data = loadmat(args.logmat)
    if "returns" not in data:
        raise KeyError("logs.mat does not contain a 'returns' field.")

    returns = _squeeze_numeric(data["returns"]).astype(np.float64)
    if returns.ndim == 0:
        returns = returns[None]

    train_idx = np.arange(1, returns.shape[0] + 1, dtype=np.int64)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(train_idx, returns, marker="o", linewidth=1.8, markersize=4)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Return")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.logmat)),
        "returns_vs_training.png",
    )
    fig.savefig(out_path, dpi=160)
    print("Saved plot to %s" % out_path)

    try:
        plt.show()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
