# ckpt_io.py
import os, json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from flax import serialization

# --- Save / Load params + config (+ extras like normalization stats) ---

def save_model(
    ckpt_dir: str | os.PathLike,
    params: Any,
    model,                       # instance of VelocityMLP
    model_cfg,                   # your ModelCfg dataclass or dict
    extras: Dict[str, np.ndarray] | None = None,
) -> None:
    """
    Saves:
      - params to params.msgpack
      - model (hidden_dims, dim_out) to model_cfg.json
      - any extras (np arrays) to extras.npz
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Serialize params
    b = serialization.to_bytes(params)
    (ckpt_dir / "params.msgpack").write_bytes(b)

    # Serialize minimal model architecture
    if is_dataclass(model_cfg):
        model_cfg = asdict(model_cfg)
    arch = {
        "hidden_dims": list(model_cfg.get("hidden_dims", getattr(model, "hidden_dims", []))),
        "dim_out": int(getattr(model, "dim_out", model_cfg.get("dim_out"))),
    }
    (ckpt_dir / "model_cfg.json").write_text(json.dumps(arch))

    # Optional: extras (e.g., normalization stats)
    if extras:
        # Only arrays or array-like values
        np.savez(ckpt_dir / "extras.npz", **{k: np.asarray(v) for k, v in extras.items()})

def load_model(
    ckpt_dir: str | os.PathLike,
    VelocityMLP_cls,              # pass the class object VelocityMLP
):
    """
    Returns:
      params, model, extras
    """
    ckpt_dir = Path(ckpt_dir)

    # Restore params without needing shapes
    params = serialization.msgpack_restore((ckpt_dir / "params.msgpack").read_bytes())

    # Restore architecture and rebuild module
    arch = json.loads((ckpt_dir / "model_cfg.json").read_text())
    model = VelocityMLP_cls(hidden_dims=tuple(arch["hidden_dims"]), dim_out=int(arch["dim_out"]))

    # Optional extras
    extras_path = ckpt_dir / "extras.npz"
    extras = dict(np.load(extras_path, allow_pickle=False)) if extras_path.exists() else {}

    return params, model, extras
