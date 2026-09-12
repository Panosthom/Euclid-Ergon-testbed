"""CBM (naval propulsion) regression processor.

UCI "Condition Based Maintenance of Naval Propulsion Plants": data.txt has 18
whitespace-separated columns -- 16 operating features + 2 targets (GT Compressor
and GT Turbine decay-state coefficients). Task: regression, output_dim=2.

Prepare + shard with::

    python -m utils.prepare_data --dataset cbm --num-clients 4 --seed 2025 \
        --processor data_processes.cbm_processor:prepare_dataset

(The "start" flow runs this for you from the spec's dataset kwargs.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# Path to data.txt relative to --data-root (default "data/"); override with --source.
_DEFAULT_REL = "UCI CBM Dataset/data.txt"
_N_FEATURES = 16  # columns 1..16; columns 17,18 are the two regression targets


def _resolve_path(data_root: Path, source: Optional[str]) -> Path:
    if source:
        return Path(source).expanduser()
    return Path(data_root) / _DEFAULT_REL


def prepare_dataset(
    *,
    dataset: str,
    data_root: Path,
    out_dir: Path,  # noqa: ARG001
    source: Optional[str],
    label_col: Optional[str],  # noqa: ARG001
    test_frac: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    path = _resolve_path(data_root, source)
    if not path.exists():
        raise FileNotFoundError(f"CBM data.txt not found at {path} (pass --source to override)")

    arr = np.loadtxt(path)  # (N, 18)
    if arr.shape[1] < _N_FEATURES + 1:
        raise ValueError(f"Expected >= {_N_FEATURES + 1} columns in {path}, got {arr.shape[1]}")
    X = arr[:, :_N_FEATURES].astype(np.float32)
    Y = arr[:, _N_FEATURES:].astype(np.float32)  # (N, 2) decay coefficients

    # Reproducible train/test split.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    # Standardize features using TRAIN statistics only (avoid leakage); guard
    # against constant columns (some sensors are fixed across the dataset).
    mu = X[train_idx].mean(axis=0)
    sigma = X[train_idx].std(axis=0)
    sigma[sigma == 0] = 1.0
    Xn = (X - mu) / sigma

    train_x = torch.from_numpy(Xn[train_idx])
    train_y = torch.from_numpy(Y[train_idx])
    test_x = torch.from_numpy(Xn[test_idx])
    test_y = torch.from_numpy(Y[test_idx])

    meta: Dict[str, Any] = {
        "dataset": dataset,
        "task": "regression",
        "input_dim": _N_FEATURES,
        "output_dim": int(Y.shape[1]),  # 2
    }
    return train_x, train_y, test_x, test_y, meta


def prepare_dataset_noniid(
    *,
    dataset: str,
    data_root: Path,
    out_dir: Path,  # noqa: ARG001
    source: Optional[str],
    label_col: Optional[str],  # noqa: ARG001
    test_frac: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Non-IID variant: training data sorted by lever position (col 0).

    Combined with shard_mode="temporal", each client receives a distinct
    operating regime (low → high engine load), simulating vessels that
    operate under heterogeneous conditions. This creates genuine feature
    distribution shift (covariate shift) across clients.

    Col 0 = lever position (lp): the primary operating-point indicator in
    the CBM dataset — low lp ≈ anchored/slow, high lp ≈ full speed.
    """
    # Reuse the standard load + split + normalise pipeline.
    train_x, train_y, test_x, test_y, meta = prepare_dataset(
        dataset=dataset,
        data_root=data_root,
        out_dir=out_dir,
        source=source,
        label_col=label_col,
        test_frac=test_frac,
        seed=seed,
    )
    # Sort training samples by lever position (ascending) so that
    # temporal sharding assigns each client a different load regime.
    sort_idx = train_x[:, 0].argsort()
    return train_x[sort_idx], train_y[sort_idx], test_x, test_y, meta
