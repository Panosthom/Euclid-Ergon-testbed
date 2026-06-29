"""AIS (maritime traffic) classification processor.

ais_data.csv columns: mmsi, navigationalstatus, sog, cog, heading, shiptype,
width, length, draught. Task: classify **shiptype** from kinematic + size
features. (mmsi is an identifier and is dropped.)

Prepare + shard with::

    python -m utils.prepare_data --dataset ais --num-clients 4 --seed 2025 \
        --processor data_processes.ais_processor:prepare_dataset

(The "start" flow runs this for you from the spec's dataset kwargs.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

_DEFAULT_REL = "AIS Dataset/ais_data.csv"
# Numeric features used by the baseline model (extend with more if you like).
_FEATURES = ["sog", "cog", "heading", "width", "length", "draught"]
_LABEL = "shiptype"
# Drop ultra-rare classes (fewer than this many samples) so the split is stable.
_MIN_CLASS_COUNT = 200


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
    label_col: Optional[str],
    test_frac: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    path = _resolve_path(data_root, source)
    if not path.exists():
        raise FileNotFoundError(f"AIS csv not found at {path} (pass --source to override)")
    label = label_col or _LABEL

    df = pd.read_csv(path)
    df = df.dropna(subset=[label])                       # need a label
    # Keep only classes with enough support.
    counts = df[label].value_counts()
    keep = counts[counts >= _MIN_CLASS_COUNT].index
    df = df[df[label].isin(keep)].reset_index(drop=True)

    classes = sorted(df[label].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = df[label].map(class_to_idx).to_numpy().astype(np.int64)
    X = df[_FEATURES].to_numpy().astype(np.float32)      # NaNs handled below

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    # Impute missing with TRAIN median, then standardize with TRAIN stats.
    med = np.nanmedian(X[train_idx], axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])
    mu = X[train_idx].mean(axis=0)
    sigma = X[train_idx].std(axis=0)
    sigma[sigma == 0] = 1.0
    Xn = (X - mu) / sigma

    train_x = torch.from_numpy(Xn[train_idx])
    train_y = torch.from_numpy(y[train_idx])
    test_x = torch.from_numpy(Xn[test_idx])
    test_y = torch.from_numpy(y[test_idx])

    meta: Dict[str, Any] = {
        "dataset": dataset,
        "task": "classification",
        "input_dim": len(_FEATURES),
        "num_classes": len(classes),
        "classes": classes,
    }
    return train_x, train_y, test_x, test_y, meta
