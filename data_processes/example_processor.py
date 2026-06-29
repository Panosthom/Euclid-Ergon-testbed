"""Example data processor.

Run it through the platform's data-prep + sharding engine::

    python -m utils.prepare_data --dataset mydata --num-clients 4 --seed 2025 \
        --processor data_processes.example_processor:prepare_dataset

That writes ``data/mydata/{train.pt,test.pt,meta.json}`` and per-client shards
``data/mydata/shards/client_<id>.pt`` (IID by default; pass ``--dirichlet-alpha``
for non-IID). For built-in datasets (mnist, cifar10) you do NOT need a processor:

    python -m utils.prepare_data --dataset mnist --num-clients 4

Contract (keyword-only): return ``(train_x, train_y, test_x, test_y, meta)`` (or a
4-tuple, or a dict with those keys). Features are float tensors shaped
``(N, ...)``; classification labels are integer tensors shaped ``(N,)``. ``meta``
should carry ``task`` and the dims your model factory reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def prepare_dataset(
    *,
    dataset: str,
    data_root: Path,
    out_dir: Path,  # noqa: ARG001  -- prepare_data owns where shards land
    source: Optional[str],
    label_col: Optional[str],  # noqa: ARG001  -- used by tabular/CSV processors
    test_frac: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Produce a small synthetic 28x28 / 10-class set so the example runs offline.

    Replace the body with your own loading/cleaning/feature-engineering. Read raw
    inputs from ``source`` (or ``data_root``); keep everything reproducible from
    ``seed``.
    """
    g = torch.Generator().manual_seed(seed)

    num_classes = 10
    n_train, n_test = 2000, 400
    shape = (1, 28, 28)  # match the example_cnn model factory's expected input

    def _make(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        y = torch.randint(0, num_classes, (n,), generator=g)
        # Class-conditional mean so there is a learnable signal, not pure noise.
        x = torch.randn((n, *shape), generator=g) + (y.view(-1, 1, 1, 1) / num_classes)
        return x.float(), y.long()

    # `test_frac` is honoured by prepare_data for built-ins; here we materialise
    # an explicit split so the example is self-contained.
    train_x, train_y = _make(n_train)
    test_x, test_y = _make(n_test)

    meta: Dict[str, Any] = {
        "dataset": dataset,
        "task": "classification",
        "input_dim": int(shape[0] * shape[1] * shape[2]),
        "num_classes": num_classes,
        "source": str(source) if source else "",
        "test_frac": test_frac,
    }
    return train_x, train_y, test_x, test_y, meta
