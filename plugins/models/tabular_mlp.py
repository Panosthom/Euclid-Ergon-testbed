"""Tabular MLP for the maritime studies (works for classification AND regression).

The platform invokes the factory with dataset metadata (``task``, ``input_dim``,
``num_classes``, ``output_dim``). We branch on ``task``:
  - classification (AIS shiptype): output log-probabilities over ``num_classes``
    (matches the platform's NLLLoss convention);
  - regression (CBM decay coefficients): linear output of ``output_dim`` values.

Reference it from a spec with::

    model=ComponentRef("tabular_mlp", {"source": "plugins.models.tabular_mlp:build_model"})
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TabularMLP(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, *, classification: bool) -> None:
        super().__init__()
        self.classification = classification
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1) if x.dim() > 2 else x
        out = self.net(x.float())
        return F.log_softmax(out, dim=1) if self.classification else out


def build_model(
    *,
    task: str = "classification",
    input_dim: int = 16,
    num_classes: int = 2,
    output_dim: int = 1,
    **_: Any,
) -> nn.Module:
    """Functional factory invoked by the runtime via ``--model-source``."""
    if task == "classification":
        return _TabularMLP(input_dim, int(num_classes), classification=True)
    if task == "regression":
        return _TabularMLP(input_dim, int(output_dim), classification=False)
    raise ValueError(f"tabular_mlp supports classification/regression, got task={task!r}")


# Discover-by-name surface (optional; the runnable path is build_model above).
try:
    from extensions.interfaces import ModelFactory
    from registry import register_model

    class TabularMLPFactory(ModelFactory):
        def build(self, *, dataset_name: str, metadata: Optional[Dict[str, Any]] = None) -> nn.Module:
            m = metadata or {}
            return build_model(
                task=str(m.get("task", "classification")),
                input_dim=int(m.get("input_dim", 16)),
                num_classes=int(m.get("num_classes", 2)),
                output_dim=int(m.get("output_dim", 1)),
            )

    register_model("tabular_mlp", TabularMLPFactory())
except Exception:  # pragma: no cover
    pass
