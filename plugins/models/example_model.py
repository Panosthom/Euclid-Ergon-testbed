"""Example model plugin: a small CNN for image classification.

Two surfaces are provided, both of which the platform understands:

1. ``ExampleCNNFactory`` -- a :class:`ModelFactory` ABC implementation, registered
   under the name ``"example_cnn"`` via the platform registry. This is the
   documented, discover-by-name contract.

2. ``build_model(...)`` -- a plain ``module:callable`` factory. The runtime
   resolves models inside the server/client subprocesses through a
   ``--model-source module:callable`` reference, so this is the symbol that is
   actually invoked end-to-end. Reference it from an ``ExperimentSpec`` with::

       model=ComponentRef("example_cnn", {"source": "plugins.models.example_model:build_model"})

The factory returns an ``nn.Module`` whose ``forward`` emits log-probabilities
(``log_softmax``), matching the platform's built-in classification convention.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from extensions.interfaces import ModelFactory
from registry import register_model

# (channels, height, width) for datasets we ship an example for. Anything else is
# treated as a square grayscale image inferred from input_dim.
_KNOWN_SHAPES: Dict[str, tuple[int, int, int]] = {
    "mnist": (1, 28, 28),
    "cifar10": (3, 32, 32),
}


def _infer_chw(dataset_name: str, input_dim: int) -> tuple[int, int, int]:
    name = (dataset_name or "").lower()
    if name in _KNOWN_SHAPES:
        return _KNOWN_SHAPES[name]
    side = int(round(math.sqrt(input_dim)))
    if side * side != input_dim:
        raise ValueError(
            f"Cannot infer image shape for dataset '{dataset_name}' from "
            f"input_dim={input_dim}. Add it to _KNOWN_SHAPES in example_model.py."
        )
    return (1, side, side)


class _SmallCNN(nn.Module):
    """Conv(32) -> Conv(64) -> GAP -> FC(num_classes), log-softmax output."""

    def __init__(self, channels: int, height: int, width: int, num_classes: int) -> None:
        super().__init__()
        self.channels, self.height, self.width = channels, height, width
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept either (N, C, H, W) or flattened (N, C*H*W).
        if x.dim() == 2:
            x = x.view(x.size(0), self.channels, self.height, self.width)
        x = self.features(x)
        x = torch.flatten(x, 1)
        return F.log_softmax(self.classifier(x), dim=1)


def build_model(
    *,
    dataset: str = "mnist",
    num_classes: int = 10,
    input_dim: int = 28 * 28,
    task: str = "classification",
    **_: Any,
) -> nn.Module:
    """Functional factory invoked by the runtime via ``--model-source``.

    The platform calls this with dataset metadata keyword args
    (``dataset``, ``meta``, ``task``, ``input_dim``, ``num_classes``,
    ``output_dim``); we accept the ones we need and ignore the rest.
    """
    if task != "classification":
        raise ValueError(f"example_cnn only supports classification, got task={task!r}")
    channels, height, width = _infer_chw(dataset, input_dim)
    return _SmallCNN(channels, height, width, num_classes)


class ExampleCNNFactory(ModelFactory):
    """ABC implementation mirroring :func:`build_model`."""

    def build(self, *, dataset_name: str, metadata: Optional[Dict[str, Any]] = None) -> nn.Module:
        meta = metadata or {}
        return build_model(
            dataset=dataset_name,
            num_classes=int(meta.get("num_classes", 10)),
            input_dim=int(meta.get("input_dim", 28 * 28)),
            task=str(meta.get("task", "classification")),
        )


# Discover-by-name registration (fires on import via the `register` bootstrap).
register_model("example_cnn", ExampleCNNFactory())
