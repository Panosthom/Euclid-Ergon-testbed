"""Example dataset plugin: an MNIST loader returning a ``DatasetBundle``.

Implements the :class:`DatasetLoader` ABC and registers under ``"mnist"``.
``load`` returns a :class:`DatasetBundle` of in-memory tensors plus metadata
(``input_dim``, ``num_classes``, ``task``) that downstream model factories read.

How datasets are consumed:
- The in-process / discover-by-name path uses this loader directly.
- The runtime server/client subprocesses resolve a dataset by *name* and read
  per-client shards from ``data/<name>/`` (see ``data_dir`` in ``ComponentRef``).
  Use this loader, or your own preprocessing, to materialise those shards. MNIST
  and CIFAR-10 are built-in names the platform already understands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from extensions.interfaces import DatasetBundle, DatasetLoader, RuntimeContext
from registry import register_dataset_loader


class MnistLoader(DatasetLoader):
    """Load MNIST into memory via torchvision, returning a DatasetBundle."""

    def __init__(self, *, data_dir: str = "data/mnist") -> None:
        self.data_dir = data_dir

    def load(self, *, context: RuntimeContext) -> DatasetBundle:
        # torchvision is part of the pinned baseline; import lazily so the module
        # imports cheaply during registration.
        from torchvision import datasets, transforms

        root = Path(self.data_dir)
        root.mkdir(parents=True, exist_ok=True)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train = datasets.MNIST(str(root), train=True, download=True, transform=transform)
        test = datasets.MNIST(str(root), train=False, download=True, transform=transform)

        metadata: Dict[str, Any] = {
            "input_dim": 1 * 28 * 28,
            "num_classes": 10,
            "task": "classification",
            "dataset_name": context.dataset_name,
            "data_dir": str(root.resolve()),
        }
        return DatasetBundle(train_data=train, test_data=test, metadata=metadata)


register_dataset_loader("mnist", MnistLoader())
