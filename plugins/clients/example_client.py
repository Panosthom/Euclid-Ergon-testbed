"""Example client plugin.

Two surfaces, both understood by the platform:

1. ``make_client(...)`` -- the functional client factory. The runtime loads it via
   ``--client-fn-source module:callable`` whenever an ``ExperimentSpec`` references
   a ``client_fn`` whose name is not ``"default"``. Reference it with::

       client_fn=ComponentRef("plugins.clients.example_client:make_client",
                              {"local_epochs": 1, "batch_size": 32})

   It must return an ``fl.client.NumPyClient`` (or ``fl.client.Client``). The
   runtime supplies the model and pre-built data loaders; the factory only wires
   the training/eval loop.

2. ``ExampleClientFactory`` -- a :class:`ClientLogicFactory` ABC implementation,
   registered under ``"example_numpy_client"`` for discover-by-name use.

For most experiments you do NOT need a custom client at all -- reference
``client_fn=ComponentRef("default", {...})`` to use the platform's built-in
training loop. This example exists to show the override contract.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, List

import flwr as fl
import numpy as np
import torch
import torch.nn.functional as F


class ExampleNumPyClient(fl.client.NumPyClient):
    """Minimal NLL-loss training/eval loop over the provided loaders."""

    def __init__(self, *, model, trainloader, testloader, device, local_epochs: int) -> None:
        self.model = model.to(device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.local_epochs = max(1, int(local_epochs))

    # --- parameter <-> ndarray plumbing ------------------------------------
    def get_parameters(self, config) -> List[np.ndarray]:  # noqa: ARG002
        return [val.detach().cpu().numpy() for val in self.model.state_dict().values()]

    def _set_parameters(self, parameters: List[np.ndarray]) -> None:
        keys = self.model.state_dict().keys()
        state = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        )
        self.model.load_state_dict(state, strict=True)

    # --- Flower hooks ------------------------------------------------------
    def fit(self, parameters, config):  # noqa: ARG002
        self._set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        n = 0
        for _ in range(self.local_epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = F.nll_loss(self.model(images), labels)
                loss.backward()
                optimizer.step()
                n += labels.size(0)
        return self.get_parameters(config={}), n, {}

    def evaluate(self, parameters, config):  # noqa: ARG002
        self._set_parameters(parameters)
        self.model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                loss_sum += F.nll_loss(output, labels, reduction="sum").item()
                correct += (output.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        total = max(total, 1)
        return loss_sum / total, total, {"accuracy": correct / total}


def make_client(
    *,
    config: Any,
    task: str,  # noqa: ARG001
    split_mode: str,  # noqa: ARG001
    trainloader,
    valloader,  # noqa: ARG001
    testloader,
    model,
    device,
) -> fl.client.NumPyClient:
    """Functional client factory matching the platform's ``client_fn_source`` contract."""
    local_epochs = getattr(config, "local_epochs", 1)
    return ExampleNumPyClient(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        device=device,
        local_epochs=local_epochs,
    )


# --- Discover-by-name ABC surface (optional) -------------------------------
try:
    from extensions.interfaces import ClientLogicFactory, RuntimeContext
    from registry import register_client_fn

    class ExampleClientFactory(ClientLogicFactory):
        def build(self, *, context: RuntimeContext, client_id: int, model: Any, data: Any) -> Any:
            # The functional factory is the runnable surface; the ABC is provided
            # for symmetry with the documented API.
            raise NotImplementedError(
                "Use the functional factory plugins.clients.example_client:make_client "
                "via client_fn=ComponentRef(...) -- the runtime drives client logic "
                "through --client-fn-source."
            )

    register_client_fn("example_numpy_client", make_client)
except Exception:  # pragma: no cover - registry import is optional for the source path
    pass
