"""Example experiment factory.

``make_spec()`` returns a fully-wired :class:`ExperimentSpec`. Run it with::

    python -m experiments.run_submitted \
        --experiment-source studies.example_experiment:make_spec

(``experiments.run_submitted`` is the *platform's* runner; ``studies`` is this
template's spec package — see studies/__init__.py for why it is not named
``experiments``.)

Importing this module first imports ``register``, so every plugin is discoverable
by name before the spec is built or submitted.
"""

from __future__ import annotations

from pathlib import Path

import register  # noqa: F401  -- fires all plugin register_* calls on import

from experiments.api import ComponentRef, ExperimentSpec


def make_spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="example_mnist_fedavg",
        # Built-in dataset; shards are read from data/mnist/shards/.
        # The "start" flow auto-shards from these kwargs (see testbed/prepare.py):
        #   - custom data: add "processor": "data_processes.example_processor:prepare_dataset"
        #   - non-IID:     add "shard_mode": "dirichlet", "dirichlet_alpha": 0.3
        dataset=ComponentRef("mnist", {"data_dir": "data/mnist"}),
        # `source` points the runtime at our functional model factory.
        model=ComponentRef(
            "example_cnn",
            {"source": "plugins.models.example_model:build_model"},
        ),
        # FedAvg is the default. To swap the aggregation algorithm, point at a
        # Strategy subclass, e.g.:
        #   aggregator=ComponentRef("fedprox",
        #       {"source": "plugins.aggregators.example_aggregator:FedProxStrategy"})
        aggregator=ComponentRef("fedavg"),
        # Use the platform's built-in training loop. To inject the custom client
        # instead, swap to:
        #   client_fn=ComponentRef(
        #       "plugins.clients.example_client:make_client",
        #       {"local_epochs": 1, "batch_size": 32},
        #   )
        # client_fn.kwargs tune the built-in training loop (sim + Pi):
        #   local_epochs, batch_size, optimizer ("adam"|"sgd"), lr, weight_decay,
        #   grad_clip, enable_codecarbon. Omitted keys use the platform defaults
        #   (adam, lr 1e-3). Energy emissions -> data/<dataset>/codecarbon/client_<id>/.
        client_fn=ComponentRef(
            "default",
            {"local_epochs": 1, "batch_size": 32, "optimizer": "adam", "lr": 1e-3,
             "enable_codecarbon": False},
        ),
        num_rounds=10,
        num_clients=4,
        server_address="127.0.0.1:8081",
        seed=2025,
        output_root=Path("logs"),
        metadata={"description": "Template MNIST + FedAvg smoke experiment"},
    )


if __name__ == "__main__":
    # Convenience: `python -m studies.example_experiment` submits directly.
    from experiments.submission import submit_experiment

    result = submit_experiment(make_spec())
    print(f"experiment_id={result.experiment_id}")
    print(f"run_dir={result.run_dir}")
