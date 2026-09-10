"""CBM naval-propulsion regression — Non-IID sharding (Dirichlet α=0.3), FedProx (μ=0.1).

    bash scripts/start.sh studies.cbm_noniid_fedprox:make_spec            # simulation
    bash scripts/run_on_testbed.sh studies.cbm_noniid_fedprox:make_spec   # real Pis
"""

from __future__ import annotations

from pathlib import Path

import register  # noqa: F401

from experiments.api import ComponentRef, ExperimentSpec


def make_spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="cbm_noniid_fedprox",
        dataset=ComponentRef(
            "cbm",
            {
                "processor": "data_processes.cbm_processor:prepare_dataset",
                "data_dir": "data/cbm",
                "shard_mode": "dirichlet",
                "dirichlet_alpha": 0.3,
            },
        ),
        model=ComponentRef("tabular_mlp", {"source": "plugins.models.tabular_mlp:build_model"}),
        aggregator=ComponentRef(
            "fedprox",
            {
                "source": "plugins.aggregators.example_aggregator:FedProxStrategy",
                "proximal_mu": 0.1,
            },
        ),
        client_fn=ComponentRef("default", {"local_epochs": 3, "batch_size": 64}),
        num_rounds=80,
        num_clients=4,
        seed=2025,
        server_address="127.0.0.1:8100",
        output_root=Path("logs"),
        metadata={"task": "regression", "shard_mode": "dirichlet", "dirichlet_alpha": 0.3, "strategy": "fedprox", "proximal_mu": 0.1},
    )
