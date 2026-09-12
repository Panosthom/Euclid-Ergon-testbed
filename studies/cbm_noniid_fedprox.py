"""CBM naval-propulsion regression — Non-IID temporal sharding, FedProx (μ=0.1).

Each client receives a consecutive time-block of the data, creating realistic
heterogeneity (different operating conditions across time periods).
Dirichlet sharding is not supported for regression datasets.

    bash scripts/start.sh studies.cbm_noniid_fedprox:make_spec            # simulation
    bash scripts/run_on_testbed.sh studies.cbm_noniid_fedprox:make_spec   # real Pis
"""

from __future__ import annotations

import os
from pathlib import Path

import register  # noqa: F401

from experiments.api import ComponentRef, ExperimentSpec

_SEED = int(os.environ.get("FL_SEED", "2025"))


def make_spec() -> ExperimentSpec:
    return ExperimentSpec(
        name=f"cbm_noniid_fedprox_s{_SEED}",
        dataset=ComponentRef(
            "cbm",
            {
                "processor": "data_processes.cbm_processor:prepare_dataset_noniid",
                "data_dir": "data/cbm",
                "shard_mode": "temporal",
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
        seed=_SEED,
        server_address="127.0.0.1:8100",
        output_root=Path("logs"),
        metadata={"task": "regression", "shard_mode": "temporal", "strategy": "fedprox", "proximal_mu": 0.1},
    )
