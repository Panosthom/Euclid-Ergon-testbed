"""CBM naval-propulsion regression — IID sharding, FedAvg aggregation.

    bash scripts/start.sh studies.cbm_iid_fedavg:make_spec            # simulation
    bash scripts/run_on_testbed.sh studies.cbm_iid_fedavg:make_spec   # real Pis
"""

from __future__ import annotations

from pathlib import Path

import register  # noqa: F401

from experiments.api import ComponentRef, ExperimentSpec


def make_spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="cbm_iid_fedavg",
        dataset=ComponentRef(
            "cbm",
            {
                "processor": "data_processes.cbm_processor:prepare_dataset",
                "data_dir": "data/cbm",
                "shard_mode": "iid",
            },
        ),
        model=ComponentRef("tabular_mlp", {"source": "plugins.models.tabular_mlp:build_model"}),
        aggregator=ComponentRef("fedavg"),
        client_fn=ComponentRef("default", {"local_epochs": 3, "batch_size": 64}),
        num_rounds=80,
        num_clients=4,
        seed=2025,
        server_address="127.0.0.1:8100",
        output_root=Path("logs"),
        metadata={"task": "regression", "shard_mode": "iid", "strategy": "fedavg"},
    )
