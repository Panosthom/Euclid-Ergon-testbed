"""AIS maritime-traffic classification study (federated).

Classifies ship type from kinematic + size features.

    bash scripts/start.sh studies.ais_experiment:make_spec            # simulation
    bash scripts/run_on_testbed.sh studies.ais_experiment:make_spec   # real Pis
"""

from __future__ import annotations

from pathlib import Path

import register  # noqa: F401  -- registers plugins

from experiments.api import ComponentRef, ExperimentSpec


def make_spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="ais_shiptype_fedavg",
        dataset=ComponentRef(
            "ais",
            {
                "processor": "data_processes.ais_processor:prepare_dataset",
                "data_dir": "data/ais",
                "shard_mode": "iid",
            },
        ),
        model=ComponentRef("tabular_mlp", {"source": "plugins.models.tabular_mlp:build_model"}),
        aggregator=ComponentRef("fedavg"),
        client_fn=ComponentRef("default", {"local_epochs": 2, "batch_size": 128}),
        num_rounds=15,
        num_clients=4,
        seed=2025,
        server_address="127.0.0.1:8101",
        output_root=Path("logs"),
        metadata={"task": "classification", "description": "AIS ship-type classification"},
    )
