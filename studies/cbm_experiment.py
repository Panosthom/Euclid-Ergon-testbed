"""CBM naval-propulsion regression study (federated).

Predicts the two GT decay-state coefficients from 16 operating sensors.

    bash scripts/start.sh studies.cbm_experiment:make_spec            # simulation
    bash scripts/run_on_testbed.sh studies.cbm_experiment:make_spec   # real Pis
"""

from __future__ import annotations

from pathlib import Path

import register  # noqa: F401  -- registers plugins

from experiments.api import ComponentRef, ExperimentSpec


def make_spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="cbm_regression_fedavg",
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
        num_rounds=20,
        num_clients=4,
        seed=2025,
        server_address="127.0.0.1:8100",
        output_root=Path("logs"),
        metadata={"task": "regression", "description": "Naval propulsion CBM decay regression"},
    )
