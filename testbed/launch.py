"""Bridge: launch an ``ExperimentSpec`` on the 5-Pi testbed via the platform's
distributed orchestrator.

Your spec (``studies/...:make_spec``) is the single source of truth. This module
derives the launch parameters from it and delegates to ``controller.start_all``
(platform-core), which SSHes into the server + 4 client Pis and starts
``server.server`` / ``client.client`` for you.

    python -m testbed.launch \
        --experiment-source studies.example_experiment:make_spec --wait

Run it from the server Pi, inside your student workspace, with the platform on
PYTHONPATH. Usually you invoke it through ``scripts/run_on_testbed.sh`` so the
exclusive-access lock and code/shard deploy happen first.

IMPORTANT differences from local simulation:
- Network addresses come from the physical inventory in
  ``controller/config.json`` (each node's ``fl_ip`` + ``server_port``), NOT from
  ``spec.server_address`` (which is only used by local ``submit_experiment``).
- ``controller.start_all`` forwards ``--model-source`` but NOT
  ``--client-fn-source``. A custom functional ``client_fn`` therefore does not
  take effect on the Pi path today -- use ``client_fn=ComponentRef("default", ...)``
  for Pi runs, or extend the platform's ``controller.start_all``. This launcher
  warns when the spec would be silently downgraded.
"""

from __future__ import annotations

import argparse
import time

from registry.manager import load_callable

# Validate that every plugin imports/registers before we attempt a run. The
# remote processes re-import them, but failing fast locally is cheaper.
import register  # noqa: F401
from experiments.api import ExperimentSpec


def _load_spec(source: str) -> ExperimentSpec:
    spec = load_callable(source)()
    if not isinstance(spec, ExperimentSpec):
        raise TypeError(f"{source} must return an ExperimentSpec")
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch an ExperimentSpec on the Pi testbed")
    parser.add_argument(
        "--experiment-source",
        default="studies.example_experiment:make_spec",
        help="module:callable returning an ExperimentSpec",
    )
    parser.add_argument("--wait", action="store_true", help="block until all rounds finish")
    parser.add_argument("--dry-run", action="store_true", help="print remote commands, do not execute")
    args = parser.parse_args()

    spec = _load_spec(args.experiment_source)

    if spec.client_fn.name != "default":
        print(
            "[testbed] WARNING: spec uses a custom client_fn "
            f"({spec.client_fn.name!r}), but controller.start_all does not forward "
            "--client-fn-source. The Pi clients will use the platform's default "
            "client loop. Use client_fn=default for Pi runs, or extend start_all."
        )

    model_source = spec.model.kwargs.get("source")
    aggregator_source = spec.aggregator.kwargs.get("source")
    cfkw = spec.client_fn.kwargs
    local_epochs = cfkw.get("local_epochs")
    batch_size = cfkw.get("batch_size")
    optimizer = cfkw.get("optimizer")
    lr = cfkw.get("lr")
    weight_decay = cfkw.get("weight_decay")
    grad_clip = cfkw.get("grad_clip")
    enable_cc = cfkw.get("enable_codecarbon")
    experiment_id = f"{spec.name}_{time.strftime('%Y%m%d_%H%M%S')}"

    print(f"[testbed] experiment_id={experiment_id}")
    print(f"[testbed] dataset={spec.dataset.name} model={spec.model.name} "
          f"model_source={model_source or '<registry>'}")
    print(f"[testbed] num_rounds={spec.num_rounds} min_clients={spec.num_clients} seed={spec.seed}")

    # Imported lazily so `python -m testbed.launch --help` works without paramiko.
    from controller import start_all

    start_all.main(
        dry_run=args.dry_run,
        dataset_override=spec.dataset.name,
        experiment_id_override=experiment_id,
        num_rounds_override=spec.num_rounds,
        min_clients_override=spec.num_clients,
        seed_override=spec.seed,
        local_epochs_override=int(local_epochs) if local_epochs is not None else None,
        batch_size_override=int(batch_size) if batch_size is not None else None,
        optimizer_override=str(optimizer) if optimizer else None,
        lr_override=float(lr) if lr is not None else None,
        weight_decay_override=float(weight_decay) if weight_decay is not None else None,
        grad_clip_override=float(grad_clip) if grad_clip is not None else None,
        model_override=spec.model.name if spec.model.name not in {"", "auto"} else None,
        model_source_override=model_source,
        aggregator_source_override=aggregator_source,
        enable_codecarbon_override=bool(enable_cc) if enable_cc is not None else None,
        wait_for_completion_flag=args.wait,
    )


if __name__ == "__main__":
    main()
