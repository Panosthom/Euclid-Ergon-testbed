"""Auto-shard the dataset for a run, deriving everything from the ExperimentSpec.

This is what "press start" runs first: it tells the platform's data engine to
(re)split the current data into ``num_clients`` per-client shards under
``data/<dataset>/shards/``. The student never calls ``utils.prepare_data`` by
hand -- the spec is the single source of truth.

Reads from ``spec.dataset.kwargs`` (all optional):
    processor       -> --processor  (module:callable for custom data; omit for built-ins)
    shard_mode      -> --shard-mode  (iid | temporal | day-block | dirichlet; default iid)
    dirichlet_alpha -> --dirichlet-alpha
    source          -> --source      (raw file path for tabular datasets)
and ``num_clients`` / ``seed`` / ``dataset.name`` from the spec itself.

    python -m testbed.prepare --experiment-source studies.example_experiment:make_spec
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# A custom processor (data_processes.x:prepare_dataset) is imported by the
# prepare_data subprocess, whose cwd is the platform root. Put this workspace on
# PYTHONPATH (OS-correct) so that import resolves.
_WORKSPACE = str(Path(__file__).resolve().parents[1])
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

from registry.manager import load_callable

import register  # noqa: F401  -- validate plugins import
from experiments.api import ExperimentSpec


def _load_spec(source: str) -> ExperimentSpec:
    spec = load_callable(source)()
    if not isinstance(spec, ExperimentSpec):
        raise TypeError(f"{source} must return an ExperimentSpec")
    return spec


def build_prepare_cmd(spec: ExperimentSpec) -> list[str]:
    kw = spec.dataset.kwargs
    cmd = [
        sys.executable, "-m", "utils.prepare_data",
        "--dataset", spec.dataset.name,
        "--num-clients", str(spec.num_clients),
        "--seed", str(spec.seed),
    ]
    if kw.get("processor"):
        cmd += ["--processor", str(kw["processor"])]
    if kw.get("shard_mode"):
        cmd += ["--shard-mode", str(kw["shard_mode"])]
    if kw.get("dirichlet_alpha") is not None:
        cmd += ["--dirichlet-alpha", str(kw["dirichlet_alpha"])]
    if kw.get("source"):
        cmd += ["--source", str(kw["source"])]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Shard the dataset for an ExperimentSpec")
    parser.add_argument("--experiment-source", default="studies.example_experiment:make_spec")
    args = parser.parse_args()

    spec = _load_spec(args.experiment_source)
    cmd = build_prepare_cmd(spec)
    print(f"[prepare] {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"  # the platform prints unicode; avoid cp1252 crashes on Windows
    env["PYTHONPATH"] = os.pathsep.join([_WORKSPACE, *([p] if (p := env.get("PYTHONPATH")) else [])])
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
