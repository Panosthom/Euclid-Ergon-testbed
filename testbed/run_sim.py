"""Run an ExperimentSpec as a LOCAL simulation on the server (no physical Pis).

This is the educational default: ``submit_experiment`` spawns the FL server plus
``num_clients`` client processes on the current machine. Many students can run at
once because each run gets its own free TCP port (no shared lock needed).

    python -m testbed.run_sim --experiment-source studies.example_experiment:make_spec

By default it **follows** the server log live, so per-round progress
(``Round N: global_test_accuracy=...``) streams into your terminal -- the platform
engine otherwise redirects subprocess output to log files. Use ``--no-follow`` for
a quiet run. Outputs land in ``logs/<name>_<UTCtimestamp>/`` either way.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import socket
import sys
import threading
import time
from pathlib import Path

# The platform engine spawns the server/client as subprocesses with cwd set to
# the *platform* root, so our `plugins`/`studies` packages are only importable
# there if this workspace is on PYTHONPATH with an OS-correct absolute path.
# Guarantee that for our children before importing anything else.
_WORKSPACE = str(Path(__file__).resolve().parents[1])
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)
os.environ["PYTHONPATH"] = os.pathsep.join(
    [_WORKSPACE, *([p] if (p := os.environ.get("PYTHONPATH")) else [])]
)

from registry.manager import load_callable

import register  # noqa: F401
from experiments.api import ExperimentSpec
from experiments.submission import submit_experiment


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _load_spec(source: str) -> ExperimentSpec:
    spec = load_callable(source)()
    if not isinstance(spec, ExperimentSpec):
        raise TypeError(f"{source} must return an ExperimentSpec")
    return spec


def _follow_server_log(output_root: Path, before: set[Path], worker: threading.Thread) -> None:
    """Stream the new run's process_server.log to stdout while `worker` runs.

    The engine creates logs/<name>_<timestamp>/ after submit starts, so we detect
    the new directory, then tail its server log until the worker thread finishes.
    """
    def _new_dir() -> Path | None:
        try:
            dirs = {p for p in output_root.iterdir() if p.is_dir()}
        except FileNotFoundError:
            return None
        new = dirs - before
        return max(new, key=lambda p: p.stat().st_mtime) if new else None

    run_dir = None
    while run_dir is None and (worker.is_alive() or _new_dir()):
        run_dir = _new_dir()
        if run_dir is None:
            time.sleep(0.3)
    if run_dir is None:
        return

    log_path = run_dir / "process_server.log"
    while not log_path.exists() and worker.is_alive():
        time.sleep(0.2)
    if not log_path.exists():
        return

    print(f"[sim] --- live server log ({log_path.name}); run continues even if you stop watching ---")
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        while True:
            line = fh.readline()
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
                continue
            if not worker.is_alive():
                rest = fh.read()
                if rest:
                    sys.stdout.write(rest)
                    sys.stdout.flush()
                break
            time.sleep(0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an ExperimentSpec as a local simulation")
    parser.add_argument("--experiment-source", default="studies.example_experiment:make_spec")
    parser.add_argument("--port", type=int, default=0, help="server port (0 = pick a free one)")
    parser.add_argument("--follow", dest="follow", action="store_true", default=True,
                        help="stream per-round server log to the terminal (default)")
    parser.add_argument("--no-follow", dest="follow", action="store_false",
                        help="quiet run; read progress from logs/<run>/ afterwards")
    args = parser.parse_args()

    spec = _load_spec(args.experiment_source)

    # The platform spawns clients with cwd=<platform root> and resolves the dataset
    # dir relative to it -- but in simulation the shards live in THIS workspace
    # (the student can't write into the read-only platform tree). Pin data_dir to an
    # absolute path under the workspace so the clients find the shards we prepared.
    ws_data = (Path(_WORKSPACE) / "data" / spec.dataset.name).resolve()
    spec = dataclasses.replace(
        spec,
        dataset=dataclasses.replace(
            spec.dataset, kwargs={**spec.dataset.kwargs, "data_dir": str(ws_data)}
        ),
    )
    # Custom datasets resolve their task/dims from <DATA_DIR>/meta.json (built-ins
    # like mnist don't need this). The engine propagates env to the client, so set
    # DATA_DIR here or the client raises "Unknown dataset '<name>'". Likewise the
    # server's centralized global test set defaults to <platform>/data/<name>/test.pt;
    # point it at our workspace copy via FL_GLOBAL_TEST_PATH.
    os.environ["DATA_DIR"] = str(ws_data)
    os.environ["FL_GLOBAL_TEST_PATH"] = str(ws_data / "test.pt")

    # CodeCarbon: the platform client reads ENABLE_CODECARBON / CODECARBON_OUTPUT_DIR
    # from the environment, and the engine propagates os.environ to the spawned
    # client processes -- so flipping it here turns energy tracking on for the sim.
    if spec.client_fn.kwargs.get("enable_codecarbon"):
        os.environ["ENABLE_CODECARBON"] = "1"
        out = spec.client_fn.kwargs.get("codecarbon_output_dir")
        if out:
            os.environ["CODECARBON_OUTPUT_DIR"] = str(out)
        print("[sim] CodeCarbon enabled (emissions.csv per client)")

    # Give each concurrent student run a distinct port so they don't collide.
    port = args.port or _free_port()
    spec = dataclasses.replace(spec, server_address=f"127.0.0.1:{port}")
    print(f"[sim] {spec.name} on 127.0.0.1:{port} | clients={spec.num_clients} rounds={spec.num_rounds}")

    if not args.follow:
        result = submit_experiment(spec)
    else:
        # Run the (blocking) engine in a worker thread and tail the server log here.
        output_root = Path(spec.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        before = {p for p in output_root.iterdir() if p.is_dir()}

        holder: dict = {}

        def _run() -> None:
            try:
                holder["result"] = submit_experiment(spec)
            except BaseException as exc:  # noqa: BLE001 - re-raised in main thread
                holder["error"] = exc

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        _follow_server_log(output_root, before, worker)
        worker.join()
        if "error" in holder:
            raise holder["error"]
        result = holder["result"]

    print(f"[sim] experiment_id={result.experiment_id}")
    print(f"[sim] run_dir={result.run_dir}")


if __name__ == "__main__":
    main()
