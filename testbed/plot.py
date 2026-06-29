"""Plot a run's metrics into PNGs, ready for slides/thesis.

Reuses the platform's plotting (utils.plot_paper_results), so you get the same
figures the platform produces -- without writing into the read-only platform tree:

  - client_metrics.png : per-client + aggregated TRAIN / VAL / LOCAL-TEST
                         (loss, and accuracy or MAE depending on the task) vs round
  - global_metrics.png : centralized GLOBAL-TEST (loss + accuracy/MAE) vs round

    bash scripts/plot.sh                       # newest run in logs/
    bash scripts/plot.sh logs/<run_dir>        # a specific run

Reads logs/<run>/metrics/ (per-client CSVs) and logs/<run>/server_rounds.csv,
and writes PNGs to logs/<run>/plots/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[1]


def _latest_run() -> Path:
    runs = sorted(
        (p for p in (_WORKSPACE / "logs").glob("*/") if (p / "server_rounds.csv").exists()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise SystemExit("No run with server_rounds.csv found under logs/. Run a simulation first.")
    return runs[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a run's metrics to PNGs")
    parser.add_argument("run_dir", nargs="?", default=None, help="logs/<run_dir> (default: newest)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve() if args.run_dir else _latest_run()
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    from utils.plot_paper_results import (
        infer_task_from_metrics,
        load_metrics,
        plot_metrics,
        plot_server_global_metrics,
    )

    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plot] run={run_dir.name}")

    # Per-client + aggregated train/val/local-test. load_metrics() wants the dir
    # that holds client_*.csv directly, i.e. metrics/<dataset>/ -- find it.
    metrics_root = run_dir / "metrics"
    candidates = [metrics_root, *(d for d in metrics_root.glob("*/") if d.is_dir())]
    metrics_dir = next((d for d in candidates if list(d.glob("client_*.csv"))), metrics_root)
    try:
        all_metrics = load_metrics(metrics_dir)
        task = infer_task_from_metrics(all_metrics)
        plot_metrics(all_metrics, out_dir / "client_metrics.png", per_client=True, task=task)
        print(f"[plot] + {out_dir / 'client_metrics.png'}  (train/val/local-test, task={task})")
    except Exception as exc:  # noqa: BLE001
        print(f"[plot] - client_metrics skipped ({exc})")

    # Centralized global-test.
    server_csv = run_dir / "server_rounds.csv"
    try:
        plot_server_global_metrics(server_csv, out_dir / "global_metrics.png")
        print(f"[plot] + {out_dir / 'global_metrics.png'}  (global-test)")
    except Exception as exc:  # noqa: BLE001
        print(f"[plot] - global_metrics skipped ({exc})")

    print(f"[plot] done -> {out_dir}")


if __name__ == "__main__":
    main()
