"""Centralized baseline for the CBM paper experiments.

Trains the same MLP (16→128→64→2) on ALL training data at once — no
federation — using identical hyperparameters to the FL clients:
  SGD lr=0.01 momentum=0.9, batch_size=64, 3 epochs per "round".

One "round equivalent" = 3 epochs on the full training set, matching the
computation budget of one FL round (4 clients × 3 local epochs each, but
with full data instead of 1/4 shards).

Outputs a CSV to logs/centralized_s{SEED}_{timestamp}/server_rounds.csv
with the same schema as the FL server_rounds.csv files so the plot scripts
can compare them directly.

Usage:
    python3 scripts/run_centralized.py          # uses FL_SEED (default 2025)
    FL_SEED=2026 python3 scripts/run_centralized.py
"""

from __future__ import annotations

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = int(os.environ.get("FL_SEED", "2025"))
NUM_ROUND_EQUIV = 80    # "rounds" to train (matches FL num_rounds)
EPOCHS_PER_ROUND = 3    # local_epochs used by FL clients
BATCH_SIZE = 64
LR = 0.01
MOMENTUM = 0.9
DATA_ROOT = Path(__file__).parent.parent / "data"
LOG_ROOT = Path(__file__).parent.parent / "logs"


def load_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from data_processes.cbm_processor import prepare_dataset
    train_x, train_y, test_x, test_y, _ = prepare_dataset(
        dataset="cbm",
        data_root=DATA_ROOT,
        out_dir=LOG_ROOT,
        source=None,
        label_col=None,
        test_frac=0.2,
        seed=SEED,
    )
    return train_x, train_y, test_x, test_y


def build_model(input_dim: int = 16, output_dim: int = 2) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 128), nn.ReLU(inplace=True),
        nn.Linear(128, 64), nn.ReLU(inplace=True),
        nn.Linear(64, output_dim),
    )


def evaluate_mae(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(x)
        mae = (pred - y).abs().mean().item()
    return mae


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"[centralized] seed={SEED}  rounds={NUM_ROUND_EQUIV}  epochs/round={EPOCHS_PER_ROUND}")

    train_x, train_y, test_x, test_y = load_data()
    print(f"[centralized] train={len(train_x)}  test={len(test_x)}")

    model = build_model(input_dim=train_x.shape[1], output_dim=train_y.shape[1])
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    criterion = nn.MSELoss()

    loader = DataLoader(TensorDataset(train_x, train_y),
                        batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"cbm_centralized_s{SEED}_{ts}"
    out_dir = LOG_ROOT / f"centralized_s{SEED}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "server_rounds.csv"

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["experiment_id", "round", "global_test_mae"])
        writer.writeheader()

        for rnd in range(1, NUM_ROUND_EQUIV + 1):
            model.train()
            for _ in range(EPOCHS_PER_ROUND):
                for xb, yb in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()

            mae = evaluate_mae(model, test_x, test_y)
            writer.writerow({"experiment_id": exp_id, "round": rnd, "global_test_mae": mae})
            fh.flush()

            if rnd % 10 == 0 or rnd == 1:
                print(f"[centralized] round {rnd:3d}/{NUM_ROUND_EQUIV}  MAE={mae:.6f}")

    print(f"[centralized] done — {csv_path}")


if __name__ == "__main__":
    main()
