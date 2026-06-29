# Student guide — running experiments on the FL edge testbed

This template is **your workspace** for federated-learning experiments. You take
your own copy of it, write your model / data / strategy, and run it — first in
simulation, then on the real Raspberry Pi edge clients. You never touch the
platform or the client Pis; the testbed handles all of that.

## The golden rule

> **You edit research code. You do NOT edit the plumbing.**

| ✅ Yours to edit | ⛔ Leave as-is (testbed plumbing) |
| --- | --- |
| `plugins/models/` — your model | `activate.sh` |
| `plugins/aggregators/` — aggregation strategy; swap via `aggregator` `source` (FedProx/FedAdam/custom) | `scripts/` |
| `plugins/clients/` — client training logic | `testbed/` |
| `plugins/datasets/` + `data_processes/` — your data | `register.py` (just add one import line) |
| `studies/<name>.py` — your experiment (`make_spec()`) | `pyproject.toml` / `requirements.txt` |

If you change the plumbing, your runs may stop working on the testbed. Keep it
stock and updates stay painless.

## 0. Your account (the admin sets this up)

The **admin** creates your account on the server and already clones this template
into `~/fl-edge-template` with `testbed/testbed.env` seeded for you — you don't run
`git clone` or touch `testbed.env`; the defaults point at the shared platform and
deploy key. You just receive a username + password (or key) and the server host.

## 1. Connect — and pick how you want to work

You always **run** on the server (the platform, your data, and the 4 client Pis
live there — even simulation needs the server). You choose only where you **edit**:

**A. Edit directly on the server** (simplest)

```text
VS Code → Remote-SSH → <user>@ergon.ee.duth.gr   →  open ~/fl-edge-template
```
Edit and run in the same place. You never log into the client Pis.

**B. Edit on your own PC, run on the server** (recommended if you use a local IDE
or Claude) — code travels through *your* GitHub fork:

1. **Fork** this public template on GitHub → `github.com/<you>/<fork>`.
2. On your **PC**: `git clone https://github.com/<you>/<fork>.git` and develop
   locally with whatever tools you like.
3. Point your **server** checkout at your fork (once):
   ```bash
   cd ~/fl-edge-template
   git remote set-url origin https://github.com/<you>/<fork>.git
   ```
4. Loop: edit on PC → `git push` → on the server `git pull` → run.

Either way, **runs happen on the server**: your dataset/shards and the platform
engine are only there, so you can't run (not even sim) on your PC alone.

## 2. Write your experiment

1. Add a model in `plugins/models/`, a dataset/processor in `plugins/datasets/` +
   `data_processes/`, a strategy in `plugins/aggregators/`, (optional) client logic
   in `plugins/clients/`. Each file **registers itself by name** at the bottom.
2. Add one import line for each new plugin file to `register.py`.
3. Wire them in a spec: copy `studies/example_experiment.py` to
   `studies/<your_name>.py` and edit `make_spec()` — reference your plugins by
   name, set `num_rounds`, `num_clients`, `seed`.

## 3. Run in simulation (your default)

Everything runs as local processes on the server — fast, no booking, many students
at once. It auto-shards your data and streams per-round progress:

```bash
bash scripts/start.sh studies.<your_name>:make_spec
```

(`bash scripts/start.sh` alone runs the bundled `example_experiment`.)

Develop and debug here until your run is clean.

## 4. Run on the real Pis (when ready)

Same code, same spec — now the 4 physical client Pis do the work. They are shared,
so this reserves them for one run at a time:

```bash
# check it's free:
bash scripts/testbed_lock.sh status
# dry-run first (prints the remote commands, runs the deploy, does NOT start FL):
bash scripts/run_on_testbed.sh studies.<your_name>:make_spec --dry-run
# the real thing:
bash scripts/run_on_testbed.sh studies.<your_name>:make_spec
```

It locks the clients → shards your data → ships your code + shards to all nodes →
starts the server + 4 clients → waits → **collects the results into your
workspace** → unlocks.

## 5. Read your results

- **Simulation:** `logs/<name>_<UTCtimestamp>/` — `server_rounds.csv`
  (`global_test_accuracy` per round), `manifest.json`, `metrics/`, `global_model.pt`.
- **Pi run:** `logs/pi_run_<timestamp>/` — `server.log` (round history + global
  accuracy), `client_<id>.log`, `global_model.pt`.

```bash
cat logs/$(ls -t logs/ | head -1)/server_rounds.csv     # sim
# or, for a Pi run, read server.log
```

**Plots for slides/thesis** — turn a run into figures (train / val / local-test /
global-test, loss + accuracy/MAE vs round):

```bash
bash scripts/plot.sh                     # newest run
bash scripts/plot.sh logs/<run_dir>      # a specific run
# -> logs/<run>/plots/{client_metrics.png, global_metrics.png}
```

## How your data is split (handled for you)

- the full dataset → `train.pt` + `test.pt`;
- `test.pt` stays on the **server** as the global test set → the **global accuracy**
  reported each round;
- `train.pt` → one **shard per client**; each client gets only its shard and splits
  it locally into train/val/test before training.

Tune it from your spec's `dataset` kwargs: `shard_mode="dirichlet"` +
`dirichlet_alpha=0.3` for non-IID, `processor="data_processes.x:prepare_dataset"`
for custom data.

## Reproducibility

Keep an explicit `seed` in your spec (`manifest.json` records the full resolved
spec). For multi-seed studies, run the same spec across a seed sweep.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No module named studies.xxx` | the arg is `studies.<file>:make_spec` and `studies/<file>.py` must exist |
| `No module named 'registry'` | run via `bash scripts/start.sh` (it sources `activate.sh`); for ad-hoc commands `source activate.sh` first |
| `No module named 'plugins'` (in a run) | you ran a raw `python -m ...`; use `scripts/start.sh` / `run_on_testbed.sh` |
| `Testbed busy` on a Pi run | someone else holds the clients; `bash scripts/testbed_lock.sh status` |

Full technical reference: [README.md](README.md).
