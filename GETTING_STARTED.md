# Getting started (students)

This is your workspace for building and running federated-learning experiments on
the edge testbed. You only edit files in **this** repo. The FL platform underneath
(server/client engine, the Raspberry Pis, orchestration) is a fixed dependency you
do not touch — you call it.

## 1. Connect

SSH into the **server** only (e.g. VS Code → Remote-SSH → the server host), and
open **your own copy** of this template. Everything happens here; you never log
into the client Pis.

> You each work in your **own clone** of the template (your own `plugins/`,
> `data/`, `logs/`) — not one shared folder. All clones depend on the **same**
> pinned platform, so experiments are comparable. Ask your instructor where your
> workspace lives if it isn't already set up.

## 2. The four things you edit

| You want to... | Edit |
| --- | --- |
| change the model | [plugins/models/example_model.py](plugins/models/example_model.py) |
| change aggregation (FedAvg, FedProx, ...) | [plugins/aggregators/example_aggregator.py](plugins/aggregators/example_aggregator.py) |
| change client training logic | [plugins/clients/example_client.py](plugins/clients/example_client.py) |
| load / preprocess your data | [plugins/datasets/example_dataset.py](plugins/datasets/example_dataset.py) and [data_processes/example_processor.py](data_processes/example_processor.py) |
| wire it all into an experiment | [studies/example_experiment.py](studies/example_experiment.py) (`make_spec()`) |

Each plugin **registers itself by a string name**; you reference those names in
`make_spec()`. After adding a new plugin file, add one import line to
[register.py](register.py).

## 3. Put your data in

Drop raw data under `data/` and point your processor at it (or use a built-in
dataset like `mnist` and skip the processor). The split is automatic at start time
— you do not run any data command by hand.

How your data is used (handled for you):
- the full dataset becomes `train.pt` + `test.pt`;
- `test.pt` stays on the **server** as the global test set → the **global accuracy**
  reported each round (`server_rounds.csv`);
- `train.pt` is split into one shard **per client**; each client gets only its
  shard and splits it locally into train/val/test before training.

Tune the split in your spec's `dataset` kwargs: `shard_mode="dirichlet"` +
`dirichlet_alpha=0.3` for non-IID, `processor="data_processes.x:prepare_dataset"`
for custom data.

## 4. Press start (simulation — your default)

One-time per workspace: copy the env file and set the two paths your admin gives
you (the shared platform location + how to activate the venv):

```bash
cp testbed/testbed.env.example testbed/testbed.env
# edit testbed/testbed.env -> PLATFORM_DIR and VENV_ACTIVATE
```

Then just run:

```bash
bash scripts/start.sh                                   # the example experiment
bash scripts/start.sh studies.my_experiment:make_spec   # your own
```

`start.sh` sources `activate.sh` for you (platform on `PYTHONPATH` + venv), so the
one command is self-contained. For interactive work (`python -c ...`,
`python register.py`) in your shell, run `source activate.sh` once yourself —
otherwise you'd get `ModuleNotFoundError: No module named 'registry'`.

This shards your data, then runs the whole federation **as local processes on the
server**. It is fast, needs no booking, and many students can run at the same time
(each gets its own port). Develop and debug here.

## 5. Run on the real Pis (when you're ready)

```bash
bash scripts/run_on_testbed.sh studies.my_experiment:make_spec
```

Same data, same spec — but the 4 **physical** client Pis do the work. They are
shared, so this reserves them for one run at a time (you'll be told if someone
else is running). Check availability:

```bash
bash scripts/testbed_lock.sh status
```

## 6. Read your results

Every run writes `logs/<name>_<UTCtimestamp>/`:
- `manifest.json` — the exact spec that ran (for your report);
- `server_rounds.csv` — accuracy/loss per round;
- `metrics/`, `checkpoints/`, per-process logs.

## Rules of thumb

- Reproducibility: keep an explicit `seed` in your spec; the manifest records it.
- Don't edit the platform dependency or the testbed topology — you don't need to,
  and it's shared.
- Start in simulation; only move to the Pis once your run is clean.

Full technical reference: [README.md](README.md).
