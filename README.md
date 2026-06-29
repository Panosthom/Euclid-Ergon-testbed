# fl-edge-template

Researcher template for the **FL Edge** federated-learning platform (Flower, on
Raspberry Pi edge clients + local simulation). You write plugins and experiment
specs here; the platform is consumed as a pinned, read-only dependency. There is
no GUI and no platform code in this repo — only your work product.

The loop:

> add a plugin under `plugins/` → register it → reference it by name in an
> `ExperimentSpec` → run via `run_submitted` → read the timestamped
> `logs/<name>_<UTCtimestamp>/` run with its `manifest.json`.

**New here?** Students: start with [STUDENT_GUIDE.md](STUDENT_GUIDE.md) (what you
edit, how to run in sim and on the Pis). Operators provisioning the testbed: see
[docs/ADMIN_SETUP.md](docs/ADMIN_SETUP.md) and onboard a student in one command
with [scripts/admin/create_student.sh](scripts/admin/create_student.sh). This
README is the full technical reference.

---

## Platform packaging (read first)

This template depends on the platform's **v2 public API** (`registry/`,
`extensions/`, `experiments/api.py`, `submission.py`, `config_adapter.py`,
`run_submitted.py`, `core/`, `runtime/`). Three facts to know before deploying:

1. **The v2 API is on the platform's `origin/main`** at commit `1d17e87` (pinned in
   `pyproject.toml`/`requirements.txt`). Every Pi must be pulled up to this commit —
   older clones (e.g. the Pi was on `b317cc3`) lack `registry/` and `import registry`
   fails there. `git -C <platform> fetch && git pull` on each Pi to align.
2. **The platform is not yet pip-installable** (no root `pyproject.toml`/`setup.py`),
   so the direct git reference in `pyproject.toml` won't install. Consume the
   platform via the **submodule + `PYTHONPATH`** fallback below; `activate.sh`
   handles the `PYTHONPATH` for you from `testbed/testbed.env`.
3. **Top-level package names.** The platform owns top-level `experiments`/`registry`/
   `extensions`, so this template puts its spec factories in `studies`, **not**
   `experiments` (a second `experiments` would shadow the platform's). If you later
   namespace the platform (e.g. `fedplatform.experiments`), rename `studies` freely.

The pin in `pyproject.toml`, `requirements.txt`, the submodule, and the commit on
the Pi must all be the **same** — one source of version truth.

---

## Install

### Option A — once the platform is pip-installable (target state)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .          # pulls the pinned platform + flwr/torch/numpy baseline
```

### Option B — submodule + editable path (works today)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Vendor the platform at the pinned commit alongside the template.
git submodule add https://github.com/Panosthom/Federated-edge-testbed.git platform
git -C platform checkout 7d09a0fc0c7cc8ffd5e19628085fe2d48fde81d2

pip install -r platform/requirements.txt
$env:PYTHONPATH = "$PWD;$PWD/platform"   # platform packages + this template's packages
```

Keep the submodule commit identical to the pin in `pyproject.toml`.

---

## Define plugins

Each plugin subclasses a platform ABC (or provides the functional client factory)
and self-registers at import time. The ABCs and registries are the frozen public
API — import them, never edit them:

```python
from extensions.interfaces import (
    ModelFactory, AggregatorFactory, ClientLogicFactory, DatasetLoader,
    DatasetBundle, RuntimeContext,
)
from registry import (
    register_model, register_aggregator, register_client_fn, register_dataset_loader,
    list_registered_components,
)
```

Shipped examples:

| Plugin | File | Registered name |
| --- | --- | --- |
| Small CNN (`ModelFactory` + `module:callable`) | [plugins/models/example_model.py](plugins/models/example_model.py) | `example_cnn` |
| FedAvg (`AggregatorFactory`) | [plugins/aggregators/example_aggregator.py](plugins/aggregators/example_aggregator.py) | `fedavg` |
| NumPyClient (functional + `ClientLogicFactory`) | [plugins/clients/example_client.py](plugins/clients/example_client.py) | `example_numpy_client` |
| MNIST (`DatasetLoader`) | [plugins/datasets/example_dataset.py](plugins/datasets/example_dataset.py) | `mnist` |

**`register.py` is the single bootstrap.** It imports every plugin submodule so
all `register_*` calls fire. Any experiment entrypoint must `import register`
first. Add one import line there per new plugin. Sanity-check what is registered:

```powershell
python register.py            # prints the registry contents as JSON
```

### How references resolve (two surfaces)

The platform discovers components by **string name** (`ComponentRef(name, kwargs)`)
and, for custom code, by the **`module:attribute`** string form:

- **Model** — referenced by name plus a `source`:
  `ComponentRef("example_cnn", {"source": "plugins.models.example_model:build_model"})`.
  The runtime passes `source` through as `--model-source` and invokes the
  callable inside the server/client subprocesses.
- **Client** — `client_fn=ComponentRef("default", {...})` uses the platform's
  built-in training loop. A custom client is referenced by its `module:callable`
  name, e.g. `ComponentRef("plugins.clients.example_client:make_client", {...})`;
  any name other than `"default"` is forwarded as `--client-fn-source`.
- **Dataset / Aggregator** — referenced by registered name (`"mnist"`, `"fedavg"`).

---

## Declare an experiment

[studies/example_experiment.py](studies/example_experiment.py) exposes
`make_spec() -> ExperimentSpec` wiring plugins by name, with explicit
`num_rounds`, `num_clients`, `seed`, and `server_address`:

```python
import register  # noqa: F401  -- registers all plugins
from experiments.api import ComponentRef, ExperimentSpec

def make_spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="example_mnist_fedavg",
        dataset=ComponentRef("mnist", {"data_dir": "data/mnist"}),
        model=ComponentRef("example_cnn", {"source": "plugins.models.example_model:build_model"}),
        aggregator=ComponentRef("fedavg"),
        client_fn=ComponentRef("default", {"local_epochs": 1, "batch_size": 32}),
        num_rounds=10, num_clients=4, seed=2025,
        server_address="127.0.0.1:8081",
    )
```

---

## Data model

How data is split across the federation (the platform enforces this; you just
supply the dataset):

- **Central, on the server:** the full dataset becomes `data/<name>/train.pt` +
  `data/<name>/test.pt`.
- **`test.pt` stays on the server** as the *centralized global test set*. After
  each round the server evaluates the aggregated model on it and logs
  `global_test_accuracy` to `server_rounds.csv` — that is your **global accuracy**.
  (Server resolves it from `--global-test-path`, default `data/<name>/test.pt`.)
- **`train.pt` is split into N per-client shards** `data/<name>/shards/client_<id>.pt`
  (one per client; IID by default, or Dirichlet for label skew).
- **Each client gets only its shard** and splits it **locally** into
  train/val/test; federated training then runs on those local splits.

So: one global test on the server for the global metric, disjoint training shards
on the clients with local train/val/test. In simulation everything lives in your
workspace `data/`; on the Pi path `deploy_student.sh` pushes `train.pt`/`test.pt`
to the server and each `client_<id>.pt` to the matching client Pi.

## Prepare data and shards

The runtime resolves a dataset by **name** and reads per-client shards from
`data/<name>/shards/client_<id>.pt`. Generate them with the platform's data
engine (sharding/partitioning is platform-core — you only supply a processor).
The "Start" flow does this for you; to run it manually:

```powershell
# Built-in datasets need no processor:
python -m utils.prepare_data --dataset mnist --num-clients 4 --seed 2025

# Custom data: write a prepare_dataset callable and reference it.
python -m utils.prepare_data --dataset mydata --num-clients 4 --seed 2025 `
    --processor data_processes.example_processor:prepare_dataset
# add --dirichlet-alpha 0.5 for non-IID shards.
```

The example processor is [data_processes/example_processor.py](data_processes/example_processor.py)
(contract: return `(train_x, train_y, test_x, test_y, meta)`). The `DatasetLoader`
plugin is the in-process / discover-by-name surface; the shards above are what the
server/client processes actually consume.

## Run

### Local simulation — the default "Start" (one command)

`scripts/start.sh` is the educational one-button path: it auto-shards your data
(from the spec) and then runs the server + N clients **as local processes** on the
current machine. No lock; many students can run at once (each gets a free port).

```bash
bash scripts/start.sh                                  # example spec
bash scripts/start.sh studies.my_experiment:make_spec  # your spec
```

`start.sh` and `run_on_testbed.sh` self-source `activate.sh`, which reads
`PLATFORM_DIR`/`VENV_ACTIVATE` from `testbed/testbed.env` to set up imports + venv.
For ad-hoc commands in your own shell, `source activate.sh` first — otherwise you
get `ModuleNotFoundError: No module named 'registry'`.

Lower-level equivalents, if you want to drive the steps yourself:

```powershell
python -m testbed.prepare  --experiment-source studies.example_experiment:make_spec  # shard
python -m testbed.run_sim  --experiment-source studies.example_experiment:make_spec  # run (free port)
# or the raw platform runner (uses spec.server_address as-is, no auto-shard):
python -m experiments.run_submitted --experiment-source studies.example_experiment:make_spec
```

Sharding derives from `spec.dataset.kwargs` (`processor`, `shard_mode`,
`dirichlet_alpha`, `source`) plus `num_clients`/`seed` — see
[testbed/prepare.py](testbed/prepare.py). The runtime resolves the dataset by name
and reads `data/<name>/shards/`.

> **Importability:** the platform spawns the server/client as subprocesses with
> cwd = the *platform* root, so your `plugins`/`studies` packages must be
> importable from anywhere — either `pip install -e .` the template, or put the
> workspace on `PYTHONPATH` with an **OS-correct absolute path** (on Windows that
> means a real `C:\...` path, not a Git-Bash `/c/...` path). The `start.sh` /
> `testbed.run_sim` / `testbed.prepare` wrappers set this for their child
> processes automatically; the raw `experiments.run_submitted` does not.

### The 5-Pi testbed (1 server + 4 client Pis, shared by students)

Students SSH into the **server Pi** and run their own experiments against the 4
**physical** client Pis. The clients are physical, so a real run is **exclusive**
— one experiment at a time, coordinated by a shared lock.

Your `ExperimentSpec` is the single source of truth: a thin bridge
([testbed/launch.py](testbed/launch.py)) derives the launch parameters from it and
delegates to the platform's distributed orchestrator (`controller.start_all`),
which SSHes into all 5 nodes and starts `server.server` / `client.client`.

**One-time setup** (per student, on the server Pi): clone this template into your
home, set up the env (Option B in Install), then
`cp testbed/testbed.env.example testbed/testbed.env` and edit the paths.
Admins provision the node inventory once in the platform's
`controller/config.json` (server + 4 clients: `ssh_ip`, `fl_ip`, `user`, `key`) —
that is the **single, shared source of truth** for the physical topology; the
template does not duplicate it. The annotated schema/reference is
[testbed/controller.config.example.json](testbed/controller.config.example.json)
(copy it to the platform path to bootstrap a fresh testbed). Run-level fields in
that config are overridden per run from your spec; only the infra fields are used.

**Run (one command, lock-guarded):**

```bash
# prepare shards first (see "Prepare data and shards"), then:
bash scripts/run_on_testbed.sh studies.example_experiment:make_spec
```

That wrapper, in order: acquires the exclusive lock (fails fast if busy, naming
the holder) → `scripts/deploy_student.sh` rsyncs your `plugins/`, `studies/`,
`data_processes/`, `register.py` **and** your shards into the shared platform tree
on all 5 Pis → launches via `controller.start_all` and blocks until the rounds
finish → releases the lock on exit (even on Ctrl-C/crash). Use `--dry-run` to
print the remote commands without executing. Check availability anytime with
`bash scripts/testbed_lock.sh status`.

**Why your code must be deployed:** the Pis resolve `--model-source
plugins.models...:build_model` **inside** each remote process, so your `plugins/`
must exist on every node — `deploy_student.sh` is what puts it there. The lock
makes writing into the shared tree safe (only your code is present during your
run).

**Two limits to know:**
- Network addresses come from the physical inventory (`fl_ip` + `server_port` in
  `controller/config.json`), **not** `spec.server_address` (that field is only for
  local simulation).
- `controller.start_all` forwards `--model-source` but **not**
  `--client-fn-source`. A custom functional `client_fn` works in local simulation
  but is **not** applied on the Pi path today — use `client_fn=default` for Pi
  runs, or extend the platform. The launcher warns when this would happen.

For simulation-style work (no physical Pis), many students can run concurrently on
the server with `submit_experiment` / `run_submitted` (each picks a distinct
`server_address` port) — no lock needed.

### Outputs

Every run writes `logs/<name>_<UTCtimestamp>/` containing:

- `manifest.json` — the full resolved spec (audit record);
- `process_server.log`, `process_client_<id>.log`, `server.log`;
- `server_rounds.csv` — per-round server metrics;
- `metrics/` — per-client metrics; `checkpoints/` — model checkpoints;
- `global_model.pt` — final aggregated model.

---

## Energy (CodeCarbon)

Per-client energy/emissions tracking is built into the platform. Turn it on from
your spec — one switch, works in both simulation and on the Pis:

```python
client_fn=ComponentRef("default", {"local_epochs": 1, "batch_size": 32,
                                    "enable_codecarbon": True})
```

Each client then writes `emissions.csv` under
`data/<dataset>/codecarbon/client_<id>/` (override with `"codecarbon_output_dir"`).
`testbed.run_sim` sets the `ENABLE_CODECARBON` env for the simulated clients;
`testbed.launch` passes the override to the Pi orchestrator. (Default off — it adds
per-round overhead.)

## Reproducibility

- **One pin.** The platform commit in `pyproject.toml` == `requirements.txt` ==
  the `platform/` submodule. Bumping the platform is a one-line change.
- **Pinned baseline.** `flwr>=1.22,<1.23`, `torch==2.2.0`, `torchvision==0.17.0`,
  `numpy==1.26.4` — the platform's tested versions. Do not float them.
- **Seeds.** Every `ExperimentSpec` carries an explicit `seed`. For multi-seed
  studies, submit the same spec across a seed sweep:
  ```powershell
  foreach ($s in 1,2,3,4,5) {
      python -c "import register; from experiments.submission import submit_experiment; from studies.example_experiment import make_spec; import dataclasses; submit_experiment(dataclasses.replace(make_spec(), seed=$s))"
  }
  ```
  `manifest.json` captures the full resolved spec (including `seed`) for each run.
- **Lockfile.** Commit a resolved lockfile so installs are byte-reproducible:
  ```powershell
  pip freeze > requirements.lock      # or: uv lock
  ```

---

## Layout

```
fl-edge-template/
├── pyproject.toml          # exact platform pin (single source of version truth)
├── requirements.txt        # mirror of the pin for plain pip
├── register.py             # imports every plugin so registrations fire
├── plugins/
│   ├── models/example_model.py        # ModelFactory + module:callable
│   ├── aggregators/example_aggregator.py
│   ├── clients/example_client.py      # functional client factory + ABC
│   └── datasets/example_dataset.py    # DatasetLoader -> DatasetBundle
├── GETTING_STARTED.md                 # student quickstart (one-button workflow)
├── docs/ADMIN_SETUP.md                # operator provisioning (soft isolation)
├── activate.sh                        # source for PYTHONPATH→platform + venv (from testbed.env)
├── studies/example_experiment.py      # make_spec() -> ExperimentSpec
├── data_processes/example_processor.py # prepare_dataset() for utils.prepare_data
├── testbed/
│   ├── prepare.py                     # auto-shard from spec ("Start" step 1)
│   ├── run_sim.py                     # local simulation, free port ("Start" step 2)
│   ├── launch.py                      # ExperimentSpec -> controller.start_all bridge (Pi path)
│   ├── testbed.env.example            # per-student paths (copy to testbed.env)
│   └── controller.config.example.json # topology schema/reference (live one is admin-maintained)
├── scripts/
│   ├── start.sh                       # SIMULATION start: shard + run locally (default)
│   ├── run_on_testbed.sh              # PI run: lock -> shard -> deploy -> launch
│   ├── deploy_student.sh              # rsync your code + shards to all 5 Pis
│   └── testbed_lock.sh                # exclusive-access lock helper
├── configs/example.json               # optional legacy-runner config
├── data/                              # git-ignored shards
└── logs/                              # git-ignored run outputs
```

### Optional: legacy JSON runners

The platform also accepts JSON specs via `legacy_config_to_spec`. See
[configs/example.json](configs/example.json):

```powershell
python -m experiments.run_experiment --config configs/example.json
```
