# Maritime studies (student walkthrough)

Two federated use cases built **as a student would** — only research files
(`data_processes/`, `plugins/models/`, `studies/`), nothing in the testbed
plumbing. They show the same template handling a **classification** and a
**regression** task.

| Study | Task | Data | Target |
| --- | --- | --- | --- |
| AIS | classification | `data/AIS Dataset/ais_data.csv` (358k rows) | ship type (17 classes) |
| CBM | regression | UCI naval-propulsion `data.txt` (12k rows) | 2 decay coefficients |

## The pieces (what a student writes)

```
data_processes/ais_processor.py     # raw csv  -> tensors  (classification)
data_processes/cbm_processor.py     # raw txt  -> tensors  (regression)
plugins/models/tabular_mlp.py       # one MLP, branches on task
studies/ais_experiment.py           # make_spec() for AIS
studies/cbm_experiment.py           # make_spec() for CBM
register.py                         # imports tabular_mlp so it registers
```

Each spec wires it together — note the `dataset` kwargs carry the **processor**, so
"start" auto-shards from raw data:

```python
dataset=ComponentRef("cbm", {"processor": "data_processes.cbm_processor:prepare_dataset",
                             "data_dir": "data/cbm", "shard_mode": "iid"})
model=ComponentRef("tabular_mlp", {"source": "plugins.models.tabular_mlp:build_model"})
```

The model reads `task` / `input_dim` / `num_classes` / `output_dim` from the dataset
metadata your processor emits, and outputs log-probabilities (classification) or raw
values (regression) accordingly.

## Run

Simulation (your default — auto-shards the raw data, streams per-round progress):

```bash
bash scripts/start.sh studies.cbm_experiment:make_spec     # regression
bash scripts/start.sh studies.ais_experiment:make_spec     # classification
```

Real Pis (when the lock is free):

```bash
bash scripts/run_on_testbed.sh studies.cbm_experiment:make_spec --dry-run
bash scripts/run_on_testbed.sh studies.cbm_experiment:make_spec
```

## Results

- **CBM (regression):** watch `global_test_loss` (MSE) fall in `server_rounds.csv` /
  `server.log`; lower is better. No accuracy (it's regression).
- **AIS (classification):** watch `global_test_accuracy` rise.

## Make it your own (ideas)

- AIS: add `navigationalstatus` (one-hot) to `_FEATURES`, or predict a different
  label via `--label-col`; try non-IID shards (`shard_mode="dirichlet"`,
  `dirichlet_alpha=0.3`) to simulate ports with skewed traffic.
- CBM: try deeper/wider MLP in `tabular_mlp.py`, or more `local_epochs`.
- Both: multi-seed runs for confidence intervals; flip `enable_codecarbon` to
  measure energy per client.

See [../STUDENT_GUIDE.md](../STUDENT_GUIDE.md) for the general workflow and the
edit-vs-plumbing boundary.
