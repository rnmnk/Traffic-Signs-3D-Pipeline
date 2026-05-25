# Traffic-Signs-3D-Pipeline

## 1. Expected input dataset layout

Training configs point `data.root` at a directory with:

```
<dataset_root>/
  metadata.json
  clusters/
    train/<candidate_id>.laz
    val/<candidate_id>.laz
    test/<candidate_id>.laz
```

Each candidate cluster is one `.laz` file under `clusters/<split>/`. The `metadata.json` is a JSON array describing every candidate.

### Required fields per record


| Key                 | Type            | Notes                                |
| ------------------- | --------------- | ------------------------------------ |
| `candidate_id`      | str             | matches `<split>/<candidate_id>.laz` |
| `split`             | str             | one of `train`, `val`, `test`        |
| `label`             | int {0, 1}      | 0 = FP, 1 = TP                       |
| `bbox3d_center_xyz` | list[float] (3) | bbox center in your CRS (meters)     |
| `bbox3d_size_xyz`   | list[float] (3) | `(width, depth, height)` in meters   |


Example:

```json
[
  {
    "candidate_id": "abc_0001_c00",
    "split": "train",
    "label": 1,
    "bbox3d_center_xyz": [26490742.86, 6759662.25, 109.58],
    "bbox3d_size_xyz":   [0.6, 0.6, 0.75]
  }
]
```

The output of `ts3d eval`, `scored_metadata.json`, uses this same schema plus an extra `score` column.

### Synthetic dataset for pipeline smoke-testing

Since the dataset used in the thesis work cannot be released, a synthetic dataset with the same on-disk layout and metadata schema can be generated with `scripts/generate_synthetic_dataset.py`:

```bash
uv run python scripts/generate_synthetic_dataset.py --out synthetic_dataset --total 10000 --seed 0
```


| Option    | Required | Default | Description                                                                                          |
| --------- | -------- | ------- | ---------------------------------------------------------------------------------------------------- |
| `--out`   | yes      | —       | Output dataset root; `metadata.json` + `clusters/<split>/*.laz` written.                             |
| `--total` | no       | `200`   | Total number of candidates; split 70/15/15 train/val/test with TP/FP balanced 1:1 within each split. |
| `--seed`  | no       | `0`     | RNG seed; controls all geometry, point counts, and ordering.                                         |


## 2. CLI commands

The CLI entrypoint is `ts3d`. Two commands are available: `train` and `eval`.

### `ts3d train`

Trains a model end-to-end on `train` + `val` splits, writing a run director under `cfg.output_dir/<experiment_name>/<run_id>/` (config copy, checkpoints, logs, `metrics.jsonl`).


| Option          | Required | Default | Description                         |
| --------------- | -------- | ------- | ----------------------------------- |
| `--config`/`-c` | yes      | —       | Path to experiment YAML.            |
| `--run-id`      | no       | auto    | Override the auto-generated run id. |
| `--log-level`   | no       | `INFO`  | structlog level.                    |


Example:

```bash
ts3d train --config configs/pointnext-b.yaml
```

**Note**: the model is threshold-free. Training optimizes `BCEWithLogitsLoss` on raw logits, and the best checkpoint is selected by **AUPRC** on the val split . The decision threshold that turns sigmoid scores into binary predictions is purely an eval-time rule: it is picked on val as the smallest threshold whose recall is at least `cfg.eval.target_recall` and then applied to whatever split is scored.

### `ts3d eval`

Scores a split with a trained checkpoint and emits a metrics report, three plots, and the scored metadata. The decision threshold is **selected on the** `val` **split** (with `fixed_recall` at `cfg.eval.target_recall`).


| Option          | Required | Default    | Description                                 |
| --------------- | -------- | ---------- | ------------------------------------------- |
| `--config`/`-c` | yes      | —          | Same experiment YAML used at train time.    |
| `--ckpt`        | yes      | —          | Path to a `.ckpt` produced by `ts3d train`. |
| `--split`       | no       | `test`     | Split to score: `train` / `val` / `test`.   |
| `--out-dir`     | no       | `eval_out` | Directory for outputs (created if missing). |
| `--log-level`   | no       | `INFO`     | structlog level.                            |


Example:

```bash
ts3d eval --config configs/pointnext-b.yaml \
          --ckpt runs/pointnext_baseline/<run-id>/checkpoints/best.ckpt \
          --split test \
          --out-dir eval_out
```

## 3. Logging & experiment tracking

- **MLflow** for experiment tracking (params, metrics, artifacts, models). Tracking is always enabled; configure only the URI / experiment / run name under `tracking:` in the YAML. Recommended local setup:
  ```bash
  mlflow server \
    --backend-store-uri sqlite:///mlruns.db \
    --default-artifact-root ./mlruns \
    --host 127.0.0.1 --port 5000
  ```
  Then point `tracking.tracking_uri` at `http://127.0.0.1:5000`.
- **structlog** for process logs. Every run writes `runs/<experiment_name>/<run_id>/logs/train.log` plus stdout.

## 4. Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                         # core deps (incl. torch)
uv sync --extra dev             # + pytest, ruff, mypy
```

Python 3.10–3.13 supported.