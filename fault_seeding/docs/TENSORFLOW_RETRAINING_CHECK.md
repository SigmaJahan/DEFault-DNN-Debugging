# TensorFlow Retraining Check

The released dataset can be inspected and validated without TensorFlow.
TensorFlow is required only to re-run mutation and model-training replication.

## Environment

From the repository root:

```bash
python3 -m venv .venv-tf
source .venv-tf/bin/activate
pip install -U pip
pip install -r requirements.txt
```

The tested environment for this release is Python 3.11 with
`tensorflow==2.15.1`, as pinned in `requirements.txt`.

## Smoke Test

Run a small end-to-end mutation/training check:

```bash
python3 scripts/tensorflow_smoke_test.py
```

The script:

- imports TensorFlow and prints the installed version,
- applies the `HBS` mutation to `data/seed_programs/fnn/FNN_31556268_correct.py`,
- trains the mutated program,
- writes a CSV under `output/tf_smoke/`, and
- validates that the CSV matches the 31-column Deep4ge schema.

Expected result: `PASS` from the underlying `scripts/demo_replication.py`
validation step. In the release audit, this command completed a full 50-epoch
run with `HBS` on `FNN_31556268_correct.py` and produced a 31-column, 50-row
CSV.

## Direct Command

The smoke test is equivalent to:

```bash
python3 scripts/demo_replication.py \
  --operator HBS \
  --seed data/seed_programs/fnn/FNN_31556268_correct.py \
  --output-dir output/tf_smoke
```

Use `--dry-run` only when checking AST-level mutation application without
training.
