import json
import os
import sys

import joblib
import numpy as np
from google.cloud import bigquery, storage
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score

PROJECT = os.environ["PROJECT"]
# Vertex sets AIP_MODEL_DIR to the GCS path where it expects the saved model.
MODEL_DIR = os.environ.get("AIP_MODEL_DIR", "/tmp/model")

# ORDER MATTERS. The serving container receives a bare list of numbers, so the
# scorer (Phase 8) must send them in exactly this order. Change one, change both.
FEATURES = ["temp_mean", "temp_max", "vibration_mean", "vibration_std",
            "rpm_mean", "pressure_mean", "voltage_mean"]

bq = bigquery.Client(project=PROJECT)
df = bq.query(f"SELECT * FROM `{PROJECT}.maintenance.v_training`").to_dataframe()

if len(df) < 500:
    sys.exit(f"only {len(df)} labeled windows — let the generator and the Dataflow "
             f"job run longer (you want 1,500+; see the Phase 2 table)")

# BigQuery hands back NULLABLE EXTENSION dtypes (Int64, Float64) that sklearn
# rejects with an opaque "could not convert" — cast to plain float64 first.
y = df.pop("label").astype("float64").to_numpy()
groups = df.pop("machine_id").to_numpy()
X = df[FEATURES].astype("float64").to_numpy()

if y.sum() < 20:
    sys.exit(f"only {int(y.sum())} positive windows out of {len(y)} — check the "
             f"Phase 4 balance query before training on this")

# ── SPLIT BY MACHINE, not with train_test_split() ────────────────────────
# Two windows two minutes apart from the SAME machine are near-duplicates. A
# random split puts one in train and its twin in test, so the model is scored
# on rows it has effectively already seen. Holding out WHOLE MACHINES measures
# the thing we actually care about: does this generalize to a machine the model
# has never met? (It is also the only split that survives an interviewer asking
# "how did you split your time series?")
uniq = np.unique(groups)
rng = np.random.default_rng(42)
rng.shuffle(uniq)
test_machines = set(uniq[: max(1, len(uniq) // 5)])          # hold out 20% of machines
is_test = np.array([g in test_machines for g in groups])

Xtr, ytr = X[~is_test], y[~is_test]
Xte, yte = X[is_test], y[is_test]
print(f"train: {len(ytr)} windows / {len(uniq) - len(test_machines)} machines")
print(f"test:  {len(yte)} windows / {len(test_machines)} machines (disjoint)")

if yte.sum() == 0:
    sys.exit("the held-out machines contain no failures — collect more data and re-run")

model = GradientBoostingRegressor(random_state=42)
model.fit(Xtr, ytr)

score = model.predict(Xte)                       # continuous risk score
auc = roc_auc_score(yte, score)
pr_auc = average_precision_score(yte, score)     # the right headline for imbalanced data
recall_50 = recall_score(yte, (score > 0.5).astype(int), zero_division=0)

# A defensible operating point: flag the riskiest windows at the base failure
# rate, rather than assuming 0.5 is meaningful for a REGRESSOR's output.
tuned = float(np.quantile(score, 1 - yte.mean()))
recall_tuned = recall_score(yte, (score >= tuned).astype(int), zero_division=0)

metrics = {
    "auc": float(auc),
    "pr_auc": float(pr_auc),
    "recall_at_0.5": float(recall_50),
    "recall_at_tuned": float(recall_tuned),
    "tuned_threshold": tuned,
    "positive_rate": float(yte.mean()),
    "n_train": int(len(ytr)),
    "n_test": int(len(yte)),
}
print("eval:", json.dumps(metrics))

# The prebuilt sklearn container loads a file named EXACTLY model.joblib.
joblib.dump(model, "/tmp/model.joblib")
with open("/tmp/metrics.json", "w") as f:
    json.dump(metrics, f)


def upload(local, name):
    bucket, prefix = MODEL_DIR[5:].split("/", 1)
    storage.Client().bucket(bucket).blob(
        f"{prefix.rstrip('/')}/{name}").upload_from_filename(local)


if MODEL_DIR.startswith("gs://"):
    upload("/tmp/model.joblib", "model.joblib")
    upload("/tmp/metrics.json", "metrics.json")   # the pipeline's deploy gate reads this
    print("uploaded model.joblib + metrics.json →", MODEL_DIR)
else:
    print("local run — model at /tmp/model.joblib, nothing uploaded")