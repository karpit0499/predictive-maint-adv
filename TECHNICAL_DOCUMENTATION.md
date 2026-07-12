# Technical Documentation — predictive-maint-adv

Streaming predictive maintenance with drift-triggered continuous training on Google Cloud.

This document describes *how the system works and why it is built this way*. It is the reference for someone operating, extending, or reviewing the system. For step-by-step construction from an empty project, see [BUILD_GUIDE.md](BUILD_GUIDE.md).

---

## Table of contents

1. [System overview](#1-system-overview)
2. [Component reference](#2-component-reference)
3. [Data model and contracts](#3-data-model-and-contracts)
4. [The labeling design](#4-the-labeling-design)
5. [The model](#5-the-model)
6. [Serving architecture](#6-serving-architecture)
7. [The continuous-training loop](#7-the-continuous-training-loop)
8. [Analytics layer](#8-analytics-layer)
9. [Identity and access model](#9-identity-and-access-model)
10. [Observability and alerting](#10-observability-and-alerting)
11. [CI/CD](#11-cicd)
12. [Operations runbook](#12-operations-runbook)
13. [Failure-mode catalogue](#13-failure-mode-catalogue)
14. [Cost model](#14-cost-model)
15. [Security posture](#15-security-posture)
16. [Known limitations](#16-known-limitations)

---

## 1. System overview

The system answers one question, continuously, for every machine in a fleet:

> **Will this machine fail within the next H minutes?**

It does so as a chain of independently-observable stages:

| Stage | Mechanism | Boundary crossed |
|---|---|---|
| Ingest | Generator → Pub/Sub topic | Producer decoupled from consumer |
| Transform | Dataflow (Beam), fixed windows per machine | Unbounded stream → bounded feature rows |
| Persist | BigQuery: raw + windowed features + failure events | Stream → warehouse |
| Label | Scheduled SQL, from the ground-truth failure log | Features → supervised training set |
| Train | Vertex AI custom container job | Training set → versioned model artifact |
| Gate | KFP v2 pipeline, `dsl.If` on measured AUC | Candidate model → deployed model |
| Serve | Vertex online endpoint + Cloud Run scorer; BQML in batch | Model → predictions |
| Monitor | SQL drift check + log-based alerting | Predictions → human / retrain trigger |
| Close the loop | Cloud Run job resubmits the pipeline | Data change → new model |

The design principle running through all of it: **each object has exactly one owner.** Terraform owns the substrate. SQL owns the derived tables and the BQML model. The pipeline owns the endpoint. dbt owns the marts. Where two systems could both claim an object, one is chosen and the other stays out — otherwise they fight over its schema on every apply.

### Key architectural properties

- **The stream never blocks on the model.** Telemetry lands in BigQuery whether or not any model exists.
- **Two independent scoring paths.** BQML (batch, in-warehouse, free, always available) and the Vertex endpoint (real-time, low-latency, billed). Both write into the same `predictions` table, tagged with `model_version`. The dashboard therefore doesn't go dark when the expensive endpoint is undeployed — a genuine resilience property, not a workaround.
- **Provenance on every prediction.** Every row records which model produced it, so prediction distributions can be compared across model versions after a retrain.
- **The label is never derived from a model input.** See §4.

---

## 2. Component reference

### 2.1 Generator (`generator/generate.py`)

A synthetic fleet of N machines, each carrying a hidden `health` value in `[0, 1]` that decays stochastically. Sensor readings are functions of `wear = 1 - health`:

| Sensor | Relationship to wear | Role |
|---|---|---|
| `temperature` | rises linearly with wear | strong signal |
| `vibration` | rises with **wear²** — spikes late | strong signal, non-linear |
| `rpm` | falls linearly with wear | signal |
| `pressure` | falls linearly with wear | weak signal |
| `voltage` | pure Gaussian noise | **deliberate distractor** |

`voltage` exists so that feature-importance output has something to be *wrong about*. A model that ranks it highly is telling you the labels are broken.

Two outputs, and the separation matters:

1. **Telemetry** → published to the Pub/Sub topic. This is what the model sees.
2. **Failure events** → written directly to BigQuery when a machine's health crosses its failure threshold. This is the **system of record**, the analogue of a plant's maintenance log. The model never sees it; the *labeler* uses nothing else.

The generator is an infinite loop by design — a factory floor does not "finish."

#### Parameter sensitivity (this is load-bearing)

The relationship between machine lifetime and window length is the single most fragile number in the project.

Dataflow aggregates into fixed **2-minute** windows. For a window to *see* degradation, a machine's lifetime must be much longer than one window. At the shipped parameters a machine lives roughly **43 minutes ≈ 21 windows** — a clean, visible ramp.

The failure mode is counter-intuitive. "Speeding things up" by lowering the publish interval and raising the decay rate compresses machine lifetimes *in wall-clock time* while the window stays at 2 minutes. Push it far enough and a machine is born, degrades, dies, and is replaced dozens of times *inside a single window*. Every window then averages dozens of complete lifecycles and comes out statistically identical to every other window. The features become pure noise, the model learns nothing, and it looks like the *model* is broken.

> **Rule: to collect data faster, add machines — never subtract sleep.**

At the shipped settings with 60 machines:

| Generator runtime | Failures | Labeled windows | Positive rate |
|---|---|---|---|
| 1 hour | ~63 | ~1,800 | ~17% |
| 2 hours | ~153 | ~3,600 | ~21% |

A drift-injection parameter (`VIB_OFFSET`) adds a constant bias to every vibration reading — a simulated sensor miscalibration, used to exercise the CT loop (§7).

### 2.2 Streaming feature pipeline (`dataflow/stream_features.py`)

Apache Beam, two branches off one Pub/Sub read:

- **Branch 1:** raw readings straight through to `telemetry_raw` (append, no windowing). This is the audit trail and the drift-analysis substrate.
- **Branch 2:** key by `machine_id` → `FixedWindows(120s)` → `GroupByKey` → aggregate → `features_windowed`.

Aggregates emitted per machine per window: mean and max temperature, mean and population-stdev vibration, mean rpm, mean pressure, mean voltage, and `reading_count`.

Three design decisions in this file, each of which fails silently if reversed:

1. **The CLI flag is `--project_id`, not `--project`.** Beam itself owns `--project`. If application argparse consumes it, Beam never receives it and the launch dies with `Missing required option: project`.
2. **`save_main_session=True`.** Beam pickles the user's functions and ships them to the workers *without* the module's imports. Without this flag the pipeline runs perfectly with the local DirectRunner and fails **only on Dataflow** with `NameError: name 'json' is not defined`.
3. **It reads a declared *subscription*, not a *topic*.** Reading a topic makes Dataflow create its own hidden subscription, which requires broad Pub/Sub admin rights on the worker service account. Reading a Terraform-declared subscription keeps it least-privilege.

**There is no label column in `features_windowed`.** A streaming job cannot know the future. Adding a permanently-`NULL` label column to a streaming table is dead weight every downstream query then has to `EXCEPT` out. The label is computed later, in BigQuery, into a *separate* table.

### 2.3 Labeler (`sql/label.sql`)

Rebuilds `features_labeled` from `features_windowed` ⨝ `failure_events`. Runs on a BigQuery schedule and is also re-run by the retrainer before every retrain. Detailed in §4.

### 2.4 BQML baseline

A boosted-tree classifier trained in one `CREATE MODEL` statement over the training view, with automatic class weights. It serves three purposes:

- a defensible **baseline** the custom model must beat;
- a **feature-importance sanity check** (temperature and vibration should dominate; voltage should sit near zero);
- a **free, always-on batch scoring path** that survives the endpoint being undeployed.

One mandatory detail: BQML treats *every* non-label column as a feature, so `machine_id` must be explicitly excluded. Leave it in and the model memorizes which specific machines fail rather than learning what failure looks like — and it is then useless on any machine it has not seen.

### 2.5 Custom training (`training/train.py`)

Runs in a container on Vertex AI. Reads the training view, splits by machine, fits a gradient-boosting regressor, computes metrics, and uploads two artifacts to the Vertex-provided model directory:

- `model.joblib` — the name the prebuilt serving container looks for, exactly;
- `metrics.json` — read by the pipeline's evaluate step, which is what makes the deploy gate *real* rather than decorative.

Two hard preconditions are enforced with early exits: a minimum number of labeled windows, and a minimum number of positives. Failing loudly on thin data is much cheaper than shipping a model trained on it.

### 2.6 Pipeline (`pipeline/maint_pipeline.py`)

KFP v2, three components:

```
train_and_register  →  evaluate_model  →  [ dsl.If: auc ≥ min_auc ]  →  deploy_model
```

- `train_and_register` submits the custom training job and returns the model resource name.
- `evaluate_model` downloads `metrics.json` from the registered model's artifact URI and returns the AUC.
- `deploy_model` finds-or-creates the endpoint, records what is *already* deployed on it, deploys the new model at 100% traffic, then **undeploys the superseded models**.

That last step is not housekeeping — it is a cost control. `traffic_percentage=100` routes 0% of traffic to the old models but **does not undeploy them**. Each keeps its own node, and each node keeps billing. After five retrains you would be paying for five idle machines serving nobody.

The gate is deliberately loose (a healthy run scores well clear of it). Its job is to catch *catastrophe* — broken labels, empty data, a corrupted feature table — not to micro-tune.

### 2.7 Scorer (`scorer/main.py`)

Cloud Run service, authenticated (no public access). `POST /score` with a feature payload:

1. Builds the instance vector **in exactly the training feature order**. The serving container receives a bare list of numbers with no column names. If the order drifts between `train.py` and `main.py`, the model silently reads temperature as vibration and *nothing errors*.
2. Calls the endpoint, guarding two states: no endpoint at all, and an endpoint that exists but has no model deployed on it (returns `503`, not `500` — say what's actually wrong).
3. **Clamps** the raw prediction into `[0, 1]`. The model is a regressor fit to a 0/1 target; its output overshoots slightly at both ends. Clamp before calling it a probability.
4. Emits `HIGH_RISK machine=… prob=…` on stdout when the score crosses the threshold — the exact string the alerting metric counts.
5. Inserts the prediction into `predictions` with the resolved `model_version`.

The endpoint handle is cached in module state, but **only after both guards pass**. Caching earlier is a live trap: the pipeline creates the endpoint minutes before it finishes deploying a model onto it. A request in that window would cache the empty endpoint with a null version, and every subsequent call on that warm instance would short-circuit the guard and write predictions with a `NULL` model_version — forever, without ever erroring. The provenance column would simply, quietly, be lost.

### 2.8 Retrainer (`retrainer/main.py`)

Cloud Run **job** (runs to completion and exits — the right shape for a periodic check, unlike a *service* which sits waiting for requests), triggered daily by Cloud Scheduler. See §7.

---

## 3. Data model and contracts

All tables live in one BigQuery dataset in the `EU` multi-region.

| Table | Written by | Partitioned on | Contract |
|---|---|---|---|
| `telemetry_raw` | Dataflow (branch 1) | `event_time` | Append-only. One row per sensor reading. Audit trail. |
| `features_windowed` | Dataflow (branch 2) | `window_end` | Append-only. One row per machine per 2-min window. **No label column.** |
| `failure_events` | Generator | — | Append-only. **Ground truth.** One row per real failure. Independent of every model input. |
| `features_labeled` | `label.sql` | — | **Replaced** on every run (`CREATE OR REPLACE`). Snapshot. Lags the live table by one full horizon. |
| `v_training` | `label.sql` (view) | — | Feature contract for training. Carries `machine_id` for *splitting only*, never as a feature. |
| `predictions` | BQML batch query **and** Cloud Run scorer | `scored_at` | Append-only. `model_version` records provenance. Two writers, one schema. |
| `drift_log` | Retrainer | — | Append-only. One row per drift check, whether or not it retrained. |

### Feature contract

Seven features, in a fixed order that is duplicated in exactly two files (`training/train.py` and `scorer/main.py`):

```
temp_mean, temp_max, vibration_mean, vibration_std,
rpm_mean, pressure_mean, voltage_mean
```

**Change one, change both.** There is no schema enforcement across this boundary — the prebuilt serving container accepts a positional array. This is the single most dangerous coupling in the system, and it is documented at both ends of it.

### Two columns present in the feature table but *excluded* from the model

| Column | Why it's excluded |
|---|---|
| `machine_id` | An identifier, not a feature. Including it lets the model memorize which machines fail instead of learning what failure looks like. It is carried into the training view **only** so training can split by machine, and it is popped before fitting. |
| `reading_count` | It encodes **the ingestion rate of the pipeline**, not the health of a machine — it is the same constant value in every row, because it counts `window_length ÷ publish_interval`. Training on it introduces a training-serving skew that materializes the first time a window lands short. It stays in the table as a data-quality signal ("was this window complete?") and out of the model. |

---

## 4. The labeling design

This is the part that is most often got wrong, so it is worth being explicit.

### The circular-label trap

The obvious shortcut is to *define* failure as "vibration crossed some threshold" and label the windows preceding it. **This is invalid**, for three compounding reasons:

1. `vibration_mean` is a **model input**. Defining the target by thresholding an input makes the target a near-deterministic function of that input. The model then learns "is vibration close to the threshold?" — a tautology, not a prediction.
2. The chosen threshold almost never corresponds to when the machine *actually* died. In this simulator, a vibration level that looks alarming still corresponds to a machine with ~20% health remaining. It hasn't failed. It's just sick.
3. It is visible to any reviewer who reads the SQL.

### What is done instead

Labels come from `failure_events` — a table written by the machine's own death, independent of every model input, and exactly what a real maintenance system provides.

```
will_fail = EXISTS( a failure for this machine
                    strictly AFTER this window's end
                    and within H minutes of it )
```

### Censoring

A window whose horizon has not yet fully elapsed **cannot be known to be a negative** — the future hasn't happened. Labeling it `0` is not conservative; it is *wrong*, and it is wrong in the most damaging possible direction, because the most recent windows are also the most degraded ones. Training on them teaches the model that badly-degraded machines are safe.

So the labeler drops them:

```sql
WHERE window_end <= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL horizon_minutes MINUTE)
```

**The observable consequence:** `features_labeled` always lags `features_windowed` by at least one full horizon. If the lag is near zero, the censoring filter did not apply, and the freshest, sickest windows are all sitting in the training set mislabeled `0`. This is a checkable invariant, and it is checked.

### The horizon

The horizon is a business decision, not a hyperparameter: in a real plant it is however long it takes to schedule an intervention (typically 24–72 hours). Here machine lifecycles are compressed to tens of minutes, so the horizon is scaled proportionately. It is declared **once**, at the top of `label.sql`, and nowhere else.

### Label validation

Four checks, run every time the labels are rebuilt. "The query ran" is not one of them.

| Check | Passing looks like | Failing means |
|---|---|---|
| **Freshness + censoring** | lag ≥ one horizon | the censoring filter didn't apply |
| **Balance** | 15–25% positives | 0% → no failures logged; <5% → not enough data yet; >60% → decay rate too high |
| **Separability** | `label=1` rows show visibly higher temp and vibration, lower rpm; `voltage` near-identical across classes | if the classes look the same on every column, the labels are noise, and no amount of hyperparameter tuning will move the AUC off 0.5. If **voltage** separates the classes, something is badly wrong — it's pure noise by construction. |
| **The schedule fired** | one successful run in history | a saved-but-never-executed schedule is the most common false pass |

---

## 5. The model

### Why a regressor on a binary target

This is the one place the design bends to an infrastructure constraint, and it is documented rather than hidden.

The model is deployed behind Google's **prebuilt scikit-learn prediction container**, which calls exactly one method: `predict()`. For a classifier, `predict()` returns hard 0/1 labels — but the product needs a *probability* ("this machine is 87% likely to fail"). `predict_proba()` is **not exposed** by the prebuilt container.

The pragmatic resolution: fit a **`GradientBoostingRegressor` on the 0/1 label**, so `predict()` itself returns a continuous risk score. AUC and PR-AUC are rank-based, so evaluation is entirely unaffected. The score can overshoot slightly outside `[0, 1]`, so the scorer clamps it.

The correct production alternative is a **Custom Prediction Routine** or a bespoke serving container. This is named in the "what I'd do next" section of the README because that is where it belongs.

### The split

**By machine, never `train_test_split()`.** Two windows two minutes apart from the same machine are near-duplicates. A random split puts one in train and its twin in test, so the model is scored on rows it has effectively already seen — and the resulting metric is a lie in the flattering direction.

Whole machines are held out (20% of them). The metric then answers the question that actually matters: *does this generalize to a machine the model has never met?* Any random split over a time series of the same entities measures memorisation instead.

A guard exits if the held-out machines happen to contain no failures.

### Metrics reported

| Metric | Why |
|---|---|
| `auc` | Threshold-free ranking quality. The pipeline's deploy gate reads this one. |
| `pr_auc` | **The honest headline** on an imbalanced problem. AUC flatters imbalance; average precision doesn't. |
| `recall_at_0.5` | Reported for completeness — and it will be low. |
| `recall_at_tuned` | The real operating point: flag the riskiest windows at the base failure rate. |
| `tuned_threshold`, `positive_rate`, `n_train`, `n_test` | Context, without which the numbers above are unreadable. |

`recall_at_0.5` being lower than `recall_at_tuned` is **expected, not a bug**. A regressor fit to a 0/1 target emits a risk score, not a calibrated probability; 0.5 has no special meaning to it. Reporting both, and explaining the difference, is worth more than another 0.01 of AUC.

### Expected values, and what deviations mean

| Observed | Diagnosis |
|---|---|
| AUC ≈ 0.93–0.96 | healthy |
| AUC ≈ 0.5 | the labels are broken — go back to the balance and separability checks |
| AUC ≈ 0.999 | something is leaking — check that `reading_count` and `machine_id` were dropped |

### A counter-intuitive property of the predictions

**The riskiest window is not the most-degraded window.** A machine at ~95% wear scores *lower* than one at ~90%. This is correct. The label is "will this machine fail within the next H minutes," and a machine that extreme has already died and been replaced *inside that window* — so the next H minutes contain no failure. Risk peaks just *before* the end, then falls.

That is genuinely what predictive maintenance looks like. It is also why test payloads must be drawn from the **real feature ranges**: gradient-boosted trees do not extrapolate. Feed one a machine that cannot physically exist and it returns a meaningless value from an arbitrary leaf — not an error.

---

## 6. Serving architecture

Two paths, on purpose:

| | Batch | Real-time |
|---|---|---|
| **Engine** | BigQuery ML, in-warehouse | Vertex AI online endpoint |
| **Trigger** | scheduled query, every 15 min | HTTP `POST` to the Cloud Run scorer |
| **Cost** | effectively free | one node, billed continuously while deployed |
| **Availability** | always | only while a model is deployed |
| **Latency** | minutes | milliseconds |
| **Writes to** | `predictions`, tagged with the BQML model name | `predictions`, tagged with the Vertex model version |

Both write to the same table with the same schema. Consequences worth stating plainly:

- The fleet dashboard **does not go dark** when the endpoint is undeployed for cost reasons. That's a real resilience story, not an excuse.
- Because every row carries `model_version`, prediction *distributions* can be compared across model versions after a retrain.
- The dashboard has data from the moment the BQML baseline exists, rather than waiting on the handful of rows a manual test request produces.

---

## 7. The continuous-training loop

CI/CD deploys on **code** change. CT retrains on **data** change. This is the latter.

### The drift metric

An **effect size** — a standardized mean shift (Cohen's *d*):

```
z_shift = | recent_mean − baseline_mean | / baseline_stddev
```

Read as *"how many baseline standard deviations has the mean moved?"* Conventionally: ~0.2 small, ~0.5 medium, ~0.8 large. The trigger fires at **1.0** — a full baseline standard deviation, which is unambiguous and does not trip on noise.

Two design details that each prevent a silent, permanent no-op:

- **It reads the LIVE feature table, not the labeled snapshot.** `features_labeled` only changes when something rebuilds it. Pointing a drift monitor at a snapshot means it faithfully reports "no drift" while the world moves, because it is looking at a photograph.
- **The baseline is the *earliest* window of data, anchored to the table's own minimum timestamp** — not "older than one day." For the first 24 hours of the project's life, an "older than one day" window is *empty*, so the mean is `NULL`, so `z_shift` is `NULL`, so the check silently does nothing. Forever, and without ever erroring.

### The loop

```
Cloud Scheduler (daily)
        │
        ▼
Cloud Run job: retrainer
        │
        ├── run drift.sql → z_shift
        │
        ├── if z_shift ≤ threshold → log it, exit 0
        │
        └── if z_shift > threshold:
                ├── RE-RUN label.sql          ← the line that makes the loop actually close
                ├── submit the KFP pipeline
                │       └── train → evaluate → IF auc ≥ gate → deploy
                └── log { z_shift, retrained: true }
```

**The label rebuild is not optional.** The pipeline trains on the training view, which reads `features_labeled` — a *snapshot*. Skip the rebuild and the "retrain" dutifully re-fits the same model on the same stale rows and changes precisely nothing: a continuous-training loop that is, silently, a no-op. One line is the difference.

And because the pipeline carries the `dsl.If` gate, a retrain that *doesn't* clear the bar simply won't ship. The loop is self-correcting in both directions.

### Exercising it honestly

The intuitive way to force drift — restart the generator with a much higher decay rate so machines wear out faster — **does not work, and understanding why is the interesting part.** Raising the decay makes each machine cycle through its life faster, but the *fleet* still contains a steady mix of healthy, middling, and dying machines at any instant, in roughly the same proportions. The distribution barely moves: a 25× increase in decay rate shifts the fleet-wide mean vibration by z ≈ 0.01. It can never trip the threshold.

What *does* work is injecting a **sensor fault** — a constant bias added to every vibration reading. That is exactly what a miscalibrated or degrading sensor array does in the real world, and it is one of the most common causes of genuine production drift. It produces z ≈ 1.6, comfortably over the threshold.

> **Precondition:** the Dataflow job must be running. A generator with no consumer produces nothing — messages pile up in the subscription, the feature table gains no rows, the drift check reads a stale window, and `z_shift` comes back ≈ 0. The drift query was correct all along.

---

## 8. Analytics layer

dbt, authenticating through Application Default Credentials (no keys).

| Model | Type | Purpose |
|---|---|---|
| `stg_predictions` | staging | Cleans `predictions`, buckets `failure_prob` into `high` / `medium` / `low`. |
| `mart_fleet_health` | mart | Latest score per machine (`ROW_NUMBER() … = 1`). One row per machine. |
| `mart_model_runs` | mart | Daily max `z_shift` and whether a retrain fired that day. |

Tests: `not_null` on identifiers, `accepted_values` on the risk bucket, and `unique` on `mart_fleet_health.machine_id` — which asserts the "exactly one current risk row per machine" invariant. That last test can legitimately fail: both scoring paths stamp `scored_at` independently, so a machine scored by batch *and* endpoint in the same instant can tie. The fix is a deterministic tiebreak in the `ORDER BY`, not deleting the test.

### Dashboard

Looker Studio, sourced from the dbt marts. Four elements:

- **Scorecard** — count of machines currently in the `high` bucket, conditionally formatted. A calm fleet should look calm.
- **Fleet table** — machines worst-first, with the risk bucket colouring the row.
- **Time series** — prediction volume broken down by *scoring path*. The two initial series are batch and endpoint, not two retrained versions; a genuine **version flip** only appears once a drift run mints a new model version. Both are worth saying out loud. Don't call it a retrain until it is one.
- **Combo chart** — daily max `z_shift` as a line with a dashed reference line at the retrain threshold, and retrain events as bars on a secondary axis. Read left to right: the line climbs → crosses the threshold → a bar fires that same day → and in the chart next door, the model version flips. Drift → retrain → deploy, in two glances.

> If every retrain bar is zero, **the chart is correct** and the fleet simply hasn't drifted. Do not "fix" it by lowering the threshold.

---

## 9. Identity and access model

Four service accounts, each scoped to what it actually does. No keys are downloaded anywhere in this project.

| Service account | Runs | Roles |
|---|---|---|
| **Dataflow worker** | the Beam job | `dataflow.worker`, `pubsub.subscriber`, `pubsub.viewer`, `bigquery.dataEditor`, `storage.objectAdmin` |
| **Pipeline runner** | training jobs, the KFP pipeline, the retrainer | `aiplatform.user`, `bigquery.dataEditor`, `bigquery.jobUser`, `bigquery.readSessionUser`, `storage.objectAdmin`, `artifactregistry.reader`, `logging.logWriter` |
| **Scorer** | the Cloud Run service | `aiplatform.user`, `bigquery.dataEditor`, `logging.logWriter` |
| **GitHub deployer** | CI/CD, via WIF | `run.admin`, `iam.serviceAccountUser`, `cloudbuild.builds.editor`, `artifactregistry.writer`, `storage.admin` |

### Four non-obvious grants, each of which fails in a way that doesn't look like IAM

**`bigquery.jobUser` is separate from `bigquery.dataEditor`.** `dataEditor` lets a principal read and write table *data*; *running a query* additionally needs `jobUser`. Any SA that executes SQL needs both. Dataflow, notably, only ever streams rows in — it never runs a query — so it doesn't need `jobUser`.

**`bigquery.readSessionUser` is a third, entirely separate permission.** When the BigQuery Storage client library is installed, pandas' `.to_dataframe()` stops pulling rows through the ordinary query API and quietly switches to the **BigQuery Storage Read API** — which is governed by `bigquery.readsessions.create`. So a service account can hold `dataEditor` **and** `jobUser`, execute the query without complaint, and then die *on the way back* while fetching the rows. The training script calls `.to_dataframe()`, so the pipeline SA needs it.

**`artifactregistry.reader` on the pipeline SA** is what lets Vertex AI *pull* the training image. Without it, the training job dies on an image pull — which looks nothing like a permissions error until you read the log closely.

**`logging.logWriter` is the one that hides all the others, and it is the most expensive omission in the entire project.** A Vertex training worker and a Cloud Run container both ship their stdout to Cloud Logging **as the service account they run under**. Without `logWriter`, that write is denied and the output is simply discarded. A crashing container then produces:

```
The replica workerpool0-0 exited with a non-zero status of 1
```

…and **nothing else**. No traceback. No `print()`. No clue. Every other failure in this list — the missing `jobUser`, the failed image pull, a plain Python bug — arrives looking exactly like this one. **Grant `logWriter` first; it is what makes every other failure legible.**

It is also load-bearing for alerting: the high-risk alert is a log-based metric on a line the scorer prints, and a log that never arrives can never match.

*(The Dataflow worker is the exception — the `dataflow.worker` role already includes log-writing, so Dataflow workers can write logs out of the box.)*

### Diagnosis order for an opaque exit-1

1. **Is the log empty?** → missing `logWriter`. Fix that before anything else.
2. **Does the script run clean locally?** → if yes, it's environmental, not code.
3. **Is the image `amd64`?** → an `arm64` image built on an Apple Silicon Mac dies *before Python starts*, and also produces no output. This is why images are built server-side with Cloud Build, never with a local `docker build`.

---

## 10. Observability and alerting

### The high-risk alert

The scorer prints a plain-text line on stdout when a score crosses the threshold. A **log-based counter metric** matches that string; a **Monitoring alert policy** fires on it.

Three things have to be simultaneously true for that to work, and each fails silently:

1. **The runtime service account must hold `logging.logWriter`** (§9). Without it the line is never recorded — the request still returns a score, the BigQuery row still lands, and the log is simply empty.
2. **stdout must be unbuffered.** Python block-buffers stdout whenever it is not attached to a TTY — which, inside a container, it never is. A short log line sits in a buffer and is never flushed. The service works perfectly; the *only* symptom is a log line that isn't there and an alert that never fires. Both the application (`logging.basicConfig(stream=sys.stdout, …)`) and the Dockerfile (`PYTHONUNBUFFERED=1`) guard against this. Belt and braces, deliberately.
3. **The message must be plain text on stdout.** Structured logging libraries ship entries that land in `jsonPayload.message`, *not* `textPayload` — so a metric filtering on `textPayload` would stop matching.

The alert policy uses **`sum`** over a rolling window, not the default `rate`. `rate` converts the counter to events-per-second, so a single event over five minutes reads as `0.003`. That is still technically above zero and the alert *does* fire — but the chart looks empty and you cannot tell one event from ten.

> **A log-based metric does not exist until it has matched something.** If the metric picker can't find it, no matching line has been written yet. Send one high-risk request first.

### What this is, and what it isn't

This is **real model monitoring** — the same instinct as a 5xx alert, but watching *model outputs* rather than service errors. It is not complete: it only covers the real-time path. Alerting on the batch table ("more than N machines above threshold") would need a scheduled query plus a second metric.

---

## 11. CI/CD

GitHub Actions, authenticating to GCP with **Workload Identity Federation** — no stored key.

Instead of downloading a service-account JSON key and pasting it into GitHub secrets (where it lives forever, leaks eventually, and is among the most common findings in cloud security audits), GCP **trusts GitHub's OIDC token directly**. GitHub proves "this workflow is running in repository X"; GCP checks that against a trust condition and mints a short-lived token. **No long-lived secret exists to steal.**

The attribute condition is **mandatory**. Without it, *any* GitHub repository on the planet could authenticate to the project. And the trust is bound to the **exact** repository path — renaming the repo breaks CI/CD until the condition is updated.

On push to `main`, the workflow:

1. rebuilds the training image via Cloud Build;
2. redeploys the scorer service;
3. **redeploys the retrainer job.**

Step 3 is not obvious and it matters. A Cloud Run *job* runs a frozen image, and executing a job **never rebuilds it**. Without this step, a fix pushed to the retrainer would sit in `main` looking deployed while the job kept running the old image. This is also why the files the retrainer copies in (the SQL and the compiled pipeline spec) must be **committed**, not merely copied locally: CI builds from the repository, not from a laptop.

---

## 12. Operations runbook

### Cost hygiene — run at the end of every session

Three switches. The third is the one people forget.

**1. Drain the Dataflow streaming job.** It holds at least one worker VM for its entire life. `drain` lets in-flight windows finish writing; `cancel` drops them. Either stops the billing.

> **Killing the launch terminal does *not* stop the job.** The pipeline was submitted to Dataflow and runs *in the cloud*; Ctrl-C only detaches the terminal. The job keeps billing with the laptop shut. Always re-list active jobs afterwards and *see* it disappear rather than assuming it did — and if the job was launched in a fallback region, list *that* region, or you will happily walk away from a job that is still running, having "confirmed" it was stopped.

**2. Undeploy the endpoint's models — all of them.** An idle endpoint still bills for its node, and a retrain may have left more than one model deployed. Verify that the deployed-models list comes back empty.

**3. Pause the Cloud Scheduler drift job.** Left running, it fires daily forever. If it detects drift *after* everything has been torn down, it will submit a pipeline that **creates a brand-new billing endpoint** while nobody is looking.

Also stop the generator. It costs nothing, but it keeps filling BigQuery and slowly shifts the baseline that drift is measured against.

### Full teardown

The endpoint, the Dataflow job, and the Scheduler job all live **outside** Terraform, so they must be killed first. Then `terraform destroy` removes everything else — including the SQL-created tables, views, and the BQML model, because the dataset is declared with `delete_contents_on_destroy`.

### Restarting for a demo

Roughly 30 minutes end to end. The training run is the long pole. Order: `terraform apply` → generator → Dataflow → let it accumulate an hour or two of data → label → train → pipeline.

---

## 13. Failure-mode catalogue

Every entry here has been hit. They are grouped by *what the error looks like*, because that is how you'll meet them.

### The error is a liar

| Symptom | Actual cause |
|---|---|
| `ModuleNotFoundError: No module named 'pkg_resources'` while installing Beam | No compatible wheel exists for your interpreter version **or CPU architecture**, so pip silently fell back to **building from source**. The tell is a `.tar.gz` download instead of a `.whl`. The message mentions nothing about wheels, Python versions, or architecture. |
| `The replica workerpool0-0 exited with a non-zero status of 1`, empty log | Missing `logging.logWriter`. The container's stdout — including its traceback — was discarded. |
| `exec format error` on a Vertex training job | An `arm64` image built locally on Apple Silicon, running on `amd64` workers. Build with Cloud Build. |
| Endpoint deploy fails with an unpickling traceback | scikit-learn / numpy version mismatch between the training environment and the prebuilt serving container. Reads like a permissions problem. Isn't. |
| `NameError: name 'json' is not defined` — but only on Dataflow, never locally | `save_main_session=True` is missing. Workers unpickle your functions without the module's imports. |
| `Not found: Dataset …` when creating a scheduled query | The query editor's processing location doesn't match the dataset's multi-region. It is **fixed at creation** and cannot be changed afterwards. |
| `DataflowRuntimeException: Workflow failed.` and nothing else | The real error is never in the terminal. It's in the job's **Diagnostics** tab. Usually a zone capacity stockout. |
| `Missing required option: project` at Beam launch | Application argparse consumed `--project`, which Beam itself owns. |
| Drift check reports `z_shift ≈ 0` no matter what | The Dataflow job isn't running, so the "recent" window is stale. |
| Drift check reports `z_shift = NULL` | One side of the comparison has no rows. |
| Retrain runs, changes nothing | Labels weren't rebuilt first; it re-fit the same model on the same snapshot. |

### The environment, not the code

- **Beam wheels on Apple Silicon.** Two separate walls: no wheels at all for very new interpreters, and no macOS-arm64 wheels for older Beam releases. Both fail by silently source-building. Clear both.
- **Zone capacity stockouts** (`ZONE_RESOURCE_POOL_EXHAUSTED`). Google is out of VMs in that zone. Your code, IAM, and quota are all fine. Levers, in order: pin a different zone; shrink the machine type (stockouts are machine-type-specific); move region. Recognizing this as a *cloud-provider* condition rather than a bug in your pipeline — and that the fix is zone/machine-shape, not code — is the right instinct.
- **API enablement is eventually consistent.** The first `terraform apply` may fail with a 403 on an API it just enabled. It converges; wait and re-apply.
- **GCS soft-delete costs money.** The default retention means you pay storage on every deleted temp object for a full week — and Dataflow churns temp objects constantly. It is disabled in Terraform.

### The thing that runs but is wrong

- **"The job runs" is not the same as "the permissions are right."** A missing *read* permission on a *config* surface fails as a warning, not an error. It costs nothing today and produces phantom problems later.
- **A saved-but-never-executed scheduled query** is the most common false pass in the whole build. Check the run history.
- **A test payload outside the training distribution** returns a meaningless value, not an error. Gradient-boosted trees don't extrapolate — they return whatever a leaf learned.
- **A dashboard aggregation defaulting to `SUM`** on a one-row-per-machine mart happens to look right, until a machine gets two rows and the table cheerfully reports a risk of 1.7.

---

## 14. Cost model

**The system is not zero-cost while idle.** Two meters run continuously while switched on:

| Resource | Why it doesn't scale to zero | Order of magnitude |
|---|---|---|
| Dataflow streaming job | Holds ≥1 worker VM for its entire life. Streaming Engine also bills per GB of streaming data processed — a separate line item. | single-digit €/day |
| Vertex AI online endpoint | Holds one node **per deployed model**. Superseded models keep their nodes until explicitly undeployed. | similar |

Everything else — Cloud Run (scales to zero), Pub/Sub, BigQuery storage and queries at this volume, one-shot training jobs, the pipeline when it isn't running — is free-tier or pennies.

A budget alert with staged thresholds is the safety net, and it is set up before anything is provisioned.

---

## 15. Security posture

- **No service-account keys exist anywhere.** Local development uses Application Default Credentials; CI/CD uses Workload Identity Federation. There is no long-lived credential in the repository, in GitHub secrets, or on any developer machine.
- **The WIF trust condition is bound to one exact repository path.** Without that condition, any repository on GitHub could authenticate.
- **Least-privilege service accounts.** Four identities, each holding only the roles its workload actually exercises. Notably, the Dataflow worker does *not* get `bigquery.jobUser` (it only streams rows in, never queries), and the scorer does not get it either.
- **The scoring service is not publicly reachable** — it is deployed without unauthenticated access, and requests carry an identity token.
- **Reading a declared subscription rather than a topic** keeps the Dataflow worker out of Pub/Sub admin territory.
- **`.gitignore` excludes Terraform state, `.tfvars`, and anything matching a key filename**, and the pre-push check is to read `git status` before committing.

---

## 16. Known limitations

Stated plainly, because a system whose author can't name its weaknesses is a system whose author hasn't looked.

1. **The data is synthetic.** The skill on display is the pipeline and the ML lifecycle, not sensor physics. The generator does bake in a genuinely learnable signal (and a deliberate noise distractor), but it is a simulator.
2. **The model is a regressor fit to a binary target**, purely to get a continuous score out of the prebuilt serving container's `predict()`. A Custom Prediction Routine is the correct fix.
3. **The deploy gate compares against a fixed constant**, not against the currently deployed model, and it evaluates on the training job's own test split rather than a holdout the pipeline controls. Both are named upgrades.
4. **The drift baseline is "the oldest rows we happen to have,"** not the exact data snapshot the deployed model was trained on. In production it would be pinned to the latter.
5. **Drift monitoring watches one feature.** A real deployment would monitor the full feature distribution and prediction distribution, most likely via managed model monitoring rather than hand-rolled SQL.
6. **Alerting covers only the real-time path.** The batch path has no alert.
7. **There is no dead-letter path** on the Dataflow pipeline. Malformed telemetry would be dropped, not quarantined.
8. **The endpoint, the Scheduler job, and the scheduled queries live outside Terraform.** The endpoint necessarily so (the pipeline creates it); the others are honest gaps.
9. **The problem is framed as binary classification within a fixed horizon.** RUL regression — hours-to-failure — is the more useful formulation and the more interesting modelling problem.
