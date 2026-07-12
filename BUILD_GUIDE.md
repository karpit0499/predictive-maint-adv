# Predictive Maintenance on GCP — Build Guide

A production-style, **streaming + MLOps** pipeline on Google Cloud (`predictive-maint-adv`), built end to end. Simulated machines emit sensor telemetry; readings stream through **Pub/Sub**; **Dataflow** turns the raw stream into windowed features and lands them in BigQuery; failures are logged as **ground-truth events**; a **Vertex AI Pipeline** trains a failure-risk model, evaluates it, and *conditionally* deploys it to an online endpoint; a scoring service flags machines about to fail; and the whole thing **retrains itself on drift**. Everything is Terraform-provisioned and shipped by keyless GitHub Actions.

That "the model retrains itself when the data drifts" loop — continuous training (CT), not just CI/CD — is the part that is usually stubbed out or skipped, and it is the reason this build is worth doing end to end.

> **How to use this guide.** Don't do it all in one sitting. **Phases 0–5 are the Core Path** — finish those and you have a real streaming pipeline that trains a model. **Phases 6–10 are the MLOps meat** (custom training, pipelines, endpoints, drift-triggered retraining). **Phases 11–14** add analytics, CI/CD, and packaging. Every phase ends with a ✅ checkpoint, so you always stop with something working.

## What you'll build

```
   synthetic fleet: 60 machines with a hidden, decaying "health"
             │                                    │
             │ telemetry (publish)                │ failure events (ground truth)
             ▼                                    ▼
   ┌──────────────────┐                 ┌──────────────────────┐
   │  Pub/Sub         │                 │  BigQuery            │
   │  sensor-telemetry│                 │  failure_events      │
   └────────┬─────────┘                 └──────────┬───────────┘
            ▼                                      │
   ┌──────────────────────────┐                    │
   │  Dataflow (Apache Beam)  │                    │
   │  2-min windows/machine   │                    │
   └────────┬─────────────────┘                    │
            ▼                                      │
   ┌────────────────────────────┐                  │
   │  BigQuery                  │                  │
   │   telemetry_raw            │                  │
   │   features_windowed  ──────┼──── labeled by ──┘
   └────────┬───────────────────┘   (will it fail in
            ▼                        the next 10 min?)
   ┌────────────────────────────────────────────────┐
   │  Vertex AI Pipeline (KFP v2)                   │
   │    train  →  evaluate  →  IF auc ≥ gate: deploy│
   │      │                          │              │
   │      ▼                          ▼              │
   │  Model Registry          Online Endpoint       │
   └────────────────────────────────┬───────────────┘
                                    ▼
              scoring service (Cloud Run) → predictions → alerts
                                    ▼
        drift check (Cloud Run job, Cloud Scheduler) → re-run the pipeline
                                    ▼
              dbt marts + Looker Studio fleet-health dashboard

  Terraform-provisioned · keyless GitHub Actions · endpoint + Dataflow are the only meters
```

## What this project covers

| Capability | Where it's built | Why it matters |
|-------|--------------------|---------------|
| Infrastructure as Code (Terraform) | Phase 1 | Reproducible, reviewable, destroyable in one command |
| Streaming data engineering (Pub/Sub + Dataflow/Beam) | Phase 2–3 | Unbounded data has to be windowed in flight, not batched later |
| Windowed feature engineering | Phase 3 | Features computed once, at ingest, instead of re-derived per query |
| **Honest labeling (ground truth, leakage, censoring)** | Phase 4 | The most common source of a model that scores well and predicts nothing |
| BigQuery ML (fast baseline models) | Phase 5 | A number with nothing to compare it against is not a result |
| Vertex AI custom training + Model Registry | Phase 6 | Portable, versioned artifacts instead of a pickle on a laptop |
| Vertex AI Pipelines (Kubeflow/KFP v2) + conditional deploy | Phase 7 | The gate that stops a bad retrain replacing a good model |
| Online endpoints & real-time inference | Phase 8 | Low-latency scoring for a single machine on demand |
| Batch prediction + Cloud Monitoring alerts | Phase 9 | Fleet-wide scoring that survives the endpoint being undeployed |
| Drift detection + continuous training (CT) | Phase 10 | Retraining on *data* change, not only on *code* change |
| Analytics engineering (dbt) + BI (Looker) | Phase 11–12 | SQL that is versioned, tested, and reviewable |
| CI/CD (GitHub Actions, keyless WIF) | Phase 13 | Deploys with no service-account key stored anywhere |

> **Product notes — read before building:**
> - The `requirements.txt` files pin exact dependency versions on purpose — several of those pins are load-bearing and the reasoning is documented at the point of use. Everything else (image tags, SDK surfaces, console layouts) moves: confirm the current **prebuilt scikit-learn prediction container tag** (Phase 6) in Google's "Pre-built containers for prediction" docs before you build, and swap it in everywhere it appears.
> - **The serving container and your training environment must agree on scikit-learn.** Whichever prebuilt image you pick ships one specific scikit-learn minor release, and `training/requirements.txt` must pin to match it. Mismatch them and the endpoint fails to unpickle `model.joblib` — a deploy-time error that reads like a permissions problem and isn't.
> - **KFP v2 syntax only.** The older KFP DSL will not compile against the SDK used here.

## Conventions used throughout

- Project **`predictive-maint-adv`**, region **`europe-west3`** (Frankfurt); BigQuery multi-region **`EU`**; bucket **`predictive-maint-adv-artifacts`**.
- **Region heads-up:** a few Vertex AI features are documented against `us-central1`. Everything here works in `europe-west3`; if a specific API 404s regionally, fall back to `us-central1` **for that resource only** and keep data in the EU.
- **Two virtual environments.** Beam pins old versions of the Google client libraries and will fight the Vertex/BigQuery SDKs if they share an environment:

  ```bash
  mkdir -p ~/dev/predictive-maint-adv && cd ~/dev/predictive-maint-adv
  python3 --version                              # MUST be 3.10, 3.11 or 3.12. If 3.13+, STOP and read the box below
  /usr/local/bin/python3 -m venv .venv           # general: SDKs, training, analysis, dbt
  /usr/local/bin/python3 -m venv .venv-beam      # ONLY Apache Beam / Dataflow
  ```

  | venv | Used by |
  |------|---------|
  | `.venv` | everything: generator, training, pipeline, scorer, retrainer, dbt |
  | `.venv-beam` | **only** `dataflow/stream_features.py` |

> **⚠️ Build the venvs on Python 3.12 — check this *before* you create them.** `/usr/local/bin/python3` on a current macOS install is very likely **3.13 or 3.14**. Run `python3 --version` and read it. Dataflow's managed worker images target **3.12**, and matching your local interpreter to the worker is the difference between "it runs" and a pickling error you cannot debug from a log. If `python3 --version` says 3.13+, install 3.12 and build **both** venvs from *that* interpreter:
>
> ```bash
> brew install python@3.12
> /opt/homebrew/bin/python3.12 -m venv .venv
> /opt/homebrew/bin/python3.12 -m venv .venv-beam
> ```
>
> **The failure mode is a liar.** When no compatible Beam wheel exists for your interpreter *or your CPU architecture*, pip does not say so. It silently falls back to **building Beam from source**, and that build dies with:
>
> ```
> ModuleNotFoundError: No module named 'pkg_resources'
> ERROR: Failed to build 'apache-beam' when getting requirements to build wheel
> ```
>
> (setuptools 81+ stopped shipping `pkg_resources`; setuptools 83 removed it.) Nothing in that message mentions Python versions, wheels, or `arm64` — so it baits you into `pip install setuptools` and pinning `grpcio-tools`. **Don't.** The rule: *if pip is source-building Beam, you have already lost.* The giveaway is a line like `Using cached apache_beam-2.75.0.tar.gz` — a **`.tar.gz`** instead of a **`.whl`**. Find the missing wheel instead of fighting the build. On Apple Silicon there are **two** separate wheel walls, and you have to clear both:
>
> | Wall | Symptom | Fix |
> |------|---------|-----|
> | Interpreter too new (3.13+) | no `cp313`/`cp314` wheel → source build | build the venv from Python **3.12** |
> | Beam too old (< 2.70) | no macOS **arm64** wheel → source build | pin Beam **2.75.0** (Phase 3) |
>
> Rebuilding `.venv-beam` correctly, from scratch:
>
> ```bash
> cd ~/dev/predictive-maint-adv
> deactivate 2>/dev/null
> rm -rf .venv-beam
> /opt/homebrew/bin/python3.12 -m venv .venv-beam
> source .venv-beam/bin/activate
> python --version                  # MUST print Python 3.12.x before you go further
> pip install --upgrade pip
> ```
>
> If `.venv` was also built from a 3.13+ interpreter, rebuild it the same way — scikit-learn and the Google SDK wheels lag new Python releases too, and you will hit the same class of silent source-build failure in Phase 5.

- **Command blocks vs file blocks.** `gcloud` / `git` / `bq` / `curl` blocks are terminal commands. Python, YAML, HCL, and SQL blocks are the *contents of a file* (or the BigQuery query editor) — don't paste them into the terminal.

---

# CORE PATH

# Phase 0 — Foundations & local tools (~1 hr)

**Goal:** Install the tools, create the project, and get billing + a budget alert guarding you *before* you provision anything.

## Step 0.1 — Install the tools

1. **Google Cloud CLI (`gcloud`)** — from `cloud.google.com/sdk`.
2. **Terraform**:
   ```bash
   brew tap hashicorp/tap
   brew install hashicorp/tap/terraform
   terraform -version
   ```
3. **git** + a GitHub account (needed for Phase 13's CI/CD).
4. **Docker Desktop** — for local container testing. The actual image builds happen **server-side via Cloud Build**, which is also what saves you from the Apple-Silicon arch trap (your Mac is arm64; Vertex AI runs amd64 — a local `docker build` produces an image that dies with `exec format error`).

> **The `gcloud: command not found` gotcha.** If a new terminal can't find `gcloud`, its `bin/` isn't on your `PATH`. Keep the SDK at a stable path (e.g. `~/google-cloud-sdk`) and add its init lines to `~/.zshrc`:
> ```bash
> echo 'source "$HOME/google-cloud-sdk/path.zsh.inc"' >> ~/.zshrc
> echo 'source "$HOME/google-cloud-sdk/completion.zsh.inc"' >> ~/.zshrc
> source ~/.zshrc
> gcloud --version
> ```

## Step 0.2 — Authenticate (two logins, and you need both)

```bash
gcloud auth login                        # authorizes the CLI (you)
gcloud auth application-default login    # writes Application Default Credentials for local SDKs/Terraform/dbt
```

The second login is why every local script in this guide works with **no key files**.

## Step 0.3 — Create the project and link billing

```bash
gcloud projects create predictive-maint-adv --name="Predictive Maintenance"
gcloud config set project predictive-maint-adv
gcloud billing accounts list        # find OPEN: True, copy that ACCOUNT_ID
gcloud billing projects link predictive-maint-adv --billing-account=XXXXXX-XXXXXX-XXXXXX
gcloud billing projects describe predictive-maint-adv   # want: billingEnabled: true
```

> **The 30-character name trap.** `--name` is the project *display name* and is capped at **30 characters**. `"Predictive Maintenance Advanced"` is 31 and fails with `INVALID_ARGUMENT`. The shortened `"Predictive Maintenance"` (22) is what we use — the project **ID** (`predictive-maint-adv`) is a separate field and is unchanged, so every later command still references `predictive-maint-adv`.

> **The closed-account trap.** `link` returns *no error* even when pointed at a **closed** billing account — nothing errors, and then nothing provisions. Always confirm `billingEnabled: true` with `describe`.

## Step 0.4 — Budget alert (your safety net)

Console → **Billing → Budgets & alerts → Create budget** → scope to `predictive-maint-adv`, set **€10**, thresholds **50 / 90 / 100 %**.

> **The cost map — memorize these two.** This project is **not** €0-idle. Two things bill *while switched on*: (1) the **Dataflow streaming job** (one small worker VM, single-digit €/day — plus, because Step 3.3 enables **Streaming Engine**, a separate charge per GB of streaming data *processed*; at 60 messages/sec that is cents, but it is a distinct line item you'll see on the bill), and (2) the **Vertex AI online endpoint** (one node, similar). Neither scales to zero. The discipline: run them for a session, then **drain the Dataflow job and undeploy the endpoint** when you finish. The cost-hygiene section at the end has the exact commands — and one more off-switch (the Cloud Scheduler job) that is easy to forget and can quietly re-create the endpoint after you've torn it down.

✅ **Checkpoint 0:** `gcloud config get-value project` prints `predictive-maint-adv`, `billingEnabled: true`, budget alert exists.

---

# Phase 1 — Provision the foundation with Terraform (~2 hrs)

**Goal:** Declare the whole substrate — APIs, Pub/Sub, BigQuery, a GCS bucket, and service accounts — in code, and create it in one command.

> **Term — Infrastructure as Code (IaC):** defining infra in versioned code files instead of by hand. Reproducible, reviewable, and deletable in one command (`terraform destroy`).

## Step 1.1 — Create the folder

```bash
mkdir -p ~/dev/predictive-maint-adv/infra
cd ~/dev/predictive-maint-adv/infra
```

## Step 1.2 — Write `main.tf`

```hcl
terraform {
  required_providers {
    google = { source = "hashicorp/google", version = "~> 5.0" }
  }
}

provider "google" {
  project = "predictive-maint-adv"
  region  = "europe-west3"
}

# ── APIs ────────────────────────────────────────────────────────────────
resource "google_project_service" "apis" {
  for_each = toset([
    "aiplatform.googleapis.com",            # Vertex AI: training, pipelines, endpoints
    "pubsub.googleapis.com",
    "dataflow.googleapis.com",
    "run.googleapis.com",
    "bigquery.googleapis.com",
    "bigquerydatatransfer.googleapis.com",  # BigQuery SCHEDULED QUERIES run on this API
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "monitoring.googleapis.com",
    "cloudscheduler.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}

# ── Pub/Sub: the telemetry stream ───────────────────────────────────────
resource "google_pubsub_topic" "telemetry" {
  name       = "sensor-telemetry"
  depends_on = [google_project_service.apis]
}

# A dedicated subscription for Dataflow. (If Beam reads from a *topic* it
# silently creates its own subscription, which needs broad pubsub.editor
# rights. Reading a *declared* subscription is least-privilege.)
resource "google_pubsub_subscription" "telemetry_df" {
  name                 = "telemetry-df-sub"
  topic                = google_pubsub_topic.telemetry.name
  ack_deadline_seconds = 60
}

# ── GCS: Dataflow staging/temp + model artifacts + pipeline root ─────────
resource "google_storage_bucket" "artifacts" {
  name                        = "predictive-maint-adv-artifacts" # GLOBALLY unique — add a suffix if taken
  location                    = "EU"
  uniform_bucket_level_access = true
  force_destroy               = true

  # GCS enables a 7-day soft-delete retention by default. Dataflow churns temp
  # objects constantly, and you are billed for storage on every deleted object
  # for the full retention window. 0 = off. Beam warns about this at launch.
  soft_delete_policy {
    retention_duration_seconds = 0
  }

  depends_on = [google_project_service.apis]
}

# ── BigQuery ────────────────────────────────────────────────────────────
# delete_contents_on_destroy = true is REQUIRED. Later phases create tables,
# views and a BQML model in this dataset via SQL (they are deliberately NOT
# declared here — one owner per object). Without this flag, `terraform destroy`
# fails on a non-empty dataset and you hand-delete things in the console.
resource "google_bigquery_dataset" "maint" {
  dataset_id                 = "maintenance"
  location                   = "EU"
  delete_contents_on_destroy = true
  depends_on                 = [google_project_service.apis]
}

resource "google_bigquery_table" "telemetry_raw" {
  dataset_id          = google_bigquery_dataset.maint.dataset_id
  table_id            = "telemetry_raw"
  deletion_protection = false

  time_partitioning {
    type  = "DAY"
    field = "event_time"
  }

  schema = jsonencode([
    { name = "machine_id",  type = "STRING" },
    { name = "event_time",  type = "TIMESTAMP" },
    { name = "temperature", type = "FLOAT" },
    { name = "vibration",   type = "FLOAT" },
    { name = "rpm",         type = "FLOAT" },
    { name = "pressure",    type = "FLOAT" },
    { name = "voltage",     type = "FLOAT" },
  ])
}

# Windowed features. NOTE: there is NO label column here. Dataflow cannot know
# the future, so the label is computed later, in BigQuery, into a SEPARATE
# table (features_labeled, Phase 4). A permanently-NULL label column in a
# streaming table is dead weight that every downstream query has to EXCEPT out.
resource "google_bigquery_table" "features" {
  dataset_id          = google_bigquery_dataset.maint.dataset_id
  table_id            = "features_windowed"
  deletion_protection = false

  time_partitioning {
    type  = "DAY"
    field = "window_end"
  }

  schema = jsonencode([
    { name = "machine_id",     type = "STRING" },
    { name = "window_end",     type = "TIMESTAMP" },
    { name = "temp_mean",      type = "FLOAT" },
    { name = "temp_max",       type = "FLOAT" },
    { name = "vibration_mean", type = "FLOAT" },
    { name = "vibration_std",  type = "FLOAT" },
    { name = "rpm_mean",       type = "FLOAT" },
    { name = "pressure_mean",  type = "FLOAT" },
    { name = "voltage_mean",   type = "FLOAT" },
    { name = "reading_count",  type = "INTEGER" }, # data quality only — NOT a model feature
  ])
}

# GROUND TRUTH. The generator writes one row here every time a machine actually
# dies. This is the project's "maintenance log" — a system of record SEPARATE
# from the sensor stream. Phase 4 labels from THIS table, never by thresholding
# a sensor reading that is also a model input. (See the circular-label gotcha.)
resource "google_bigquery_table" "failure_events" {
  dataset_id          = google_bigquery_dataset.maint.dataset_id
  table_id            = "failure_events"
  deletion_protection = false

  schema = jsonencode([
    { name = "machine_id", type = "STRING" },
    { name = "failed_at",  type = "TIMESTAMP" },
  ])
}

resource "google_bigquery_table" "predictions" {
  dataset_id          = google_bigquery_dataset.maint.dataset_id
  table_id            = "predictions"
  deletion_protection = false

  time_partitioning {
    type  = "DAY"
    field = "scored_at"
  }

  schema = jsonencode([
    { name = "machine_id",    type = "STRING" },
    { name = "scored_at",     type = "TIMESTAMP" },
    { name = "failure_prob",  type = "FLOAT" },
    { name = "model_version", type = "STRING" },  # WHICH model produced this score
  ])
}

# Drift log (Phase 10 writes here)
resource "google_bigquery_table" "drift_log" {
  dataset_id          = google_bigquery_dataset.maint.dataset_id
  table_id            = "drift_log"
  deletion_protection = false

  schema = jsonencode([
    { name = "checked_at", type = "TIMESTAMP" },
    { name = "metric",     type = "STRING" },
    { name = "z_shift",    type = "FLOAT" },
    { name = "retrained",  type = "BOOLEAN" },
  ])
}

# ── Service accounts (least privilege) ──────────────────────────────────
resource "google_service_account" "dataflow" {
  account_id   = "df-worker"
  display_name = "Dataflow Worker"
}

resource "google_service_account" "pipeline" {
  account_id   = "vertex-pipeline"
  display_name = "Vertex Pipeline Runner"
}

resource "google_service_account" "scorer" {
  account_id   = "scorer"
  display_name = "Scoring Service"
}

# ── IAM ─────────────────────────────────────────────────────────────────
# roles/bigquery.dataEditor lets an SA read/write table DATA; running a QUERY
# additionally needs roles/bigquery.jobUser. Any SA that executes SQL gets both.
# roles/bigquery.readSessionUser is a THIRD, separate permission: pandas'
# .to_dataframe() does not read results over the normal query API — it opens a
# BigQuery Storage Read API session. An SA can hold dataEditor + jobUser, run
# the query successfully, and still die on `bigquery.readsessions.create` while
# pulling the rows back. train.py calls .to_dataframe(), so the pipeline SA
# needs it.
# artifactregistry.reader on the pipeline SA is what lets Vertex AI PULL your
# training image — without it the training job dies on an image pull, which
# looks nothing like a permissions error until you read the log closely.
#
# roles/logging.logWriter is the one that hides all the others, and it is the
# single most expensive omission in this file. A Vertex training worker and a
# Cloud Run container both ship their stdout to Cloud Logging *as the service
# account they run under*. Without logWriter that write is denied and the output
# is simply discarded — so a crashing container produces "The replica
# workerpool0-0 exited with a non-zero status of 1" and NOTHING ELSE. No
# traceback, no print(), no clue. You end up debugging the wrong layer for an
# hour. It is also load-bearing for Phase 9: the alert there is a log-based
# metric on a line the scorer prints, and a log that never arrives can never
# match. Every SA that runs a container gets logWriter.
# (df-worker is the exception — roles/dataflow.worker already includes
# logging.logEntries.create, so Dataflow workers can write logs out of the box.)
locals {
  df_roles = [
    "roles/dataflow.worker",
    "roles/pubsub.subscriber",     # consume messages
    "roles/pubsub.viewer",         # READ the subscription's config — see the gotcha in Step 3.3
    "roles/bigquery.dataEditor",   # Dataflow only streams rows in — never runs a query
    "roles/storage.objectAdmin",
  ]
  pipeline_roles = [
    "roles/aiplatform.user",
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",           # train.py + the drift check run queries
    "roles/bigquery.readSessionUser",   # train.py's .to_dataframe() → Storage Read API
    "roles/storage.objectAdmin",
    "roles/artifactregistry.reader",
    "roles/logging.logWriter",          # training worker + retrainer job stdout → Cloud Logging
  ]
  scorer_roles = [
    "roles/aiplatform.user",
    "roles/bigquery.dataEditor",   # the scorer only streams inserts — no queries
    "roles/logging.logWriter",     # the Cloud Run container's stdout — INCLUDING the
                                   # HIGH_RISK line Phase 9's alert fires on
  ]
}

resource "google_project_iam_member" "df" {
  for_each = toset(local.df_roles)
  project  = "predictive-maint-adv"
  role     = each.value
  member   = "serviceAccount:${google_service_account.dataflow.email}"
}

resource "google_project_iam_member" "pipe" {
  for_each = toset(local.pipeline_roles)
  project  = "predictive-maint-adv"
  role     = each.value
  member   = "serviceAccount:${google_service_account.pipeline.email}"
}

resource "google_project_iam_member" "score" {
  for_each = toset(local.scorer_roles)
  project  = "predictive-maint-adv"
  role     = each.value
  member   = "serviceAccount:${google_service_account.scorer.email}"
}
```

> **HCL formatting gotcha (this *will* bite if you compress it).** Every attribute inside a nested block goes on its **own line**. `time_partitioning { type = "DAY" field = "window_end" }` on one line is a **parse error**. If `terraform plan` throws a cryptic `Argument or block definition required`, look for a nested block you collapsed onto one line. The `schema = jsonencode([...])` lists are safe — they're JSON strings, not HCL blocks.

## Step 1.3 — Apply and verify

```bash
cd ~/dev/predictive-maint-adv/infra
terraform init       # downloads the Google provider
terraform plan       # read this — it's your infra before it exists
terraform apply      # type 'yes'
terraform state list
```

> **Gotcha — the first `terraform apply` can fail on API enablement.** You may see `Error 403: Cloud Pub/Sub API has not been used in project predictive-maint-adv before or it is disabled`. The `depends_on` lines order the graph correctly, but API enablement is **eventually consistent** — the API reports "on" a few seconds before Google's backends agree. This is not a code bug: wait ~30 seconds and run `terraform apply` again. It converges.

If the bucket name is taken (they're global), change it in `main.tf` — and then in **every later phase where the bucket appears** (Dataflow flags, pipeline root, training). Do that find-replace now, not later.

**What is deliberately NOT in this file:** `features_labeled` and `v_training` (Phase 4), the `bqml_failure` model (Phase 5), and every dbt model (Phase 11). Those are created by SQL. Declaring an object in both Terraform *and* SQL makes the two fight over its schema, so each object gets exactly one owner.

✅ **Checkpoint 1:** `terraform state list` shows the topic, the subscription, the bucket, **five** BigQuery tables (`telemetry_raw`, `features_windowed`, `failure_events`, `predictions`, `drift_log`), and three service accounts.

And confirm both container-running SAs can actually write logs — this one line saves you the worst debugging session in the guide:

```bash
gcloud projects get-iam-policy predictive-maint-adv \
  --flatten="bindings[].members" \
  --filter="bindings.role=roles/logging.logWriter" \
  --format="value(bindings.members)" | grep -E 'vertex-pipeline|scorer'
```

Both `vertex-pipeline@…` and `scorer@…` must appear. If you ever grant this by hand with `gcloud add-iam-policy-binding`, **put it in `main.tf` too** — otherwise the next `terraform apply` reconciles it away and the bug comes back with no memory of why.

---

# Phase 2 — Simulate a fleet of machines (~1.5 hrs)

**Goal:** You don't have real turbines, so you'll *manufacture* believable telemetry — including machines that gradually degrade, fail, and get replaced. Good synthetic data with a real, learnable failure signal is what makes every later phase honest.

> **Why synthetic is fine here:** the skill on display is the *pipeline and the ML lifecycle*, not sensor physics. But the data must contain a learnable pattern (temperature and vibration creep up before failure) or your model has nothing to find. This generator bakes that in — and, crucially, it **logs each real failure as a ground-truth event**, so your labels never have to be reverse-engineered from the sensor readings themselves.

## Step 2.1 — The generator

```bash
mkdir -p ~/dev/predictive-maint-adv/generator
```

**`generator/generate.py`**:

```python
import os
import json
import time
import random
from datetime import datetime, timezone

from google.cloud import pubsub_v1, bigquery

PROJECT = os.environ["PROJECT"]
publisher = pubsub_v1.PublisherClient()
TOPIC = publisher.topic_path(PROJECT, "sensor-telemetry")
bq = bigquery.Client(project=PROJECT)
FAILURES = f"{PROJECT}.maintenance.failure_events"

N_MACHINES = int(os.environ.get("N_MACHINES", "60"))
SLEEP = float(os.environ.get("SLEEP", "1.0"))        # seconds between rounds — LEAVE AT 1.0
DECAY = float(os.environ.get("DECAY", "0.0001"))     # ~43-minute machine lifecycle
VIB_OFFSET = float(os.environ.get("VIB_OFFSET", "0.0"))   # Phase 10 injects drift with this

machines = {f"M{i:03d}": {"health": random.uniform(0.7, 1.0)} for i in range(N_MACHINES)}


def reading(mid, m):
    wear = 1.0 - m["health"]                          # 0 = healthy, 1 = dead
    return {
        "machine_id": mid,
        "event_time": datetime.now(timezone.utc).isoformat(),
        "temperature": round(60 + 40 * wear + random.gauss(0, 2), 2),            # heats up as it wears
        "vibration":   round(0.2 + 2.5 * (wear ** 2) + VIB_OFFSET
                             + random.gauss(0, 0.05), 3),                        # spikes late
        "rpm":         round(1500 - 200 * wear + random.gauss(0, 15), 1),
        "pressure":    round(30 - 8 * wear + random.gauss(0, 0.5), 2),
        "voltage":     round(230 + random.gauss(0, 1.5), 2),                     # pure noise (a distractor)
    }


def log_failure(mid):
    """GROUND TRUTH. In a real plant this row comes from the maintenance system,
    not from the sensors. Keeping it in its own table is what lets Phase 4 label
    honestly instead of thresholding a column it also trains on."""
    errs = bq.insert_rows_json(FAILURES, [{
        "machine_id": mid,
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }])
    if errs:
        print("failure_events insert errors:", errs)


print(f"publishing {N_MACHINES} machines → {TOPIC}")
print(f"DECAY={DECAY}  SLEEP={SLEEP}  VIB_OFFSET={VIB_OFFSET}")
failures = 0
while True:
    for mid, m in machines.items():
        publisher.publish(TOPIC, json.dumps(reading(mid, m)).encode("utf-8"))
        m["health"] -= random.uniform(0.5, 1.5) * DECAY * (1 + (1 - m["health"]))  # decays faster when worn
        if random.random() < 0.002:                    # occasional sudden fault
            m["health"] -= random.uniform(0.05, 0.15)
        if m["health"] <= 0.05:                        # FAILURE → log it, then replace the machine
            failures += 1
            print(f"{mid} FAILED (#{failures}) — logged to failure_events, machine replaced")
            log_failure(mid)
            m["health"] = random.uniform(0.85, 1.0)
    time.sleep(SLEEP)
```

**`generator/requirements.txt`**:

```
google-cloud-pubsub==2.23.0
google-cloud-bigquery==3.25.0
```

## Step 2.2 — Run it

```bash
cd ~/dev/predictive-maint-adv
source .venv/bin/activate
pip install -r generator/requirements.txt
PROJECT=predictive-maint-adv python generator/generate.py
```

Leave this running in its own terminal while you build Phase 3. Within a couple of minutes you'll see `FAILED` lines scroll past.

> **⚠️ The generator never stops — by design.** It is an infinite publish loop simulating a factory floor, and a factory floor does not "finish." It runs until **Ctrl-C**. The `FAILED` lines are not errors — they are the *ground truth* you are deliberately manufacturing: each one means a machine crossed its failure threshold, got logged to `failure_events`, and was replaced with a fresh unit. Those rows become the **labels** in Phase 4. A machine ID appearing twice (`M010 FAILED (#11)` … `M010 FAILED (#53)`) is correct: it was replaced, lived a second lifecycle, and died again.
>
> A healthy 45-minute run at the settings above produces roughly **50–60 failures** across 60 machines. Wildly more means your `DECAY`/`SLEEP` are wrong — read the next box.

> **⚠️ The parameters are load-bearing — and lowering `SLEEP` to "speed things up" destroys the project.** This is the single most important box in the guide, so here is the arithmetic.
>
> Dataflow aggregates into **fixed 2-minute wall-clock windows** (Phase 3). For a window to *see* a machine degrading, the machine's lifetime must be **much longer than one window**. At the settings above, a machine lives ≈ **43 minutes ≈ 21 windows** — a clean, visible ramp from healthy to dead.
>
> Now halve `SLEEP`. Each machine decays twice as fast *in wall-clock time*, so its lifetime halves — but the window is still 2 minutes. Windows-per-lifecycle halves. Keep going and you reach the trap: at `SLEEP=0.05, DECAY=0.02` (a "fast-forward burst" that sounds clever), a machine is born, degrades, dies, and is replaced roughly **80 times inside a single 2-minute window**. Every window then averages 80 complete lifecycles and comes out identical to every other window. The features become pure noise, the model can learn nothing, and it looks like the *model* is broken.
>
> **To collect data faster, add machines — never subtract sleep.** `N_MACHINES=60` at `SLEEP=1.0` yields (measured by simulating this exact loop):
>
> | Generator time | Failures | Labeled windows | Positive rate |
> |---|---|---|---|
> | 1 hour  | ~63  | ~1,800 | ~17 % |
> | 2 hours | ~153 | ~3,600 | ~21 % |
>
> One to two hours is enough for everything downstream. Start it, build Phase 3, and let it run.

## Step 2.3 — Verify both streams

Telemetry (peek at one message without consuming it — `pull` does **not** ack unless you pass `--auto-ack`):

```bash
gcloud pubsub subscriptions pull telemetry-df-sub --limit=1 --format=json
```

Ground truth (this table fills within a few minutes of starting):

```bash
bq query --use_legacy_sql=false \
  'SELECT COUNT(*) AS failures, MIN(failed_at) AS first, MAX(failed_at) AS latest
   FROM maintenance.failure_events'
```

✅ **Checkpoint 2:** the `pull` returns a real telemetry message, and `failure_events` is accumulating rows.

---

# Phase 3 — Streaming feature engineering with Dataflow (~3 hrs)

**Goal:** Consume the raw stream and, in flight, compute **windowed features** per machine (rolling stats over 2-minute windows), writing both raw rows and feature rows to BigQuery. This is the phase where the system stops being a queue and starts being a data pipeline.

> **Term — Apache Beam / Dataflow:** Beam is a programming model for data pipelines that run the same code in batch *or* streaming; **Dataflow** is Google's managed runner. **Windowing** chops an unbounded stream into finite chunks ("every 2 minutes, per machine") so you can aggregate something that never ends.

## Step 3.1 — The pipeline

```bash
mkdir -p ~/dev/predictive-maint-adv/dataflow
```

**`dataflow/stream_features.py`**:

```python
import json
import argparse
import statistics as st

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms import window as beam_window

RAW_SCHEMA = (
    "machine_id:STRING,event_time:TIMESTAMP,temperature:FLOAT,"
    "vibration:FLOAT,rpm:FLOAT,pressure:FLOAT,voltage:FLOAT"
)
# No label column: a streaming job cannot know the future. Phase 4 adds it.
FEAT_SCHEMA = (
    "machine_id:STRING,window_end:TIMESTAMP,temp_mean:FLOAT,temp_max:FLOAT,"
    "vibration_mean:FLOAT,vibration_std:FLOAT,rpm_mean:FLOAT,pressure_mean:FLOAT,"
    "voltage_mean:FLOAT,reading_count:INTEGER"
)


def parse(msg):
    return json.loads(msg.decode("utf-8"))


def summarize(kv, win=beam.DoFn.WindowParam):
    machine_id, readings = kv
    readings = list(readings)
    n = len(readings)

    def col(k):
        return [x[k] for x in readings]

    temp, vib = col("temperature"), col("vibration")
    return {
        "machine_id": machine_id,
        "window_end": win.end.to_utc_datetime().isoformat(),
        "temp_mean": sum(temp) / n,
        "temp_max": max(temp),
        "vibration_mean": sum(vib) / n,
        "vibration_std": st.pstdev(vib) if n > 1 else 0.0,
        "rpm_mean": sum(col("rpm")) / n,
        "pressure_mean": sum(col("pressure")) / n,
        "voltage_mean": sum(col("voltage")) / n,
        "reading_count": n,          # data-quality signal (was the window complete?)
    }


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_id", required=True)   # NOT --project: that name belongs to Beam
    ap.add_argument("--subscription", required=True)
    ap.add_argument("--dataset", default="maintenance")
    args, beam_args = ap.parse_known_args()

    opts = PipelineOptions(
        beam_args,
        streaming=True,
        save_main_session=True,     # ships module-level imports (json, statistics) to the workers
        project=args.project_id,
    )

    with beam.Pipeline(options=opts) as p:
        readings = (
            p
            | "Read" >> beam.io.ReadFromPubSub(subscription=args.subscription)
            | "Parse" >> beam.Map(parse)
        )

        # Branch 1: raw rows straight to BigQuery
        _ = readings | "RawToBQ" >> beam.io.WriteToBigQuery(
            f"{args.project_id}:{args.dataset}.telemetry_raw",
            schema=RAW_SCHEMA,
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
            create_disposition=beam.io.BigQueryDisposition.CREATE_NEVER,
        )

        # Branch 2: 2-minute fixed windows per machine → aggregate → BigQuery
        _ = (
            readings
            | "KV" >> beam.Map(lambda r: (r["machine_id"], r))
            | "Window" >> beam.WindowInto(beam_window.FixedWindows(120))  # 120 s
            | "Group" >> beam.GroupByKey()
            | "Summarize" >> beam.Map(summarize)
            | "FeatToBQ" >> beam.io.WriteToBigQuery(
                f"{args.project_id}:{args.dataset}.features_windowed",
                schema=FEAT_SCHEMA,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_NEVER,
            )
        )


if __name__ == "__main__":
    run()
```

**`dataflow/requirements.txt`**:

```
apache-beam[gcp]==2.75.0
```

> **⚠️ Why 2.75 and not an older pin — the Apple Silicon wheel wall.** Apache Beam published **no macOS `arm64` wheels at all until 2.70.0**. Every release before that ships `macosx_..._x86_64` only. On an M-series Mac that means pip finds nothing compatible, silently falls back to **building Beam from source**, and the source build dies on `ModuleNotFoundError: No module named 'pkg_resources'` (setuptools 81+ stopped shipping `pkg_resources`; setuptools 83 removed it outright). The error never mentions wheels, architecture, or Python — so you chase setuptools for an hour. **Anything below Beam 2.70 is simply not installable on Apple Silicon without a source-build fight.** Take the wheel. If you ever need to check this for another package:
>
> ```bash
> pip install --only-binary=:all: 'apache-beam[gcp]==2.75.0'   # refuses to source-build; fails loudly instead of silently
> ```
>
> Intel Macs and Linux are unaffected, which is exactly why this bug survives in so many tutorials.

> **Three deliberate choices in that file, all learned the hard way:**
> 1. **`--project_id`, not `--project`.** Beam *itself* owns a `--project` pipeline option that Dataflow requires. If your argparse consumes `--project`, it never reaches Beam and the launch dies with `Missing required option: project`. Use a different name and pass it in via `PipelineOptions(project=...)`.
> 2. **`save_main_session=True`.** Your functions rely on module-level imports (`json`, `statistics`). Without this flag the pipeline runs **fine locally** and then fails **on Dataflow only** with `NameError: name 'json' is not defined` — the workers unpickle your functions without your imports. It is the single most classic Beam-on-Dataflow failure.
> 3. **Read a `subscription`, not a `topic`.** Reading a topic makes Dataflow create its own hidden subscription, which needs broad Pub/Sub admin rights on the worker SA. The Terraform-declared `telemetry-df-sub` keeps it least-privilege.

## Step 3.2 — Test locally with the DirectRunner first (free)

Always debug on your laptop before paying for Dataflow. The DirectRunner runs the same code locally, using your Application Default Credentials:

```bash
cd ~/dev/predictive-maint-adv
source .venv-beam/bin/activate                 # the BEAM venv, not the general one
python --version                               # MUST print Python 3.12.x — if not, see the venv gotcha in Conventions
pip install -r dataflow/requirements.txt       # want "Downloading apache_beam-2.75.0-cp312-...-arm64.whl", NOT a .tar.gz

export GRPC_VERBOSITY=ERROR                    # silences the macOS fork-noise flood — see below
export GRPC_ENABLE_FORK_SUPPORT=0

python dataflow/stream_features.py \
  --project_id predictive-maint-adv \
  --subscription projects/predictive-maint-adv/subscriptions/telemetry-df-sub \
  --runner DirectRunner
```

> **⚠️ This command never returns — by design.** It is a **streaming** pipeline (`streaming=True`) reading an **unbounded** Pub/Sub source. There is no end of input, so there is no exit. It blocks until **Ctrl-C**, forever, and that is the correct behaviour — a batch pipeline finishes, a streaming one does not. The same is true of the Dataflow job in Step 3.3, which sits in `Running` until you cancel it (Step 3.4). Do not sit waiting for a prompt that is never coming.
>
> **The wall of `ev_poll_posix.cc:593] FD from fork parent still in poll list` is not an error.** It is gRPC's C-core grumbling about file descriptors inherited across the `fork()` the DirectRunner uses to spawn local workers. Note the leading **`I`** in `I0712 11:54:01…` — that's an **I**NFO line, not an error. It is macOS-only, it is harmless, and it scrolls fast enough to look like a crash loop. The two `GRPC_*` exports above quiet it.

**So how do you know it's working?** Not by staring at that terminal. You'll have **three** open: the generator in one, the DirectRunner in another, and a third to check the actual output. In the third:

```bash
bq query --use_legacy_sql=false 'SELECT COUNT(*) AS n FROM maintenance.telemetry_raw'
```

`telemetry_raw` should start climbing within ~30 seconds — that's Branch 1, which writes every reading straight through. `features_windowed` is Branch 2 and cannot produce anything until a **full 120-second window closes** and flushes, so give it ~3 minutes before you judge it:

```bash
bq query --use_legacy_sql=false \
  'SELECT * FROM maintenance.features_windowed ORDER BY window_end DESC LIMIT 10'
```

Rows appearing means the pipeline, the schema, and the permissions all work. Ctrl-C to stop.

> The local run **acks** the messages it consumes, so those few hundred readings won't reach the Dataflow job. Harmless — the generator is still producing.

## Step 3.3 — Launch it on Dataflow (the real thing)

```bash
source .venv-beam/bin/activate
python dataflow/stream_features.py \
  --project_id predictive-maint-adv \
  --subscription projects/predictive-maint-adv/subscriptions/telemetry-df-sub \
  --runner DataflowRunner \
  --project predictive-maint-adv \
  --region europe-west3 \
  --worker_zone europe-west3-a \
  --worker_machine_type n1-standard-2 \
  --enable_streaming_engine \
  --temp_location gs://predictive-maint-adv-artifacts/dataflow/temp \
  --staging_location gs://predictive-maint-adv-artifacts/dataflow/staging \
  --service_account_email df-worker@predictive-maint-adv.iam.gserviceaccount.com \
  --max_num_workers 2 \
  --job_name maint-stream-features
```

(Yes, both `--project_id` — yours — and `--project` — Beam's — appear. That's the whole point of gotcha #1.)

> **⚠️ `ZONE_RESOURCE_POOL_EXHAUSTED` — the failure that is not your fault.** The launch succeeds, then the job flips to `FAILED` and your terminal prints only:
>
> ```
> DataflowRuntimeException: Dataflow pipeline failed. State: FAILED, Error:
> Workflow failed.
> ```
>
> That message is useless on purpose — **the real error is never in your terminal.** Go to **Dataflow → Jobs → your job → Logs → Diagnostics**. If it reads:
>
> ```
> Startup of the worker pool in europe-west3 failed to bring up any of the desired 1 workers.
> ZONE_RESOURCE_POOL_EXHAUSTED: ... zone 'europe-west3-c' does not have enough resources
> ```
>
> …then Google is simply **out of VMs in that zone**. Your code, your IAM, and your quota are all fine. Dataflow picks a zone for you, and it picked a full one. Three levers, in order:
>
> 1. **Pin a different zone.** `europe-west3` has `a`, `b`, `c` — if `--worker_zone europe-west3-a` is also exhausted, try `-b`.
> 2. **Shrink the machine.** Stockouts are **machine-type-specific**. Dataflow's streaming default is a 4-vCPU `n1-standard-4`; a 2-vCPU box is far easier to place. `--enable_streaming_engine` moves shuffle and windowing state off the worker and into the Dataflow service, which is what makes `n1-standard-2` genuinely sufficient here — both flags are in the command above for exactly this reason.
> 3. **Move region.** If all three zones are dry, use `--region europe-west1 --worker_zone europe-west1-b`. Costs you nothing: the BigQuery dataset and the GCS bucket are both **`EU` multi-region**, so they stay reachable with no data movement and no egress surprise.
>
> Knowing that a capacity stockout is a *cloud-provider* condition rather than a bug in your pipeline — and that the fix is zone/machine-shape, not code — is what stops you rewriting working code for an hour.

Open **Dataflow → Jobs**: a live graph of the pipeline with elements flowing through it — the clearest single view of the system actually working.

> **⚠️ Three warnings scroll past at launch. Two of them matter.** Beam logs everything at `WARNING`, which trains you to ignore all of it. Don't.
>
> **`GETTING_PUBSUB_SUBSCRIPTION_FAILED ... PERMISSION_DENIED ... User not authorized`** — a **real IAM gap**, and a subtle one. `roles/pubsub.subscriber` grants `pubsub.subscriptions.consume`, so pulling messages works and the job runs fine. It does **not** grant `pubsub.subscriptions.get`, so Dataflow cannot read the subscription's *configuration* to check its ack deadline. Today that costs you nothing. The day you tune the ack deadline, Dataflow mis-tunes its pull behaviour against a value it was never allowed to read, and you debug phantom throughput problems. The Terraform above already includes **`roles/pubsub.viewer`** for this reason. If you provisioned before that line existed, patch it live — no job restart needed, the next poll picks it up:
>
> ```bash
> gcloud pubsub subscriptions add-iam-policy-binding telemetry-df-sub \
>   --member="serviceAccount:df-worker@predictive-maint-adv.iam.gserviceaccount.com" \
>   --role="roles/pubsub.viewer" --project predictive-maint-adv
> ```
>
> The lesson generalises: **"the job runs" is not the same as "the permissions are right."** A missing read permission on a *config* surface fails as a warning, not an error.
>
> **`Bucket ... has soft-delete policy enabled`** — this one costs **money**. GCS defaults to a 7-day soft-delete retention, and Dataflow creates and deletes temp objects continuously, so you pay storage on every deleted object for a week. The Terraform sets `retention_duration_seconds = 0`. On an already-created bucket:
>
> ```bash
> gcloud storage buckets update gs://predictive-maint-adv-artifacts --clear-soft-delete
> ```
>
> **`Streaming job has set up its own fixed sharding configuration. Liquid sharding will be disabled.`** — genuinely benign. `WriteToBigQuery` fixes its own sharding, so Dataflow is telling you it won't auto-tune parallelism. Ignore this one.

> **⚠️ What "stuck" looks like (it isn't) — the launch command never returns.** Two separate things are slow here, and neither is broken.
>
> **1. Your terminal blocks forever.** `with beam.Pipeline(options=opts) as p:` calls `wait_until_finish()` when the block exits. For a **streaming** job, "finish" never arrives — the job is designed to run until you stop it. So the command hangs, indefinitely, printing job-state updates. This is also why a failed job surfaces in your terminal as `DataflowRuntimeException` rather than silence: `wait_until_finish()` raises on a terminal `FAILED` state.
>
> **Ctrl-C is safe and does *not* cancel the job.** The pipeline was already submitted to Dataflow; it runs in the cloud, independent of your laptop. Killing the local process just detaches your terminal. That cuts both ways — it is precisely why you must explicitly `drain` the job in Step 3.4, or it bills all night while your laptop is closed.
>
> **2. The first rows take 3–6 minutes.** Dataflow has to provision a worker VM and pull its container before a single element is processed. A flat row count for five minutes is normal.
>
> Don't judge the job from the terminal. Judge it from its state and its output:
>
> ```bash
> gcloud dataflow jobs list --region europe-west3 --status active   # want JOB_STATE_RUNNING
> bq query --use_legacy_sql=false 'SELECT COUNT(*) AS n FROM maintenance.telemetry_raw'
> ```
>
> `RUNNING` + a climbing `telemetry_raw` = done. Ctrl-C the terminal and move on.

## Step 3.4 — Verify, then learn the off-switch

```bash
bq query --use_legacy_sql=false \
  'SELECT COUNT(*) AS windows, COUNT(DISTINCT machine_id) AS machines,
          MIN(reading_count) AS min_readings, MAX(reading_count) AS max_readings
   FROM maintenance.features_windowed'
```

You should see ~60 machines and `reading_count` sitting at **120** (one reading per second × 120 seconds). Hold that thought — it matters in Phase 4.

```bash
# OFF-SWITCH #1 — run this at the end of EVERY session:
gcloud dataflow jobs list --region europe-west3 --status active
gcloud dataflow jobs drain <JOB_ID> --region europe-west3
gcloud dataflow jobs list --region europe-west3 --status active   # re-run after ~1 min: want an EMPTY list
```

Then **Ctrl-C the generator** in its terminal. It costs nothing, but left running it keeps filling BigQuery and quietly shifts the baseline you'll measure drift against in Phase 8.

> **Drain, don't cancel.** `drain` stops pulling new messages and lets in-flight windows finish writing; `cancel` drops them. Either way the job stops billing. A streaming Dataflow job holds at least one worker VM the entire time it exists — it does **not** scale to zero.
>
> **Ctrl-C on the launch terminal does NOT stop the job.** This is the expensive misunderstanding. The pipeline was submitted to Dataflow and runs *in the cloud*; killing your local Beam process only detaches your terminal. The job keeps billing with your laptop shut. `drain` is the only off-switch — which is why the `jobs list` above is run **twice**, before and after, so you *see* it disappear rather than assume it did.
>
> **If you fell back to `europe-west1`** because of the zone stockout in Step 3.3, change `--region` in *all three* commands to match — `jobs list` is region-scoped, so pointing it at the wrong region returns an empty list and you will happily walk away from a job that is still billing, having "confirmed" it was stopped.

✅ **Checkpoint 3:** the Dataflow job is running, `telemetry_raw` fills continuously, and `features_windowed` gets a fresh batch of ~60 rows every ~2 minutes.

---

# Phase 4 — Honest labels & a training view (~1.5 hrs)

**Goal:** A predictive-maintenance model needs a *label*: "did this machine fail soon after this window?" Your stream doesn't know the future, so you compute the label retroactively in BigQuery — **from the ground-truth failure log**, not from the sensor readings.

> **Term — the labeling horizon:** features look *backward* (what did the last 2 minutes look like?); the label looks *forward* (did this machine die within the next H minutes?). H is a business decision — in a real plant it's however long it takes to schedule an intervention, typically 24–72 hours. Here machine lifecycles are compressed to ~43 minutes, so **H = 10 minutes** is the proportionate equivalent. At that horizon you get ~20 % positives, which is a healthy imbalance for this problem.

## Step 4.1 — Label from ground truth

> **⚠️ The circular-label trap — the mistake this phase exists to avoid.** The obvious shortcut is to *define* failure as "vibration crossed some threshold", then label the windows before it. **Don't.** `vibration_mean` is one of your model's input features. Defining the target by thresholding an input makes the target a near-deterministic function of that input — the model then learns "is vibration close to the threshold?", which is a tautology dressed up as a prediction. There is a second problem too: the threshold you pick almost never corresponds to when the machine *actually* died (a vibration of 1.8 in this simulator means ~20 % health remaining — the machine has not failed, it's just sick).
>
> Label from **`failure_events`** instead. That table is written by the machine's own failure, is independent of every model input, and is exactly what a real maintenance log gives you.

Create **`sql/label.sql`** (a file — you'll run it here, and Phase 10's retrainer will run the *same file* automatically):

```bash
mkdir -p ~/dev/predictive-maint-adv/sql
```

```sql
-- sql/label.sql
-- HORIZON is the ONE number that defines this project's prediction problem.
-- Change it here and nowhere else.
DECLARE horizon_minutes INT64 DEFAULT 10;

CREATE OR REPLACE TABLE `predictive-maint-adv.maintenance.features_labeled` AS
SELECT
  f.*,
  CAST(EXISTS(
    SELECT 1
    FROM `predictive-maint-adv.maintenance.failure_events` e
    WHERE e.machine_id = f.machine_id
      AND e.failed_at >  f.window_end                                            -- strictly in the FUTURE
      AND e.failed_at <= TIMESTAMP_ADD(f.window_end, INTERVAL horizon_minutes MINUTE)
  ) AS INT64) AS will_fail
FROM `predictive-maint-adv.maintenance.features_windowed` f
-- LABEL CENSORING: a window whose horizon has not fully elapsed yet would be
-- labeled 0 purely because the future has not happened. Those rows are not
-- "negatives", they are UNKNOWN — training on them teaches the model that the
-- most recent, most-degraded windows are safe. Drop them.
WHERE f.window_end <= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL horizon_minutes MINUTE);
```

Run it:

```bash
cd ~/dev/predictive-maint-adv
bq query --use_legacy_sql=false < sql/label.sql
```

## Step 4.2 — Check the label balance (do not skip this)

```sql
SELECT
  will_fail,
  COUNT(*) AS n,
  ROUND(COUNT(*) / SUM(COUNT(*)) OVER (), 3) AS share
FROM `predictive-maint-adv.maintenance.features_labeled`
GROUP BY will_fail;
```

**Expect roughly 15–25 % positives.** If you get:
- **0 % positives** — `failure_events` is empty. Is the generator running? Did it print `FAILED` lines?
- **Under 5 %** — you haven't collected enough data yet. Let the generator + Dataflow run longer (Phase 2's table says 1–2 hours).
- **Over 60 %** — your `DECAY` is too high, so machines are dying constantly and almost every window precedes a failure. Restore `DECAY=0.0001`.

## Step 4.3 — The training view

```sql
CREATE OR REPLACE VIEW `predictive-maint-adv.maintenance.v_training` AS
SELECT
  machine_id,          -- NOT a feature. Carried ONLY so training can split by machine.
  temp_mean, temp_max, vibration_mean, vibration_std,
  rpm_mean, pressure_mean, voltage_mean,
  will_fail AS label
FROM `predictive-maint-adv.maintenance.features_labeled`;
```

> **⚠️ Why `reading_count` is NOT in the training view — a training-serving skew you would never see coming.** It looks like a feature. It isn't. `reading_count` is *120 in every single row* (you verified this in Step 3.4) because it just counts `120 seconds ÷ SLEEP=1.0`. It carries **zero information about machine health** — what it actually encodes is **the ingestion rate of your pipeline**. Train on it and nothing bad happens... until the day the generator runs at a different speed, or a worker hiccups and a window lands with 90 readings instead of 120. Then a column the model has learned to rely on shifts under it, at serving time, for reasons that have nothing to do with any machine. Keep it in `features_windowed` as a data-quality signal ("was this window complete?"), and keep it out of the model. Understanding *why* it has to be dropped is worth more than any hyperparameter you could tune.

## Step 4.4 — Schedule the labeling

> **⚠️ Set the editor's location BEFORE you open the schedule dialog.** A scheduled query inherits its processing location from the query editor, and it is **fixed at creation** — you cannot change it afterwards. Your `maintenance` dataset is in the **`EU` multi-region** (Phase 1 Terraform), which is a *different location* from `europe-west3`, even though Frankfurt sits physically inside the EU. A query pinned to `europe-west3` cannot see an `EU` dataset, and the dialog rejects it with a confusing `Not found: Dataset predictive-maint-adv:maintenance` — which reads like the dataset doesn't exist when in fact you're simply looking for it in the wrong place.
>
> Confirm where it actually lives, then match the editor to it:
>
> ```bash
> bq show --format=prettyjson predictive-maint-adv:maintenance | grep -i '"location"'
> ```
>
> In the query editor: **More → Query settings → Additional settings → Data location** → set to `EU` → **Save**. *Then* open the schedule dialog.

Paste the contents of `sql/label.sql` into the editor, then **Schedule → Create new scheduled query**, and fill the dialog in:

| Field | Value |
|-------|-------|
| Name for scheduled query | `rebuild_labels` |
| Repeat frequency | **Minutes** / **30** (BigQuery's floor is 15 min) |
| Start | **Start now** |
| End | **End never** |
| Set a destination table for query results | **leave unchecked** |

> **Why no destination table.** `label.sql` is a *script* — it opens with `DECLARE` and writes `features_labeled` itself via `CREATE OR REPLACE TABLE`. The destination-table field exists for bare `SELECT` queries that need somewhere to put their output; setting it here conflicts with the script's own write. The same fact explains the `NOTE: Could not compute bytes processed estimate for script` banner at the bottom of the editor — BigQuery can't dry-run a script's cost because control flow means it doesn't know which statements will execute. Both are expected. Neither is an error.

Prefer the CLI, or need to force the location explicitly? This does the whole thing in one command:

```bash
cd ~/dev/predictive-maint-adv
bq mk --transfer_config \
  --project_id=predictive-maint-adv \
  --data_source=scheduled_query \
  --display_name=rebuild_labels \
  --target_dataset=maintenance \
  --location=EU \
  --schedule='every 30 minutes' \
  --params="$(jq -Rs '{query: .}' sql/label.sql)"
```

(`--target_dataset` is required by the API but ignored for scripts like this one, which name their own destination.)

> **Gotcha:** scheduled queries run on the **BigQuery Data Transfer API**. If the console refuses with `BigQuery Data Transfer API has not been used in project…`, that API isn't on — it's in the Phase 1 Terraform (`bigquerydatatransfer.googleapis.com`), so `terraform apply` and retry.

**Verify it actually runs.** A saved-but-never-executed schedule is the most common false pass here: the config saves fine even when the transfer service can't execute it, and the failure only surfaces on the first fire. Console → **BigQuery → Scheduled queries → `rebuild_labels` → Run history**. You want one run marked **Succeeded**. Then re-run the freshness query below and confirm `newest` has moved forward — that's the proof it *rebuilds*, not merely that it exists.

## Step 4.5 — Verify the labels are *correct*, not merely present

Four checks. "The query ran" is not one of them.

**1. The table is fresh and censored.**

```sql
SELECT
  COUNT(*) AS rows,
  COUNT(DISTINCT machine_id) AS machines,
  MIN(window_end) AS oldest,
  MAX(window_end) AS newest,
  TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(window_end), MINUTE) AS lag_min
FROM `predictive-maint-adv.maintenance.features_labeled`;
```

`lag_min` must be **≥ 10**. That gap *is* the censoring `WHERE` clause doing its job. If it's 0–2, the filter didn't apply and your most recent — most degraded — windows are all sitting there mislabeled `0`.

**2. Label balance** — the Step 4.2 query. Both classes present, positives at **0.15–0.25**.

**3. The label is learnable.** The check that catches a labeling bug the balance query cannot:

```sql
SELECT
  label,
  COUNT(*) AS n,
  ROUND(AVG(temp_mean), 2)      AS temp,
  ROUND(AVG(vibration_mean), 3) AS vib,
  ROUND(AVG(rpm_mean), 0)       AS rpm,
  ROUND(AVG(voltage_mean), 3)   AS voltage
FROM `predictive-maint-adv.maintenance.v_training`
GROUP BY label;
```

`label = 1` rows should show visibly **higher** `temp` and `vib` and **lower** `rpm` than `label = 0`. `voltage` should be near-identical across both — it's the noise distractor, and if it *does* separate the classes, something is wrong. If the two rows look the same on every column, your labels are noise: Phase 5 will hand you an AUC around 0.5 and no amount of hyperparameter tuning will move it. Fix it here, not there.

**4. The schedule has actually fired** — Run history shows **Succeeded** (Step 4.4).

✅ **Checkpoint 4:** `v_training` returns rows; `label` is a mix of 0s and 1s at roughly 15–25 % positive; `lag_min ≥ 10`; the two label classes separate on `temp_mean` and `vibration_mean`; and `rebuild_labels` has one successful run in its history.

---

# Phase 5 — A baseline model in minutes with BigQuery ML (~1 hr)

**Goal:** Before touching Vertex training, get a *working model* out of one SQL statement — the fastest possible baseline, and a legitimate data-role skill on its own.

> **Term — BigQuery ML (BQML):** train and run ML models with `CREATE MODEL` statements directly on BigQuery tables. No data movement, no infrastructure.

## Step 5.1 — Train

```sql
CREATE OR REPLACE MODEL `predictive-maint-adv.maintenance.bqml_failure`
OPTIONS(
  model_type = 'BOOSTED_TREE_CLASSIFIER',
  input_label_cols = ['label'],
  auto_class_weights = TRUE          -- handles the ~20% class imbalance for you
) AS
SELECT * EXCEPT(machine_id)          -- machine_id is an ID, not a feature (see below)
FROM `predictive-maint-adv.maintenance.v_training`;
```

> **`EXCEPT(machine_id)` is mandatory.** BQML treats **every** non-label column as a feature. Leave the ID in and the model happily memorizes "machine M017 tends to fail" instead of learning what failure *looks like* — and it will be useless on any machine it hasn't seen. This is the single most common BQML footgun.

## Step 5.2 — Evaluate

```sql
SELECT * FROM ML.EVALUATE(MODEL `predictive-maint-adv.maintenance.bqml_failure`);
```

Note the `roc_auc` — that's the number your Vertex model in Phase 6 has to beat, and it's what makes your final claim ("my custom model beat a BQML boosted-tree baseline by X") mean something. A number with nothing to compare it to is not a result.

```sql
SELECT * FROM ML.FEATURE_IMPORTANCE(MODEL `predictive-maint-adv.maintenance.bqml_failure`)
ORDER BY importance_gain DESC;
```

`temp_mean` and `vibration_mean` should dominate; `voltage_mean` should sit near zero — it's pure sensor noise in the simulator, deliberately included as a **distractor**. A model that ranks it highly is telling you something is wrong.

## Step 5.3 — Batch-score the fleet into `predictions`

```sql
INSERT INTO `predictive-maint-adv.maintenance.predictions`
  (machine_id, scored_at, failure_prob, model_version)
SELECT
  machine_id,
  CURRENT_TIMESTAMP() AS scored_at,
  (SELECT p.prob FROM UNNEST(predicted_label_probs) p WHERE p.label = 1) AS failure_prob,
  'bqml_failure' AS model_version
FROM ML.PREDICT(
  MODEL `predictive-maint-adv.maintenance.bqml_failure`,
  -- The LATEST window per machine, straight from the LIVE feature table.
  -- features_windowed, not features_labeled: scoring needs the freshest data
  -- and does not need a label at all (and features_labeled deliberately lags
  -- by one horizon — see the censoring note in Phase 4).
  (SELECT * EXCEPT(rn) FROM (
     SELECT *, ROW_NUMBER() OVER (PARTITION BY machine_id ORDER BY window_end DESC) AS rn
     FROM `predictive-maint-adv.maintenance.features_windowed`
   ) WHERE rn = 1)
);

SELECT machine_id, ROUND(failure_prob, 3) AS risk
FROM `predictive-maint-adv.maintenance.predictions`
WHERE model_version = 'bqml_failure'
ORDER BY scored_at DESC, risk DESC
LIMIT 10;
```

> **Two details in that query that save you a debugging session.** (1) `predicted_label_probs` is an array of `(label, prob)` structs in **no guaranteed order** — `[OFFSET(0)]` might be the probability of class **0**. Always select `WHERE p.label = 1`. (2) BQML **passes unrecognized columns through** to the output, which is exactly how `machine_id` and `window_end` survive `ML.PREDICT` even though the model never trained on them.

> **Why everything scores into one `predictions` table.** The BQML batch path and the endpoint path (Phase 8) both write here, tagged by `model_version`. That gives you one place to answer "what is the fleet's risk right now?", one dashboard source, and — because the tag records *which model produced each score* — the ability to compare prediction distributions across model versions after a retrain. It also means your Phase 12 dashboard has data in it from Phase 5 onward, instead of waiting on the handful of rows a manual `curl` produces.

Schedule this: **BigQuery → Scheduled queries**, every **15 minutes**, name `batch_score_fleet`. Same location rule as Step 4.4 — set the editor's **Data location** to `EU` *before* opening the schedule dialog, or you'll get the same misleading `Not found: Dataset` error. Leave the destination table unchecked here too: the `INSERT INTO` names its own target.

✅ **Checkpoint 5:** `ML.EVALUATE` prints a healthy `roc_auc`, and `predictions` fills with a risk score per machine every 15 minutes. **You now have a complete, working predictive-maintenance system.** Everything below is the MLOps that makes it production-grade.

---

# MLOPS PHASES

# Phase 6 — Custom training on Vertex AI + Model Registry (~3 hrs)

**Goal:** Move from "SQL model in the warehouse" to a **portable, versioned model artifact** trained by Vertex AI and catalogued in the Model Registry — the object every later phase deploys, evaluates, and retrains.

> **Term — Model Registry:** a versioned catalogue of trained models. Each training run produces a new *version*; deployments and rollbacks reference versions. It's git-for-models.

> **⚠️ The serving-container decision (read this before writing code).** You'll deploy behind Google's **prebuilt sklearn prediction container**, which calls exactly one method on your saved model: **`predict()`**. For a *classifier*, `predict()` returns hard 0/1 labels — but the product needs a *probability* ("this machine is 87 % likely to fail"). `predict_proba()` is **not** exposed by the prebuilt container. The pragmatic fix used here: train a **`GradientBoostingRegressor` on the 0/1 label**, so `predict()` itself returns a continuous risk score. AUC and recall are computed from that score directly, so evaluation is unaffected. The score can overshoot slightly outside [0, 1] (measured: −0.03 to +1.05), so the scorer clamps it. The "proper" production alternative is a Custom Prediction Routine or your own serving container — name that in the README's "what I'd do next."

## Step 6.1 — The training script

```bash
mkdir -p ~/dev/predictive-maint-adv/training
```

**`training/train.py`**:

```python
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
# has never met? Any random split of a time series over the same entities
# measures memorisation, not generalisation.
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
```

**`training/requirements.txt`**:

```
google-cloud-bigquery==3.25.0
google-cloud-bigquery-storage==2.27.0
google-cloud-storage==2.18.0
scikit-learn==1.5.2
numpy==1.26.4
pandas==2.2.2
db-dtypes==1.3.0
joblib==1.4.2
```

> **⚠️ `.to_dataframe()` needs a permission you would never guess from the code.** `google-cloud-bigquery-storage` is in that requirements file for a reason: when it's installed, `.to_dataframe()` stops pulling rows through the ordinary query API and quietly switches to the **BigQuery Storage Read API**, which is far faster — and governed by a *completely separate* IAM permission, `bigquery.readsessions.create`. So the pipeline SA can hold `bigquery.dataEditor` **and** `bigquery.jobUser`, execute the query without complaint, and then die on the way back with:
>
> ```
> PERMISSION_DENIED: request failed: the user does not have
> 'bigquery.readsessions.create' permission for 'projects/predictive-maint-adv'
> ```
>
> Surfaced through the SDK, this arrives as the near-useless `The replica workerpool0-0 exited with a non-zero status of 1` — the real traceback is only in Cloud Logging. `roles/bigquery.readSessionUser` (in the Phase 1 Terraform) is the fix. If you skipped it, add it and re-run — **no image rebuild needed**, the container was always fine.
>
> The escape hatch, if you ever hit this somewhere you can't grant IAM: `.to_dataframe(create_bqstorage_client=False)` forces the REST path. Slower, but it needs nothing beyond `jobUser`.

> **Reading the real error from a failed Vertex job.** The Python SDK only ever tells you the exit code. Every training-job failure in this project is diagnosed the same way — take the `job_id` from the SDK's `View backing custom job` URL and:
>
> ```bash
> gcloud logging read \
>   'resource.type="ml_job" AND resource.labels.job_id="YOUR_JOB_ID"' \
>   --project=predictive-maint-adv --limit=100 \
>   --format='value(textPayload)' --order=asc
> ```
>
> Scroll past the wall of `Vertex AI is provisioning job running framework` lines; your container's stdout and its traceback are at the bottom. Learn this command now — you'll use it again in Phases 7 and 10.

> **Why `scikit-learn` and `numpy` are pinned, and pinned *low*.** The model you train here is a joblib pickle that gets **unpickled inside Google's prebuilt serving container**. Two independent version contracts have to hold:
> - **scikit-learn**: the `sklearn-cpu.1-5` image ships 1.5.x, so you train on `1.5.2`. A pickle from 1.6 loading into 1.5 is a coin flip, and when it loses, the endpoint deploy fails with an unpickling traceback that looks like a permissions problem.
> - **numpy**: pinned to `<2` on purpose. numpy 2 can read numpy 1 pickles, but **numpy 1 cannot reliably read numpy 2 pickles** — so if the serving container is still on numpy 1.x and you train on numpy 2.x, the model loads on your laptop and dies on the endpoint. Pinning low is the safe direction of the asymmetry.

## Step 6.2 — Smoke-test it locally first (free, and catches 90 % of the failures)

Do not pay for a 15-minute Vertex job to discover a dtype error.

```bash
cd ~/dev/predictive-maint-adv
source .venv/bin/activate
pip install -r training/requirements.txt
PROJECT=predictive-maint-adv python training/train.py
```

You want to see the machine-disjoint split, then the `eval:` line. **Expect `auc` around 0.93–0.96** and `pr_auc` around 0.85–0.90 with a couple of hours of data. If AUC comes back near 0.5, your labels are broken (go back to Step 4.2); if it comes back at 0.999, something is leaking (check that you dropped `reading_count` and `machine_id`).

> **`recall_at_0.5` will be lower than `recall_at_tuned` — that's expected, not a bug.** You trained a *regressor* on a 0/1 target: its output is a risk score, not a calibrated probability, and 0.5 has no special meaning to it. The tuned threshold (flag the riskiest windows at the base failure rate) is the honest operating point. Report both; explain the difference. That single paragraph in your README does more for you than another 0.01 of AUC.

## Step 6.3 — Build the training container (server-side, via Cloud Build)

**`training/Dockerfile`**:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY train.py .
ENTRYPOINT ["python", "train.py"]
```

```bash
cd ~/dev/predictive-maint-adv/training

# One-time: create the Docker repo in Artifact Registry
gcloud artifacts repositories create maint --repository-format=docker \
  --location=europe-west3 --description="predictive maint images"

REPO=europe-west3-docker.pkg.dev/predictive-maint-adv/maint
gcloud builds submit --tag $REPO/trainer:latest .
```

> **⚠️ Build with Cloud Build, never `docker build`, on Apple Silicon.** Your Mac is arm64; Vertex AI training runs on amd64. A local build produces an arm64 image and the training job dies with `exec format error` — an error that tells you nothing about architecture. `gcloud builds submit` builds on Google's x86 machines, so the mismatch cannot happen. This is also why Docker Desktop is only listed as *optional* in Phase 0.

> **Gotcha — Cloud Build push permission.** If the build fails at the **push** step with `denied: permission`, grant the writer role to the Compute Engine default SA (which Cloud Build runs as) and retry:
> ```bash
> PN=$(gcloud projects describe predictive-maint-adv --format='value(projectNumber)')
> gcloud projects add-iam-policy-binding predictive-maint-adv \
>   --member="serviceAccount:${PN}-compute@developer.gserviceaccount.com" \
>   --role="roles/artifactregistry.writer"
> ```

## Step 6.4 — Run a custom training job → register the model

**`training/run_training.py`**:

```python
from google.cloud import aiplatform

PROJECT, REGION = "predictive-maint-adv", "europe-west3"
BUCKET = "gs://predictive-maint-adv-artifacts"
REPO = "europe-west3-docker.pkg.dev/predictive-maint-adv/maint"

# Confirm the CURRENT prebuilt sklearn prediction image tag in Google's
# "Pre-built containers for prediction" docs — the version suffix moves, and it
# MUST match the scikit-learn pin in training/requirements.txt.
SERVING_IMAGE = "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"

aiplatform.init(project=PROJECT, location=REGION, staging_bucket=BUCKET)

job = aiplatform.CustomContainerTrainingJob(
    display_name="maint-trainer",
    container_uri=f"{REPO}/trainer:latest",
    model_serving_container_image_uri=SERVING_IMAGE,
)

model = job.run(
    model_display_name="maint-failure",
    environment_variables={"PROJECT": PROJECT},   # train.py reads this — without it: KeyError
    replica_count=1,
    machine_type="n1-standard-4",
    service_account="vertex-pipeline@predictive-maint-adv.iam.gserviceaccount.com",
)
print("Registered model:", model.resource_name, "version:", model.version_id)
```

Before running it, let the pipeline SA **act as itself** — `job.run(service_account=...)` requires the caller to have `actAs` on that SA, and in Phases 7 and 10 the *caller* will be the pipeline SA:

```bash
gcloud iam service-accounts add-iam-policy-binding \
  vertex-pipeline@predictive-maint-adv.iam.gserviceaccount.com \
  --member="serviceAccount:vertex-pipeline@predictive-maint-adv.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

(As project owner *you* already have `actAs`. This binding is for later, when the pipeline submits jobs on its own behalf.)

```bash
cd ~/dev/predictive-maint-adv
source .venv/bin/activate
# The [pipelines] extra pulls in PyYAML, which aiplatform.PipelineJob needs to
# read a compiled pipeline spec (Phase 7) — even when that spec is a .json file.
# You don't need it for THIS step, but installing the extra now means your local
# venv and the retrainer container (Phase 10) agree, instead of diverging into
# "works on my Mac, exits 1 in Cloud Run".
pip install "google-cloud-aiplatform[pipelines]==1.71.0"
python training/run_training.py
```

~10–15 minutes (VM spin-up + training + model upload). Watch **Vertex AI → Training**.

> **Two gotchas this step exists to catch:**
> - The prebuilt sklearn container looks for a file literally named **`model.joblib`** at the model dir root. "Model not found" at deploy time is almost always this filename.
> - Passing `service_account=` explicitly matters. Without it the job runs as the **Compute Engine default SA**, whose permissions vary by org policy. Explicit SA = reproducible least privilege — and it's *why* `vertex-pipeline` needed `bigquery.jobUser` back in Phase 1 (`train.py` runs a query), `bigquery.readSessionUser` (`train.py` reads the result with `.to_dataframe()`), `artifactregistry.reader` (Vertex must pull your image), and `logging.logWriter` (the worker writes its own stdout). All four failures land on the *worker*, not the caller — so they show up as an opaque exit-1, not as a permission error at submit time. And the fourth one **hides the other three**: with no logWriter there is no log, so a `jobUser` or image-pull failure arrives as a bare exit-1 with an empty log stream. Grant logWriter first; it is what makes every other failure legible.

✅ **Checkpoint 6:** **Vertex AI → Model Registry** shows a `maint-failure` model with a version; the training log printed the `eval:` JSON; and `metrics.json` sits next to `model.joblib` in the model's artifact folder in GCS.

---

# Phase 7 — Orchestrate it as a Vertex AI Pipeline (~4 hrs)

**Goal:** Wrap train → evaluate → **conditional deploy** into a single **KFP v2 pipeline** that Vertex runs on demand or on a trigger. This is the MLOps centrepiece: a repeatable, parameterized, visualized DAG that only ships a model if it clears an evaluation gate.

> **Term — pipeline (KFP / Kubeflow):** a DAG of containerized steps ("components"). Vertex AI Pipelines runs them, caches unchanged steps, and draws a visual graph of every run. The KFP **v2** SDK is current; v1 syntax will not compile.

## Step 7.1 — Install the tooling

```bash
cd ~/dev/predictive-maint-adv && source .venv/bin/activate
pip install "kfp>=2.7,<3" "google-cloud-aiplatform>=1.71"
mkdir -p pipeline
```

## Step 7.2 — Define the pipeline

**`pipeline/maint_pipeline.py`**:

```python
from kfp import dsl, compiler


@dsl.component(base_image="python:3.12-slim",
               packages_to_install=["google-cloud-aiplatform==1.71.0"])
def train_and_register(project: str, region: str, staging_bucket: str,
                       trainer_image: str, serving_image: str,
                       pipeline_sa: str) -> str:
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=region,
                    staging_bucket=staging_bucket)      # REQUIRED for training jobs
    job = aiplatform.CustomContainerTrainingJob(
        display_name="maint-trainer-pipeline",
        container_uri=trainer_image,
        model_serving_container_image_uri=serving_image,
    )
    model = job.run(
        model_display_name="maint-failure",
        environment_variables={"PROJECT": project},
        replica_count=1,
        machine_type="n1-standard-4",
        service_account=pipeline_sa,
    )
    return model.resource_name


@dsl.component(base_image="python:3.12-slim",
               packages_to_install=["google-cloud-aiplatform==1.71.0",
                                    "google-cloud-storage==2.18.0"])
def evaluate_model(project: str, region: str, model_resource: str) -> float:
    """Reads the metrics.json that train.py uploaded next to model.joblib."""
    import json
    from google.cloud import aiplatform, storage
    aiplatform.init(project=project, location=region)
    uri = aiplatform.Model(model_resource).uri              # gs://.../model
    bucket, prefix = uri[5:].split("/", 1)
    blob = storage.Client().bucket(bucket).blob(f"{prefix.rstrip('/')}/metrics.json")
    metrics = json.loads(blob.download_as_text())
    print("metrics:", metrics)
    return float(metrics["auc"])


@dsl.component(base_image="python:3.12-slim",
               packages_to_install=["google-cloud-aiplatform==1.71.0"])
def deploy_model(project: str, region: str, model_resource: str, endpoint_name: str):
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=region)

    model = aiplatform.Model(model_resource)
    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
    endpoint = endpoints[0] if endpoints else aiplatform.Endpoint.create(
        display_name=endpoint_name)

    # Capture what is ALREADY on the endpoint, BEFORE we add to it.
    previous = [dm.id for dm in endpoint.list_models()]
    print("already deployed:", previous)

    model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-2",
        traffic_percentage=100,          # the new model takes ALL traffic...
        min_replica_count=1,
        max_replica_count=1,
    )
    print("deployed:", model_resource)

    # ...but traffic_percentage=100 only routes 0% of traffic to the OLD models.
    # It does NOT undeploy them. Each one keeps its own node, and each node keeps
    # BILLING — so after five retrains you are quietly paying for five idle
    # machines serving nobody. Undeploy the superseded ones now.
    for dm_id in previous:
        print("undeploying superseded model:", dm_id)
        endpoint.undeploy(deployed_model_id=dm_id)


@dsl.pipeline(name="maint-train-deploy")
def maint_pipeline(project: str, region: str, staging_bucket: str,
                   trainer_image: str, serving_image: str, pipeline_sa: str,
                   endpoint_name: str = "maint-endpoint",
                   min_auc: float = 0.80):
    t = train_and_register(project=project, region=region,
                           staging_bucket=staging_bucket,
                           trainer_image=trainer_image,
                           serving_image=serving_image,
                           pipeline_sa=pipeline_sa)
    e = evaluate_model(project=project, region=region, model_resource=t.output)
    with dsl.If(e.output >= min_auc, name="good-enough"):
        deploy_model(project=project, region=region,
                     model_resource=t.output, endpoint_name=endpoint_name)


if __name__ == "__main__":
    compiler.Compiler().compile(maint_pipeline, "maint_pipeline.json")
    print("compiled → maint_pipeline.json")
```

> **The `dsl.If` gate is the whole point of this phase.** "Only deploy if the measured AUC clears the bar" is *conditional deployment* — the guardrail that stops a bad retrain from silently replacing a good model. And the evaluate step is **real**: it reads the metrics the training run actually produced. `min_auc = 0.80` is a deliberately loose gate given that a healthy run scores ~0.95 — it's there to catch catastrophe (broken labels, empty data), not to micro-tune. In a fuller production build you'd evaluate on a **holdout dataset the pipeline controls**, rather than the training job's own test split, and you'd compare against the *currently deployed* model rather than a fixed constant. Both limitations are worth naming explicitly in the README rather than leaving implicit.

> **Debugging KFP components — the trap that eats an hour.** A `@dsl.component`'s `print()` output is **buffered**, and is often lost entirely when the container exits on an exception. Cloud Logging will show you the step "exited with non-zero status of 1" with *no traceback whatsoever*. Don't trust the empty log: reproduce the failing call locally with the same SDK version, where the real error appears instantly.
>
> **First, though, distinguish "unhelpful log" from "no log at all."** Pull the training worker's own log stream:
>
> ```bash
> gcloud logging read \
>   'resource.type="ml_job" AND resource.labels.job_id="<JOB_ID>"' \
>   --project predictive-maint-adv --limit 100 --order desc \
>   --format='table(severity, textPayload, jsonPayload.message)'
> ```
>
> (`<JOB_ID>` is in the failed node's error message. Note the `jsonPayload.message` column — container stdout lands there, *not* in `textPayload`. Formatting with `--format='value(textPayload)'` alone silently prints blank lines for every line you actually care about.)
>
> If the *only* entries are Vertex's own lifecycle messages ("Job is running.", "The replica workerpool0-0 exited...") and **not one line came from your container** — no `train:` counts, no `eval:` JSON, no traceback — that is not a Python bug. Your service account cannot write logs. Check that `roles/logging.logWriter` is on `vertex-pipeline` (Phase 1); without it the worker's stdout is dropped on the floor and *every* crash looks identical. Fix that first, re-run, and the real error will be waiting for you.
>
> Diagnosis order that actually converges: (1) is the log empty → logWriter; (2) does `PROJECT=… python training/train.py` run clean locally → if yes it's environmental, not code; (3) is the image `amd64` (`docker image inspect --format '{{.Architecture}}'`) → an `arm64` image built on an M-series Mac dies before Python starts and also produces no output.

## Step 7.3 — Compile and run

```bash
cd ~/dev/predictive-maint-adv/pipeline
python maint_pipeline.py     # → maint_pipeline.json
```

**`pipeline/run_pipeline.py`**:

```python
from google.cloud import aiplatform

PROJECT, REGION = "predictive-maint-adv", "europe-west3"
BUCKET = "gs://predictive-maint-adv-artifacts"
REPO = "europe-west3-docker.pkg.dev/predictive-maint-adv/maint"
PIPELINE_SA = "vertex-pipeline@predictive-maint-adv.iam.gserviceaccount.com"
SERVING_IMAGE = "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"

aiplatform.init(project=PROJECT, location=REGION, staging_bucket=BUCKET)

job = aiplatform.PipelineJob(
    display_name="maint-train-deploy",
    template_path="maint_pipeline.json",
    pipeline_root=f"{BUCKET}/pipeline-root",
    parameter_values={
        "project": PROJECT,
        "region": REGION,
        "staging_bucket": BUCKET,
        "trainer_image": f"{REPO}/trainer:latest",
        "serving_image": SERVING_IMAGE,
        "pipeline_sa": PIPELINE_SA,
        "endpoint_name": "maint-endpoint",
        "min_auc": 0.80,
    },
)
job.submit(service_account=PIPELINE_SA)
print("submitted:", job.resource_name)
```

```bash
python run_pipeline.py
```

Open **Vertex AI → Pipelines** → your run, and watch the DAG go green: train → evaluate → (conditionally) deploy. The whole run takes ~20–30 minutes, almost all of it the training step.

> **Heads up: this is the moment the endpoint appears, and the endpoint bills.** From here until you undeploy it (cost-hygiene section), there's a node running.

✅ **Checkpoint 7:** a pipeline run completes; because the metric cleared `min_auc`, an endpoint named `maint-endpoint` appears in **Vertex AI → Online prediction**. The green DAG in the pipeline view is the confirmation that train → evaluate → conditional deploy all ran.

---

# Phase 8 — Real-time scoring service (~2 hrs)

**Goal:** A Cloud Run service that takes a machine's latest features, calls the endpoint, returns a failure risk, and records it in `predictions` for the dashboard and the alerts.

## Step 8.1 — The scorer

```bash
mkdir -p ~/dev/predictive-maint-adv/scorer
```

**`scorer/main.py`**:

```python
import logging
import os
import sys
from datetime import datetime, timezone

from flask import Flask, request
from google.cloud import aiplatform, bigquery

# ── LOGGING: load-bearing for Phase 9, and silently broken by default ────────
# Cloud Run captures whatever the container writes to stdout and files it in
# Cloud Logging as `textPayload`. Two things have to be true for that to work,
# and a bare print() gets neither of them right:
#
#   1. stdout must be UNBUFFERED. Python block-buffers stdout whenever it is not
#      attached to a TTY — which, inside a container, it never is. A short line
#      like "HIGH_RISK machine=M001 prob=0.976" sits in a 4–8 KB buffer and is
#      never flushed, so Cloud Logging never sees it. The curl still returns
#      0.976 and the BigQuery row still lands: the ONLY symptom is a log line
#      that isn't there and an alert that never fires. (Belt and braces:
#      PYTHONUNBUFFERED=1 is also set in the Dockerfile below.)
#
#   2. The message must be PLAIN TEXT on stdout. It is tempting to reach for
#      google-cloud-logging's setup_logging() here — don't. It ships structured
#      entries, which land in `jsonPayload.message`, NOT `textPayload`. The
#      Phase 9 log-based metric filters on textPayload=~"HIGH_RISK", so it would
#      stop matching and you'd be debugging Monitoring for an hour.
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger("scorer")

app = Flask(__name__)
PROJECT, REGION = os.environ["PROJECT"], "europe-west3"
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "maint-endpoint")
RISK_THRESHOLD = float(os.environ.get("RISK_THRESHOLD", "0.8"))

aiplatform.init(project=PROJECT, location=REGION)
bq = bigquery.Client(project=PROJECT)

# EXACT same order as FEATURES in training/train.py. The serving container gets
# a bare list of numbers with no column names — if the order drifts, the model
# silently reads temperature as vibration and nothing errors. Change one, change both.
FEATURES = ["temp_mean", "temp_max", "vibration_mean", "vibration_std",
            "rpm_mean", "pressure_mean", "voltage_mean"]

_endpoint = None
_version = None


def endpoint():
    global _endpoint, _version
    if _endpoint is None:
        eps = aiplatform.Endpoint.list(filter=f'display_name="{ENDPOINT_NAME}"')
        if not eps:
            raise RuntimeError(f"no endpoint named {ENDPOINT_NAME} — run the Phase 7 pipeline")
        ep = eps[0]
        dms = ep.gca_resource.deployed_models
        if not dms:
            raise RuntimeError(f"endpoint {ENDPOINT_NAME} exists but has no model deployed "
                               f"— did you undeploy it for cost reasons?")
        # Record WHICH MODEL produced each score. Logging the endpoint's name
        # here instead — the tempting shortcut — tells you nothing when
        # you later ask "which model version made this prediction?"
        _version = f"{dms[0].model.split('/')[-1]}@{dms[0].model_version_id or '1'}"
        # CACHE LAST — only after BOTH guards pass. Assigning _endpoint before
        # checking deployed_models is a live trap: the
        # Phase 7 pipeline CREATES the endpoint minutes before it finishes
        # deploying a model onto it, so a request in that window would cache the
        # empty endpoint with _version = None. Every later call on that warm
        # instance sees `_endpoint is not None`, returns early, skips the guard
        # entirely — and writes predictions to BigQuery with a NULL model_version
        # forever. It never errors. You just quietly lose your provenance column.
        _endpoint = ep
        log.info("endpoint resolved model_version=%s", _version)
    return _endpoint


@app.route("/", methods=["GET"])
def health():
    return {"status": "ok"}, 200


@app.route("/score", methods=["POST"])
def score():
    body = request.get_json(silent=True) or {}
    machine_id = body.get("machine_id", "unknown")
    try:
        instance = [float(body[f]) for f in FEATURES]      # order MUST match training
    except KeyError as e:
        return {"error": f"missing feature {e}"}, 400

    try:
        pred = endpoint().predict(instances=[instance])
    except RuntimeError as e:
        log.error("endpoint unavailable: %s", e)
        return {"error": str(e)}, 503                      # say so plainly, don't 500

    # The REGRESSOR's predict() returns one float per instance. Gradient boosting
    # on a 0/1 target overshoots slightly (measured range: -0.03 to +1.05), so
    # clamp before you call it a probability.
    prob = min(1.0, max(0.0, float(pred.predictions[0])))

    if prob > RISK_THRESHOLD:
        # Phase 9's log-based metric counts this EXACT line. Plain stdout, no
        # structured payload — see the logging note at the top of this file.
        log.info("HIGH_RISK machine=%s prob=%.3f", machine_id, prob)

    errs = bq.insert_rows_json(f"{PROJECT}.maintenance.predictions", [{
        "machine_id": machine_id,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "failure_prob": prob,
        "model_version": _version,
    }])
    if errs:
        log.error("BQ insert errors: %s", errs)
        return {"error": "insert failed"}, 500

    return {"machine_id": machine_id, "failure_prob": prob,
            "model_version": _version}, 200
```

**`scorer/requirements.txt`**:

```
flask==3.0.3
gunicorn==22.0.0
google-cloud-aiplatform==1.71.0
google-cloud-bigquery==3.25.0
```

**`scorer/Dockerfile`**:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
# Belt and braces with the logging.basicConfig() call in main.py. Python
# block-buffers stdout inside a container, and a buffered log line is a log line
# Cloud Logging never receives — which means the Phase 9 alert can never fire.
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 main:app
```

> **If you already deployed the scorer with `print()` and no `PYTHONUNBUFFERED`,** you don't have to rebuild to test the theory — flip the env var on the running service and re-curl:
> ```bash
> gcloud run services update scorer --region europe-west3 \
>   --update-env-vars PYTHONUNBUFFERED=1
> ```
> The `HIGH_RISK` line appears immediately. Then apply the code changes above properly and redeploy, so the fix survives the next image build.

## Step 8.2 — Deploy

> **First-`--source`-deploy gotcha:** `gcloud run deploy --source` builds via Cloud Build running as the **Compute Engine default service account**, which sometimes lacks build permission out of the box. If the deploy fails with a build-permission error, grant it once:
> ```bash
> PROJECT_NUMBER=$(gcloud projects describe predictive-maint-adv --format='value(projectNumber)')
> gcloud projects add-iam-policy-binding predictive-maint-adv \
>   --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
>   --role="roles/run.builder"
> ```

```bash
cd ~/dev/predictive-maint-adv/scorer
gcloud run deploy scorer --source . --region europe-west3 \
  --set-env-vars PROJECT=predictive-maint-adv,ENDPOINT_NAME=maint-endpoint \
  --service-account scorer@predictive-maint-adv.iam.gserviceaccount.com \
  --no-allow-unauthenticated
```

> **Gotcha — `ERROR: (gcloud.run.deploy) could not find source [./scorer]`.** `--source` is resolved relative to your **current working directory**, not the repo root. You will spend this project hopping between `pipeline/`, `retrainer/` and `scorer/`, and the same command that worked ten minutes ago now fails because `pwd` moved. Two habits fix it permanently: `cd` into the service folder and use `--source .` (as above), or always run deploys from the repo root and use `--source ./scorer`. Pick one and stick to it. `pwd` before you deploy.

## Step 8.3 — Test with two payloads: a dying machine and a healthy one

> **Wait for the Phase 7 DAG to read 4/4, not just for the endpoint to appear.** `deploy_model` creates the endpoint *first* and then deploys a model onto it, and the deploy takes another 10–15 minutes (Vertex is provisioning an `n1-standard-2` and pulling the serving container onto it). So `gcloud ai endpoints list` will show you `maint-endpoint` well before anything is actually serving. Curl it in that window and you get `endpoint … exists but has no model deployed` — which is the guard working, but it also means that Cloud Run instance is now warm. If you hit that error, force a cold instance before your real test, or you may be testing against poisoned in-memory state:
>
> ```bash
> gcloud run services update scorer --region europe-west3 \
>   --update-env-vars CACHE_BUST=$(date +%s)
> ```

```bash
SCORER_URL=$(gcloud run services describe scorer --region europe-west3 --format='value(status.url)')
TOKEN=$(gcloud auth print-identity-token)

# A machine at ~90% wear: hot, shaking, running slow. Expect a HIGH score.
curl -s -X POST "$SCORER_URL/score" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"machine_id":"M001","temp_mean":95.0,"temp_max":101.0,"vibration_mean":2.10,
       "vibration_std":0.06,"rpm_mean":1325,"pressure_mean":23.0,"voltage_mean":230.0}'

# A healthy machine. Expect a score near zero — this is the control case that
# proves the model DISCRIMINATES rather than just saying "risky" to everything.
curl -s -X POST "$SCORER_URL/score" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"machine_id":"M002","temp_mean":64.0,"temp_max":70.0,"vibration_mean":0.24,
       "vibration_std":0.05,"rpm_mean":1480,"pressure_mean":29.2,"voltage_mean":230.0}'
```

Expect roughly **0.97–1.0** for the first and **~0.00** for the second — and a `HIGH_RISK` line in **Cloud Run → scorer → Logs** for the first only.

> **⚠️ Why those specific numbers, and why a "worse" machine can score *lower*.** Two things here trip people up, and both are worth understanding rather than working around.
>
> **(1) Test payloads must be inside the training distribution.** Gradient-boosted trees do not extrapolate — they only ever return a value they learned from some leaf. Feed them a machine that *cannot exist* and you get a meaningless answer, not an error. The obvious "make it extreme" test payload — `temp_max: 110, pressure_mean: 22, voltage_mean: 229` — is a trap. In this simulator temperature physically maxes out around 105, pressure never drops below ~22.4, and `voltage_mean` (averaged over 120 readings) is pinned within ±0.6 of 230. All three values are impossible, the tree lands somewhere arbitrary, and the payload scores **0.05** — so the checkpoint could never pass and it looked like the *model* was broken. Every number in the two payloads above is drawn from the real feature ranges.
>
> **(2) The riskiest window is not the most-degraded window.** A window at ~95 % wear scores *lower* than one at ~90 %. That is correct, not a bug: the label is "will this machine fail in the next 10 minutes," and a machine that extreme has already died and been replaced *inside that window* — so the next 10 minutes contain no failure. Risk peaks just *before* the end, then falls. That is genuinely what predictive maintenance looks like, and explaining it unprompted is a very good sign to a reviewer.

✅ **Checkpoint 8:** the high-wear payload returns a risk near 1.0 and logs `HIGH_RISK`; the healthy payload returns ~0.0 and logs nothing; both land rows in `predictions` tagged with a real model version.

---

# Phase 9 — Batch scoring + alerting (~1.5 hrs)

**Goal:** Score the whole fleet on a schedule, and alert a human when a machine crosses the risk threshold.

## Step 9.1 — Batch scoring (already done, and why it's separate)

You built this in **Step 5.3** — the BQML model scoring every machine's latest window into `predictions` every 15 minutes. That's deliberate architecture, not laziness:

> **Two scoring paths, on purpose.** *Batch* runs on **BQML inside the warehouse** — no endpoint, no node, no cost, and it keeps working even when you undeploy the endpoint to save money. *Real-time* runs on the **Vertex endpoint** — low latency, for the one machine someone is asking about right now. Both write to the same `predictions` table, tagged by `model_version`. That's a real resilience story ("the fleet dashboard doesn't go dark when the endpoint is down"), and it's why the dashboard in Phase 12 has data even when everything expensive is switched off.

Confirm it's running:

```sql
SELECT model_version, COUNT(*) AS scores, MAX(scored_at) AS latest
FROM `predictive-maint-adv.maintenance.predictions`
GROUP BY model_version;
```

## Step 9.2 — Alert on high risk (log-based metric)

The scorer prints `HIGH_RISK machine=... prob=...` (Phase 8). Turn that line into an alert.

1. Console → **Logging → Log-based metrics → Create metric**. Type **Counter**, name `high_risk_scores`, filter:
   ```
   resource.type="cloud_run_revision"
   resource.labels.service_name="scorer"
   textPayload=~"HIGH_RISK"
   ```

2. **Confirm the line is actually in Cloud Logging before you build anything on top of it.** Do not skip this — it is the whole reason the next two gotchas exist. Re-send the Phase 8 high-risk `curl`, wait ~30 seconds, then:

   ```bash
   gcloud run services logs read scorer --region europe-west3 --limit 20
   ```

   You want to see, verbatim:

   ```
   2026-07-12 14:52:07 HIGH_RISK machine=M001 prob=0.976
   ```

   No line → **stop here** and fix the scorer, not the alert. Work through the two gotchas below.

3. Console → **Monitoring → Alerting → Create policy**. The console splits this across four screens, and the old single-page "any time series is above 0" wording no longer appears anywhere — it now lives on the *third* screen under a different name. Fill them in like this:

   | Screen | Field | Value |
   |---|---|---|
   | **Select a metric** | Policy configuration mode | **Builder** |
   | | Metric | `Cloud Run Revision - logging/user/high_risk_scores` |
   | **Transform data** | Rolling window | `5 min` |
   | | Rolling window function | **`sum`** — *not* the default `rate` |
   | | Across time series | leave as-is (one scorer service = one series) |
   | **Configure trigger** | Condition type | **Threshold** |
   | | Alert trigger | **Any time series violates** ← this *is* the old "any time series is above 0" |
   | | Threshold position | **Above threshold** |
   | | Threshold value | **`0`** |
   | **Notifications and name** | Notification channels | your email (see below) |
   | | Alert policy name | `high-risk-score-detected` |

   Then **Review alert → Create Policy**.

   To add the email channel: open the **Notification channels** dropdown → **Manage notification channels** (new tab) → **Email → Add new** → your address → save → return to the policy tab and hit the refresh icon in the dropdown. The channel won't appear until you refresh.

4. Fire it deliberately: re-send the Phase 8 high-risk `curl` two or three times, wait 5–10 minutes, then check **Monitoring → Alerting → Incidents** and your inbox.

> **Why `sum` and not `rate`.** `rate` converts the counter to *events per second*, so a single `HIGH_RISK` line over a 5-minute window reads as `0.0033`. That is still technically above `0` and the alert does fire — but the chart looks empty, the number is meaningless to a human, and you cannot tell one event from ten. `sum` gives you whole numbers (1, 2, 3…), which is what you actually want to reason about and what you'd want on a screenshot in your README.

> **Ignore "No data is available for the selected time frame" on the preview chart.** A log-based metric only counts lines written **after the metric was created**, so until you send a fresh high-risk `curl` the chart is legitimately empty. It does not block policy creation, and it is not a sign that anything is wrong.

> **Gotcha — a log-based metric doesn't exist until it has matched something.** If the alert policy's metric picker can't find `logging/user/high_risk_scores`, it's because no matching log line has been written yet. Send one high-risk `curl` first, wait a minute, then create the policy.

> **Gotcha — the missing `HIGH_RISK` line is almost always buffering, not your filter.** This is the single most likely place in the project to lose an hour. The curl returns `0.976`, the row lands in `predictions`, and Cloud Logging shows nothing. It is not Monitoring, it is not the filter, it is not IAM: Python **block-buffers stdout** inside a container, and a bare `print()` never gets flushed. Phase 8's `scorer/main.py` and `Dockerfile` (`logging.basicConfig(stream=sys.stdout, …)` + `ENV PYTHONUNBUFFERED=1`) exist entirely to prevent this. If you typed the scorer from an older draft, go back and apply both.

> **Gotcha — don't debug this in Logs Explorer's query box.** The console now defaults to the new **LQL** query language, which does not understand the legacy `textPayload=~"HIGH_RISK"` syntax. It silently wraps what you typed into `SEARCH("\`textPayload=~\`")` — i.e. it goes looking for the *literal string* `textPayload=~` — and returns **0 results even when the log line exists**. You then "fix" a scorer that was never broken. Two ways out, both better than fighting the editor:
>
> - `gcloud run services logs read scorer --region europe-west3 --limit 20` (fastest, and what step 2 above uses).
> - In Logs Explorer, **clear the query box entirely**, set the range to **Last 1 hour**, and filter with the dropdowns instead: **All resources → Cloud Run Revision → scorer**. Then read the lines with your eyes.
>
> The `textPayload=~"HIGH_RISK"` filter in the log-based *metric* (step 1) is fine — that field is evaluated by the logging backend, not by the LQL editor. It's only the interactive query box that mangles it.

> **Gotcha — the whole alert silently depends on an IAM role three phases back.** This metric matches a line the *scorer container* prints. A Cloud Run container writes stdout to Cloud Logging **as its runtime service account** — so if `scorer@` is missing `roles/logging.logWriter` (Phase 1), the `HIGH_RISK` line is never recorded, the metric never matches, and the alert can never fire. Nothing errors: the curl still returns `0.97`, the BigQuery row still lands, and the log is simply empty. Confirm the plumbing before you go hunting through Monitoring:
>
> ```bash
> gcloud logging read \
>   'resource.type="cloud_run_revision" AND resource.labels.service_name="scorer"
>    AND textPayload=~"HIGH_RISK"' \
>   --project predictive-maint-adv --limit 5 --freshness=1h
> ```
>
> Rows → the alert will work. Empty, after a high-risk curl that returned a high score → it's the missing role, not your filter.

> **This is real model monitoring** — the same instinct as a 5xx alert, but watching *model outputs* rather than service errors. In production you'd also alert on the batch table (e.g. "more than N machines above 0.8"), which is a scheduled query plus a second log-based metric — worth a sentence in the README.

✅ **Checkpoint 9:** `predictions` refreshes on schedule with BQML scores, and a test high-risk score triggered an email alert.

---

# Phase 10 — Drift detection + continuous training (~3 hrs)

**Goal:** The senior move. Detect when incoming data has **drifted** away from what the model was trained on, and automatically **re-run the training pipeline** when it does. This closes the loop from CI/CD (deploy on *code* change) to **CT** (retrain on *data* change).

> **Term — drift:** the live data distribution moves away from the training distribution (machines age, a sensor gets recalibrated, a supplier changes a part), so a once-good model quietly degrades. Catching it *before* accuracy craters is the difference between proactive and reactive ML.

## Step 10.1 — The drift query

**`sql/drift.sql`**:

```sql
-- z_shift is an EFFECT SIZE (a standardized mean shift — Cohen's d):
--     |recent_mean - baseline_mean| / baseline_stddev
-- Read it as "how many baseline standard deviations has the mean moved?"
-- Convention: ~0.2 small, ~0.5 medium, ~0.8 large. We fire at 1.0 — a full
-- baseline standard deviation, which is unambiguous and never trips on noise.
WITH bounds AS (
  SELECT MIN(window_end) AS t0, MAX(window_end) AS t1
  FROM `predictive-maint-adv.maintenance.features_windowed`
),
baseline AS (
  -- The EARLIEST 2 hours of data, as a proxy for the training distribution.
  -- (Do NOT use "older than 1 day": for the first 24 hours of the project's
  -- life that window is EMPTY, so the mean is NULL, so z is NULL, so the drift
  -- check silently does nothing — forever, and without ever erroring.)
  SELECT AVG(vibration_mean) AS mu, STDDEV(vibration_mean) AS sd, COUNT(*) AS n
  FROM `predictive-maint-adv.maintenance.features_windowed`, bounds
  WHERE window_end < TIMESTAMP_ADD(bounds.t0, INTERVAL 2 HOUR)
),
recent AS (
  SELECT AVG(vibration_mean) AS mu, COUNT(*) AS n
  FROM `predictive-maint-adv.maintenance.features_windowed`, bounds
  WHERE window_end >= TIMESTAMP_SUB(bounds.t1, INTERVAL 30 MINUTE)
)
SELECT
  SAFE_DIVIDE(ABS(recent.mu - baseline.mu), NULLIF(baseline.sd, 0)) AS z_shift,
  baseline.mu AS baseline_mean,
  baseline.sd AS baseline_sd,
  baseline.n  AS baseline_windows,
  recent.mu   AS recent_mean,
  recent.n    AS recent_windows
FROM baseline, recent;
```

> **Note it reads `features_windowed`, the LIVE table — not `features_labeled`.** `features_labeled` is a *snapshot* that only changes when something rebuilds it. Pointing a drift monitor at a snapshot means it faithfully reports "no drift" while the world moves, because it is looking at a photograph. Drift monitoring must read the stream.

> **In production you'd pin the baseline to the exact data snapshot the deployed model was trained on**, not "the oldest rows we happen to have." Say that in the README — it's the difference between running a drift query and understanding one.

## Step 10.2 — The retrainer

```bash
mkdir -p ~/dev/predictive-maint-adv/retrainer
```

**`retrainer/main.py`**:

```python
import os
import sys
from datetime import datetime, timezone

from google.cloud import aiplatform, bigquery

PROJECT, REGION = os.environ["PROJECT"], "europe-west3"
BUCKET = "gs://predictive-maint-adv-artifacts"
REPO = "europe-west3-docker.pkg.dev/predictive-maint-adv/maint"
PIPELINE_SA = "vertex-pipeline@predictive-maint-adv.iam.gserviceaccount.com"
SERVING_IMAGE = "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"
THRESHOLD = float(os.environ.get("Z_THRESHOLD", "1.0"))

bq = bigquery.Client(project=PROJECT)

# 1) How far has the LIVE sensor distribution moved from the baseline?
row = list(bq.query(open("drift.sql").read()).result())[0]
z = row["z_shift"]
print(f"z_shift={z}")
print(f"  baseline: mean={row['baseline_mean']} sd={row['baseline_sd']} "
      f"n={row['baseline_windows']}")
print(f"  recent:   mean={row['recent_mean']} n={row['recent_windows']}")

if z is None:
    print("z_shift is NULL — one side of the comparison has no rows. "
          "Is the Dataflow job draining/drained?")

retrained = False
if z is not None and z > THRESHOLD:
    # 2) REBUILD THE LABELS BEFORE RETRAINING.
    #    The pipeline trains on v_training -> features_labeled, which is a
    #    SNAPSHOT. Skip this and the "retrain" dutifully re-fits the same model
    #    on the same stale rows and changes precisely nothing — a continuous-
    #    training loop that is, silently, a no-op. This one line is what makes
    #    the loop actually close.
    print("drift detected → rebuilding labels from the latest windows + failure events")
    bq.query(open("label.sql").read()).result()

    aiplatform.init(project=PROJECT, location=REGION, staging_bucket=BUCKET)
    job = aiplatform.PipelineJob(
        display_name="maint-retrain-on-drift",
        template_path="maint_pipeline.json",
        pipeline_root=f"{BUCKET}/pipeline-root",
        parameter_values={
            "project": PROJECT,
            "region": REGION,
            "staging_bucket": BUCKET,
            "trainer_image": f"{REPO}/trainer:latest",
            "serving_image": SERVING_IMAGE,
            "pipeline_sa": PIPELINE_SA,
            "endpoint_name": "maint-endpoint",
            "min_auc": 0.80,
        },
    )
    job.submit(service_account=PIPELINE_SA)
    retrained = True
    print("retraining pipeline submitted:", job.resource_name)
else:
    print("no significant drift → no action")

errs = bq.insert_rows_json(f"{PROJECT}.maintenance.drift_log", [{
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "metric": "vibration_mean",
    "z_shift": float(z) if z is not None else None,
    "retrained": retrained,
}])
if errs:
    print("drift_log insert errors:", errs)
    sys.exit(1)
```

**`retrainer/requirements.txt`**:

```
google-cloud-bigquery==3.25.0
google-cloud-aiplatform[pipelines]==1.71.0
db-dtypes==1.3.0
```

> **Gotcha — `ModuleNotFoundError: No module named 'yaml'`, and why the `[pipelines]` extra is not optional.** Note that this is **not** the bare `google-cloud-aiplatform` you installed everywhere else. `aiplatform.PipelineJob(template_path=…)` parses the compiled pipeline spec through the SDK's `yaml_utils` — and it does that **even when the template is a `.json` file**, which is exactly why this trap is so easy to walk into. PyYAML is *not* a dependency of the base SDK; it only arrives with the `[pipelines]` extra. Install the bare package and the image builds cleanly, pushes cleanly, deploys cleanly, and then the job exits 1 at runtime with:
>
> ```
> ImportError: PyYAML is not installed and is required to parse PipelineJob or
> PipelineSpec files.
> ```
>
> You never see it locally, because `pip install kfp` (Step 7.1) drags PyYAML into your venv as a transitive dependency — so `run_pipeline.py` works on your Mac and the identical code dies in Cloud Run. **The rule: any environment that constructs a `PipelineJob` needs `google-cloud-aiplatform[pipelines]`, not the bare package.** Right now that is the retrainer container and your local venv. If you later add a Cloud Function or a CI step that submits the pipeline, it needs the extra too:
>
> ```bash
> grep -rn "PipelineJob" --include=*.py ~/dev/predictive-maint-adv
> ```
>
> Every directory that shows up in that output needs the extra in its `requirements.txt`.

**`retrainer/Dockerfile`**:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py drift.sql label.sql maint_pipeline.json ./
ENTRYPOINT ["python", "main.py"]
```

The retrainer needs three files that live elsewhere in the repo. Copy them in (and re-copy whenever you change them — the image is a frozen snapshot):

```bash
cd ~/dev/predictive-maint-adv
cp sql/drift.sql sql/label.sql pipeline/maint_pipeline.json retrainer/
```

## Step 10.3 — Deploy it as a Cloud Run *job* on a schedule

A Cloud Run **job** runs to completion and exits — the right shape for a periodic check (unlike a *service*, which sits waiting for requests).

```bash
cd ~/dev/predictive-maint-adv/retrainer

gcloud run jobs deploy retrainer --source . --region europe-west3 \
  --set-env-vars PROJECT=predictive-maint-adv,Z_THRESHOLD=1.0 \
  --service-account vertex-pipeline@predictive-maint-adv.iam.gserviceaccount.com

# Run it once by hand. With no drift injected, it should print a small z_shift
# (typically under 0.1) and "no significant drift → no action".
gcloud run jobs execute retrainer --region europe-west3 --wait
```

> **Gotcha — `execute` does not rebuild.** `gcloud run jobs execute` runs whatever image is already registered against the job. Edit `main.py`, `requirements.txt`, or any of the three copied-in files and *nothing changes* until you re-run `gcloud run jobs deploy … --source .`. Fixing a bug and re-executing is a very natural thing to do, and it will fool you into thinking the fix didn't work. **Deploy, then execute.** Every time.

> **Gotcha — `execute --wait` tells you the job failed, and nothing else.** All you get is `Task retrainer-xxxxx-task0 failed with exit code: 1`. The traceback is in Cloud Logging. Note the execution name from the error and read it:
>
> ```bash
> gcloud logging read \
>   'resource.type="cloud_run_job"
>    resource.labels.job_name="retrainer"
>    labels."run.googleapis.com/execution_name"="retrainer-8d2p2"
>    severity>=WARNING' \
>   --limit 50 --format='value(textPayload)' --project predictive-maint-adv
> ```
>
> (`gcloud run jobs executions logs read` does **not** exist in the GA surface — it's `gcloud beta run jobs executions logs read`, or the `gcloud logging read` above, which always works.)

Then schedule it daily:

```bash
gcloud run jobs add-iam-policy-binding retrainer --region europe-west3 \
  --member="serviceAccount:vertex-pipeline@predictive-maint-adv.iam.gserviceaccount.com" \
  --role="roles/run.invoker"

gcloud scheduler jobs create http retrain-check \
  --location europe-west3 \
  --schedule "0 6 * * *" \
  --http-method POST \
  --uri "https://run.googleapis.com/v2/projects/predictive-maint-adv/locations/europe-west3/jobs/retrainer:run" \
  --oauth-service-account-email vertex-pipeline@predictive-maint-adv.iam.gserviceaccount.com
```

> **⚠️ Off-switch #3, and the one everybody forgets.** This Scheduler job now fires **every day, forever**. If it ever detects drift after you've torn everything down, it will submit a pipeline that **creates a new billing endpoint** while you're not looking. `gcloud scheduler jobs pause retrain-check --location europe-west3` is in the cost-hygiene checklist for exactly this reason.

## Step 10.4 — Force real drift and watch the loop close

**How *not* to do it:** the intuitive move is to restart the generator with a much higher `DECAY` so machines wear out faster. **This does not produce drift, and it is worth understanding why.** Raising the decay makes each machine cycle through its life faster — but the *fleet* still contains a steady mix of healthy, middling, and dying machines at any instant, in roughly the same proportions. The distribution barely moves. Simulating this exact loop: raising `DECAY` by **25×** shifts the fleet-wide mean vibration by **z = 0.01**. It can never trip a threshold, and you'd spend an evening assuming your drift query was broken.

**How to do it:** inject a **sensor fault**. `VIB_OFFSET` adds a constant bias to every vibration reading — exactly what a miscalibrated or degrading sensor array does in the real world, and one of the most common causes of genuine production drift.

> **⚠️ Restart the Dataflow job FIRST — it is not running.** Step 3.4 told you to drain it at the end of every session, and you have had several sessions since. A generator with no consumer produces **nothing**: messages pile up in the subscription, `features_windowed` gains no new rows, the drift check reads a stale window, and `z_shift` comes back **≈ 0**. You then spend an evening debugging a drift query that was correct all along. This is the single easiest way to lose a night on this project.

Relaunch the pipeline (same command as Step 3.3), and leave it running:

```bash
cd ~/dev/predictive-maint-adv && source .venv-beam/bin/activate
export GRPC_VERBOSITY=ERROR
export GRPC_ENABLE_FORK_SUPPORT=0
python dataflow/stream_features.py \
  --project_id predictive-maint-adv \
  --subscription projects/predictive-maint-adv/subscriptions/telemetry-df-sub \
  --runner DataflowRunner \
  --project predictive-maint-adv \
  --region europe-west3 \
  --worker_zone europe-west3-a \
  --worker_machine_type n1-standard-2 \
  --enable_streaming_engine \
  --temp_location gs://predictive-maint-adv-artifacts/dataflow/temp \
  --staging_location gs://predictive-maint-adv-artifacts/dataflow/staging \
  --service_account_email df-worker@predictive-maint-adv.iam.gserviceaccount.com \
  --max_num_workers 2 \
  --job_name maint-drift-demo
```

Confirm it reaches `JOB_STATE_RUNNING` (`gcloud dataflow jobs list --region europe-west3 --status active`), then Ctrl-C the terminal — the cloud job keeps going.

Now stop the generator (Ctrl-C) and restart it **with the offset**, keeping everything else identical:

```bash
cd ~/dev/predictive-maint-adv && source .venv/bin/activate
PROJECT=predictive-maint-adv VIB_OFFSET=1.0 python generator/generate.py
```

Leave it (and the Dataflow job) running for **~30 minutes** so the recent window fills with drifted data. Then:

```bash
gcloud run jobs execute retrainer --region europe-west3 --wait
```

Expect `z_shift ≈ 1.6` (measured), comfortably over the 1.0 threshold → labels rebuild → a new pipeline run appears in **Vertex AI → Pipelines**. And because that pipeline carries the `dsl.If` gate, a retrain that *doesn't* clear `min_auc` simply won't ship. The loop is self-correcting.

Check the log:

```sql
SELECT checked_at, metric, ROUND(z_shift, 3) AS z_shift, retrained
FROM `predictive-maint-adv.maintenance.drift_log`
ORDER BY checked_at DESC;
```

> **Then set it back.** Restart the generator without `VIB_OFFSET` when you're done, or every future run trains on biased data.

✅ **Checkpoint 10:** a `drift_log` row with `z_shift > 1.0` and `retrained = true`, and the automatically-triggered pipeline run visible in the Vertex AI console.

---

# Phase 11 — dbt analytics layer (~2 hrs)

**Goal:** Turn raw predictions and drift logs into clean, tested marts — the analytics-engineering layer on top of the ML.

> **Term — dbt:** builds BigQuery tables/views out of `SELECT` statements, with a dependency graph (`ref()`), automated tests, and generated docs. The point isn't the SQL — it's that the SQL becomes versioned, tested, and reproducible.

## Step 11.1 — Set up dbt

dbt authenticates through your **Application Default Credentials** (the second login from Phase 0.2), so no keys are involved.

```bash
cd ~/dev/predictive-maint-adv && source .venv/bin/activate
pip install dbt-bigquery
dbt init maint_dbt
```

`dbt init` asks a short series of questions — answer them exactly like this:

| Prompt | Answer |
|--------|--------|
| Which database would you like to use? | type the number next to **bigquery** |
| Desired authentication method | **oauth** |
| project (GCP project id) | **predictive-maint-adv** |
| dataset (BigQuery dataset) | **maintenance** |
| threads | **4** |
| job_execution_timeout_seconds | press **Enter** (default 300) |
| Desired location | **EU** (must match your BigQuery multi-region) |

dbt writes these to `~/.dbt/profiles.yml` — on your machine, **not** in the repo. Confirm the connection, then delete the sample models:

```bash
cd maint_dbt
dbt debug            # must end with "All checks passed!"
rm -r models/example
```

### If `dbt init` didn't do what it promised

`dbt init` is an interactive prompt, and it fails in two silent ways. Both are common enough that it's worth knowing the manual path — it's more reliable than the wizard and takes thirty seconds.

**Failure 1 — `Internal Error: Profile should not be None if loading profile completed`.** dbt found a `dbt_project.yml` but the profile it names doesn't exist in `~/.dbt/profiles.yml`. The wizard writes nothing if you interrupt it, and it *skips scaffolding entirely* if it thinks a project already exists. Don't fight it — write the profile by hand:

```bash
grep -n "^name:\|^profile:" dbt_project.yml   # note the profile name it prints

mkdir -p ~/.dbt
cat > ~/.dbt/profiles.yml << 'EOF'
maint_dbt:
  target: dev
  outputs:
    dev:
      type: bigquery
      method: oauth
      project: predictive-maint-adv
      dataset: maintenance
      location: EU
      threads: 4
      job_execution_timeout_seconds: 300
      priority: interactive
EOF
```

> **The top-level key must match `profile:` in `dbt_project.yml` character for character.** If `grep` printed `profile: 'predictive_maint_adv'`, rename the `maint_dbt:` key above to `predictive_maint_adv:`. This single mismatch is the most common dbt setup failure in existence, and the error message names neither file.

**Failure 2 — `project path <.../dbt_project.yml> not found`, while `profiles.yml` and the connection test both pass green.** Nothing is broken; you're just in the wrong directory. dbt looks for `profiles.yml` globally in `~/.dbt/`, but for `dbt_project.yml` only in the **current** directory. That asymmetry is why the connection can test OK while project loading fails.

If `dbt init` scaffolded into the repo root instead of `maint_dbt/`, put it where the rest of this guide expects it:

```bash
cd ~/dev/predictive-maint-adv
mkdir -p maint_dbt
mv dbt_project.yml models macros seeds snapshots tests analyses maint_dbt/ 2>/dev/null
cd maint_dbt && rm -rf models/example && dbt debug
```

> **Every `dbt` command from here on runs from `~/dev/predictive-maint-adv/maint_dbt`.** If a `dbt` command errors with "project path not found," that is *always* the reason — check `pwd` before you check anything else.

If `dbt debug` now fails on authentication instead, your ADC token expired: `gcloud auth application-default login`.

## Step 11.2 — The models

**`models/stg_predictions.sql`**

```sql
SELECT
  machine_id,
  scored_at,
  failure_prob,
  CASE WHEN failure_prob >= 0.8 THEN 'high'
       WHEN failure_prob >= 0.5 THEN 'medium'
       ELSE 'low' END AS risk_bucket,
  model_version
FROM {{ source('maintenance', 'predictions') }}
```

**`models/mart_fleet_health.sql`**

```sql
WITH latest AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY machine_id ORDER BY scored_at DESC) AS rn
  FROM {{ ref('stg_predictions') }}
)
SELECT machine_id, scored_at, failure_prob, risk_bucket, model_version
FROM latest
WHERE rn = 1
```

**`models/mart_model_runs.sql`**

```sql
SELECT
  DATE(checked_at) AS day,
  metric,
  MAX(z_shift) AS max_z_shift,
  LOGICAL_OR(retrained) AS retrained_that_day
FROM {{ source('maintenance', 'drift_log') }}
GROUP BY day, metric
```

**`models/sources.yml`**

```yaml
version: 2

sources:
  - name: maintenance
    database: predictive-maint-adv
    schema: maintenance
    tables:
      - name: predictions
      - name: drift_log

models:
  - name: stg_predictions
    columns:
      - name: machine_id
        tests:
          - not_null
      - name: risk_bucket
        tests:
          - accepted_values:
              values: ['high', 'medium', 'low']

  - name: mart_fleet_health
    columns:
      - name: machine_id
        tests:
          - not_null
          - unique          # exactly one CURRENT risk row per machine
```

## Step 11.3 — Build

```bash
dbt build            # from ~/dev/predictive-maint-adv/maint_dbt
```

> **If the `unique` test on `mart_fleet_health.machine_id` fails**, two scores for the same machine share an identical `scored_at` (the batch query stamps every row with the same `CURRENT_TIMESTAMP()`, so a machine scored by *both* the batch query and the endpoint in the same instant can tie). Add `model_version` to the `ORDER BY` to break the tie deterministically. Don't delete the test — a test you understand is worth more than a green one you rigged.

✅ **Checkpoint 11:** `dbt build` creates all three marts with every test passing.

---

# Phase 12 — Fleet health dashboard (~1.5 hrs)

**Goal:** One screen that tells both the *operational* story (which machines need attention) and the *ML* story (is the model still healthy, and when did it last retrain).

Everything here is UI work — no code, but a surprising number of ways to build a chart that is quietly wrong. The gotchas below are the ones that bite.

## Step 12.1 — Connect the data

lookerstudio.google.com → **Create → Data source → BigQuery** → `predictive-maint-adv` → `maintenance` → **`mart_fleet_health`**. Then **Resource → Manage added data sources → Add a data source** twice more for **`mart_model_runs`** and **`stg_predictions`** (both are dbt-built views sitting in the same dataset).

## Step 12.2 — Fleet table (machines, worst first)

**Add a chart → Table.** Data source `mart_fleet_health`.

- **Dimension:** `machine_id`, then `risk_bucket` as a second dimension.
- **Metric:** `failure_prob` → click the chip → aggregation **Max**.
- **Sort:** `failure_prob`, **Descending**.

Then **Style → Conditional formatting → + Add**, three times:

| Field | Condition | Value | Colour |
|-------|-----------|-------|--------|
| `risk_bucket` | Equal to (=) | `high` | red background, white text |
| `risk_bucket` | Equal to (=) | `medium` | amber |
| `risk_bucket` | Equal to (=) | `low` | green |

Set **Scope** to *Colour row* if you want the whole row to light up — more readable on a wall-mounted fleet screen than a single tinted cell.

> **Two traps in one chart.** (1) Looker defaults the metric aggregation to **SUM**. `mart_fleet_health` is one row per machine (the `rn = 1` filter), so `SUM(failure_prob)` *happens* to equal the value and looks right — until a machine somehow gets two rows and the table cheerfully reports a risk of `1.7`. Set **Max** explicitly and it's correct by construction. (2) Conditional formatting only offers fields the chart actually uses, so `risk_bucket` must be a **dimension on the table** or it won't appear in the rule dropdown at all.

## Step 12.3 — Scorecard (machines currently `high`)

**Add a chart → Scorecard.** Data source `mart_fleet_health`.

- **Metric:** `machine_id` → aggregation **Count Distinct (CTD)**.
- **Filter** (inside the chart's Setup panel → *Add filter → Create a filter*): `Include` · `risk_bucket` · `Equal to (=)` · `high`. Name it `high_only`.
- **Style → Conditional formatting:** if value `> 0`, colour it red. A calm fleet should look calm; a burning one should shout.

> **Filter at the chart level, never the report level.** A report-level filter would also strip `medium` and `low` out of the table you just built, leaving you staring at a one-row fleet and wondering where everything went.

## Step 12.4 — Time series (prediction volume by scoring path)

**Add a chart → Time series.** Data source `stg_predictions`.

- **Date range dimension:** `scored_at` → click the chip → granularity **Date Hour**.
- **Metric:** `Record Count`.
- **Breakdown dimension:** `model_version` — but first, make it readable. Click **+ Add a field** at the bottom of the Data panel and create:

  | Field | Value |
  |-------|-------|
  | Display name | `scoring_path` |
  | Data type | Text |
  | Formula | `CASE WHEN model_version = 'bqml_failure' THEN 'BQML (batch)' ELSE CONCAT('Vertex endpoint ', model_version) END` |

  **Apply**, then use `scoring_path` as the breakdown dimension instead of the raw `model_version`.

> **Granularity is load-bearing.** Batch scoring runs every 15 minutes. At `Date` granularity you get one fat point per day and any version handover is invisible; at **Date Hour** you can watch one line die and another pick up. Also: a time series has **no sort option** — it's always chronological. The *"Breakdown dimension sort"* control you'll find in Setup orders the **legend/series**, not the x-axis, and it's purely cosmetic.

> **What you'll actually see at first, and why it isn't a retrain.** Your two initial series are `bqml_failure` and a long Vertex model ID. Those aren't two retrained versions — they're the **two scoring paths from Step 9.1**, batch and endpoint, both writing into `predictions` tagged by `model_version`. That's the *resilience* property ("the fleet dashboard doesn't go dark when the endpoint is undeployed"). The **version flip** — the CT property — only appears once Phase 10's drift run mints a genuinely new model version. Both are real; just don't call it a retrain until it is one.

## Step 12.5 — Drift trend + retrains (the money shot)

The literal reading — `retrained_that_day` as a breakdown dimension — splits your line into two gappy series and looks terrible. Build a **Combo chart** instead: same information, and it's the chart people remember.

**Add a chart → Combo chart.** Data source `mart_model_runs`.

- **Date range dimension:** `day`.
- **Metric 1:** `max_z_shift` → aggregation **Max**.
- **Metric 2:** **+ Add a field**:

  | Field | Value |
  |-------|-------|
  | Display name | `retrain_fired` |
  | Data type | Number |
  | Formula | `CASE WHEN retrained_that_day THEN 1 ELSE 0 END` |
  | Aggregation | **Max** |
  | Running calculation | **None** |

- **Filter:** if `drift_log` ever carries more than one metric, add `Include · metric · Equal to · vibration_mean` so you aren't blending metrics.

**Style tab:**

| Setting | Value |
|---------|-------|
| Series #1 (`max_z_shift`) | **Line**, **Left** axis |
| Series #2 (`retrain_fired`) | **Bars**, **Right** axis |
| Right Y-axis | Min **0**, Max **1** |
| Reference line | Type **Constant**, Value **1.0**, **Left** axis, dashed red, label `retrain threshold` |

Drag this chart to sit **directly beside** the Step 12.4 time series.

> **`Running calculation` must be `None`.** Looker offers *Running max* on the same dropdown, and it is catastrophic here: once a retrain fires on any single day, a running max pins the bar at `1` for **every day thereafter, forever**. Your chart would advertise a permanent retrain — the exact opposite of the story. Likewise set **Aggregation: Max**, not Sum: `mart_model_runs` groups by `day` *and* `metric`, so Sum would report `2` retrains on a day you retrained once as soon as you log a second drift metric.

> **The reference line is the whole point.** Without it, the z-shift line is an uninterpretable wiggle. With it, a stranger reads the story unaided: the line climbs → crosses the dashed threshold → a bar fires that same day → and in the chart next door, the model version flips. Drift → retrain → deploy, in two glances.

> **If every bar is zero, the chart is correct.** You simply haven't triggered Phase 10's drift run yet. Do not "fix" it by lowering the threshold. An empty bar series under a flat z-shift line is an *honest* picture of a fleet that hasn't drifted. Rigging the threshold to make the chart look busy breaks the one thing the chart is for.

## Step 12.6 — Name it, share it, link it

- **Report title** (top-left, currently "Untitled Report"): `predictive-maint-adv — Fleet Health & Model Ops`. Project name first so it matches the repo, the GCP project, and the dataset; the subtitle tells a reader what they're about to look at before it loads.
- **File → Report settings → Theme and layout:** pick a dark or high-contrast theme. Default Looker white is fine and forgettable.
- **Share → Manage access → Anyone with the link → Viewer** — *only if you are comfortable with the underlying data being public*. A dashboard nobody can open is worth nothing; a dashboard that exposes data you didn't mean to publish is worse than nothing. Keep it private otherwise.
- Put the link in the repo README next to the architecture diagram.

> **Sharing the report does not share the data.** Under **Resource → Manage added data sources**, each source has a **Data credentials** setting. It must be **Owner's credentials**, not *Viewer's credentials* — otherwise anyone who opens the link needs their own BigQuery permissions on the project, and instead of a dashboard they get a wall of access errors. Open the link in a private window and confirm it renders before sharing it.
>
> **A public link is a public link.** Anything on the dashboard — machine IDs, sensor values, model versions — is readable by anyone who has the URL, and the URL is not secret. Only publish a report whose data you are comfortable exposing, and treat the report URL itself as something you may not want in a public repository.

**Suggested layout:** scorecard top-left, fleet table beneath it (left half); the two time series stacked on the right half. Operational story down the left, ML story down the right.

✅ **Checkpoint 12:** a live, publicly-viewable dashboard a stranger could read in 30 seconds.

---

# Phase 13 — CI/CD with GitHub Actions (keyless WIF) (~2 hrs)

**Goal:** Push to `main` → GitHub Actions rebuilds the training image and redeploys the scorer, authenticating with **no stored key**.

> **Term — Workload Identity Federation (WIF):** instead of downloading a service-account JSON key and pasting it into GitHub secrets (where it lives forever, leaks eventually, and is the single most common finding in cloud security audits), GCP **trusts GitHub's OIDC token directly**. GitHub proves "this workflow is running in repo X"; GCP checks that against a trust condition and mints a short-lived token. No long-lived secret exists to steal.

## Step 13.1 — WIF pool + provider

```bash
gcloud services enable iam.googleapis.com sts.googleapis.com iamcredentials.googleapis.com \
  --project predictive-maint-adv

PROJECT_ID=predictive-maint-adv
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
GITHUB_USER="<your-github-username>"
REPO="${GITHUB_USER}/predictive-maint-adv"

gcloud iam workload-identity-pools create github \
  --project=$PROJECT_ID --location=global --display-name="GitHub Actions Pool"

gcloud iam workload-identity-pools providers create-oidc github-provider \
  --project=$PROJECT_ID --location=global --workload-identity-pool=github \
  --display-name="GitHub OIDC" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='${REPO}'"
```

> **The attribute condition is mandatory.** Without it, *any* GitHub repository on the planet could authenticate to your project. And note the trust is bound to this **exact** repo path — renaming the repo later breaks CI/CD until you update the condition.

## Step 13.2 — Deployer SA + let the repo impersonate it

```bash
gcloud iam service-accounts create github-deployer --project=$PROJECT_ID \
  --display-name="GitHub Actions Deployer"
DEPLOYER="github-deployer@${PROJECT_ID}.iam.gserviceaccount.com"

for ROLE in roles/run.admin roles/iam.serviceAccountUser roles/cloudbuild.builds.editor \
            roles/artifactregistry.writer roles/storage.admin ; do
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${DEPLOYER}" --role="$ROLE"
done

gcloud iam service-accounts add-iam-policy-binding $DEPLOYER --project=$PROJECT_ID \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/attribute.repository/${REPO}"
```

## Step 13.3 — The workflow file (written to disk, never pasted into a shell)

```bash
cd ~/dev/predictive-maint-adv
mkdir -p .github/workflows

cat > .github/workflows/deploy.yml << 'EOF'
name: Deploy predictive-maint
on:
  push:
    branches: [main]
permissions:
  contents: read
  id-token: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - id: auth
        uses: google-github-actions/auth@v3
        with:
          workload_identity_provider: 'projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github/providers/github-provider'
          service_account: 'github-deployer@predictive-maint-adv.iam.gserviceaccount.com'
      - uses: google-github-actions/setup-gcloud@v3
      - name: Rebuild trainer image
        run: |
          gcloud builds submit ./training \
            --tag europe-west3-docker.pkg.dev/predictive-maint-adv/maint/trainer:latest --quiet
      - name: Deploy scorer
        run: |
          gcloud run deploy scorer --source ./scorer --region europe-west3 \
            --set-env-vars PROJECT=predictive-maint-adv,ENDPOINT_NAME=maint-endpoint \
            --service-account scorer@predictive-maint-adv.iam.gserviceaccount.com \
            --no-allow-unauthenticated --quiet
      - name: Deploy retrainer job
        run: |
          gcloud run jobs deploy retrainer --source ./retrainer --region europe-west3 \
            --set-env-vars PROJECT=predictive-maint-adv,Z_THRESHOLD=1.0 \
            --service-account vertex-pipeline@predictive-maint-adv.iam.gserviceaccount.com \
            --quiet
EOF

sed -i '' "s/PROJECT_NUMBER/$PROJECT_NUMBER/" .github/workflows/deploy.yml
grep workload_identity_provider .github/workflows/deploy.yml
```

> **Why the retrainer is in CI too.** A Cloud Run *job* runs a frozen image and `gcloud run jobs execute` never rebuilds it (Phase 10). Without this step, a fix pushed to `retrainer/requirements.txt` — the `[pipelines]` extra, say — would sit in `main` looking deployed while the job kept running the broken image. Note it deploys from the **repo root** (`--source ./retrainer`), which is where `actions/checkout` leaves you. That's also the reason the copied-in files (`drift.sql`, `label.sql`, `maint_pipeline.json`) need to be committed, not just `cp`'d locally: CI builds from the repo, not from your Mac.

> **`sed -i ''` is macOS (BSD) syntax** — the empty `''` is a required "no backup file" argument there. On Linux, drop it: `sed -i "s/.../.../" ...`. The `grep` must now print your real project **number**, not the literal string `PROJECT_NUMBER`. If it doesn't, the auth step fails with a resource-not-found error that looks like a permissions problem and isn't.

> **⚠️ That heredoc block is a *file*, not commands.** Don't paste the YAML itself into the terminal — zsh will try to run `id:`, `uses:`, and `with:` as commands. The `cat > ... << 'EOF'` wrapper writes it to disk in one shot.

## Step 13.4 — Push

```bash
cd ~/dev/predictive-maint-adv

cat > .gitignore << 'EOF'
.venv/
.venv-beam/
__pycache__/
*.pyc
infra/.terraform/
*.tfstate
*.tfstate.*
*.tfvars
*-key.json
.DS_Store
EOF

git init && git branch -M main && git add .
git status                      # confirm NO *.tfstate, NO *-key.json
git commit -m "streaming predictive maintenance with drift-triggered retraining"
gh auth login
gh repo create predictive-maint-adv --public --source=. --remote=origin --push
```

> **If `gh repo create` says the repo already exists**, skip it and wire up the remote:
> ```bash
> git remote add origin https://github.com/$GITHUB_USER/predictive-maint-adv.git
> git push -u origin main
> ```

**Watch the deploy.** The push triggers the workflow immediately — repo → **Actions** tab → the **Deploy predictive-maint** run. Yellow (running) → green (done); click in for step logs if it fails. First runs commonly fail on a role the `github-deployer` SA is missing; the log names the exact permission.

✅ **Checkpoint 13:** push a trivial change → the Action runs green and redeploys the scorer.

---

# Phase 14 — Document the system (~2 hrs)

**Goal:** Make a stranger understand what the system does, why it is built this way, and how well it works — in one screen.

- **README** with the architecture diagram, one sentence per service (why Dataflow, why a conditional deploy gate, why drift-triggered retraining), the green pipeline-DAG screenshot, and honest monthly cost.
- **The headline numbers, with their caveats:**
  - the **BQML baseline vs the Vertex model** (`roc_auc` from `ML.EVALUATE` vs `auc` from `metrics.json`) — a comparison, not a lone number;
  - **PR-AUC as well as AUC**, because the classes are imbalanced;
  - the fact that you split **by machine, not randomly**, and why.
- **The three decisions that show judgment** — put these in prose, because they're what separate this from a tutorial:
  1. **Labels come from a ground-truth failure log, not from thresholding `vibration_mean`** — because `vibration_mean` is a model *input*, and defining the target from an input makes the target circular.
  2. **`reading_count` was deliberately excluded from the features** — it encoded the pipeline's ingestion rate, not machine health, and would have become a training-serving skew the moment the ingestion rate changed.
  3. **Windows whose label horizon hasn't elapsed are dropped** — they aren't negatives, they're unknowns, and training on them teaches the model that the most recently degraded machines are safe.
- **Known manual steps** — an honest account of what *isn't* codified: the WIF/deployer setup, the `run.builder` grant, the Cloud Scheduler job, and the endpoint (created by the pipeline, living outside Terraform).
- **"What I'd do next":** a Custom Prediction Routine so a real classifier can serve `predict_proba`; a pipeline-controlled holdout set for the evaluate step, and a gate that compares against the *currently deployed* model instead of a fixed constant; Vertex AI Model Monitoring instead of the SQL drift check; **RUL regression** (hours-to-failure, not a binary flag); a dead-letter topic on the Dataflow path.

**The one-paragraph summary** (useful as the README's opening):
> A streaming predictive-maintenance system on GCP — Pub/Sub telemetry into a Dataflow windowed-feature pipeline, labeled from a ground-truth failure log rather than by thresholding a model input, models trained and versioned through a Vertex AI Pipeline that only deploys when a new model clears an evaluation gate, real-time scoring on an online endpoint alongside a BQML batch path, and a drift check that rebuilds labels and automatically retrains the model — all Terraform-provisioned and shipped by keyless GitHub Actions.

✅ **Checkpoint 14:** a stranger reads the README and understands what the system does, why, how it deploys, and how well it works.

---

## Cost hygiene — what to turn off after every session

This project is **not** €0-idle. Three switches:

**1. The Dataflow streaming job** (the biggest ongoing cost)

```bash
gcloud dataflow jobs list --region europe-west3 --status active
gcloud dataflow jobs drain <JOB_ID> --region europe-west3
```

**2. The endpoint's deployed model** (an idle endpoint still bills for its node)

```bash
ENDPOINT_ID=$(gcloud ai endpoints list --region europe-west3 \
  --filter='displayName=maint-endpoint' --format='value(name)')

# There may be more than one deployed model if a retrain ever ran — undeploy ALL of them:
for DM in $(gcloud ai endpoints describe $ENDPOINT_ID --region europe-west3 \
              --format='value(deployedModels[].id)'); do
  gcloud ai endpoints undeploy-model $ENDPOINT_ID --region europe-west3 \
    --deployed-model-id "$DM" --quiet
done

gcloud ai endpoints describe $ENDPOINT_ID --region europe-west3 \
  --format='value(deployedModels[].id)'    # must print NOTHING
```

**3. The Cloud Scheduler retrain job** — the one people forget. Left running, it can detect drift after you've torn everything down and **submit a pipeline that creates a brand-new billing endpoint** while you aren't looking.

```bash
gcloud scheduler jobs pause retrain-check --location europe-west3
# later: gcloud scheduler jobs resume retrain-check --location europe-west3
```

Also stop the **generator** (Ctrl-C). It's free, but it keeps filling BigQuery and slowly shifts your drift baseline.

## Full teardown

The endpoint, the Dataflow job, and the Scheduler job all live **outside** Terraform, so kill them first:

```bash
# 1) drain Dataflow + undeploy all models (above), then:
gcloud scheduler jobs delete retrain-check --location europe-west3 --quiet
gcloud ai endpoints delete "$ENDPOINT_ID" --region europe-west3 --quiet

# 2) everything Terraform provisioned — including the SQL-created tables, views
#    and the BQML model, thanks to delete_contents_on_destroy on the dataset
cd ~/dev/predictive-maint-adv/infra && terraform destroy
```

Everything else — Cloud Run, Pub/Sub, BigQuery storage, one-shot training jobs, the pipeline when it isn't running — scales to zero or costs pennies. Rebuilding the whole thing for a demo takes ~30 minutes, and the training run is the long pole.
