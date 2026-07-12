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
  depends_on                  = [google_project_service.apis]
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
locals {
  df_roles = [
    "roles/dataflow.worker",
    "roles/pubsub.subscriber",     # consume messages
    "roles/pubsub.viewer",         # READ the subscription's config — Beam calls GetSubscription
    "roles/bigquery.dataEditor",   # Dataflow only streams rows in — never runs a query
    "roles/storage.objectAdmin",
  ]
  pipeline_roles = [
    "roles/aiplatform.user",
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/bigquery.readSessionUser",
    "roles/storage.objectAdmin",
    "roles/artifactregistry.reader",
    "roles/logging.logWriter",      # the training worker ships stdout to Cloud Logging as
                                    # THIS SA. Without it the container's output — including
                                    # its traceback — is dropped, and a crash surfaces as a
                                    # bare "exited with non-zero status of 1" and nothing else.
  ]
  scorer_roles = [
    "roles/aiplatform.user",
    "roles/bigquery.dataEditor",   # the scorer only streams inserts — no queries
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