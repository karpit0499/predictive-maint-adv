# predictive-maint-adv — Streaming Predictive Maintenance with Continuous Training on GCP

An end-to-end, production-style ML system on Google Cloud. A simulated fleet of machines emits sensor telemetry; readings stream through **Pub/Sub**; **Dataflow (Apache Beam)** turns the unbounded stream into windowed features in flight; failures are recorded as **ground-truth events**; a **Vertex AI Pipeline (KFP v2)** trains a failure-risk model, evaluates it, and *conditionally* deploys it behind an online endpoint; a Cloud Run service scores machines in real time while BigQuery ML scores the fleet in batch; and a scheduled **drift check rebuilds the labels and retrains the model automatically** when the live sensor distribution moves away from the training distribution.

Everything is provisioned with **Terraform** and shipped by **GitHub Actions using keyless Workload Identity Federation**.

The thing that makes this more than a tutorial is the last loop: **continuous training (CT)**, not just CI/CD. The model is retrained on *data* change, not only redeployed on *code* change.

📊 **Dashboard:** Fleet Health & Model Ops, built in Looker Studio on the dbt marts (see Phase 12 of the build guide).

---

## Architecture

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
```

## Why each service

| Service | Why it's here |
|---|---|
| **Pub/Sub** | Decouples the sensor fleet from everything downstream. Producers never block on consumers; a stalled pipeline buffers instead of dropping data. |
| **Dataflow / Apache Beam** | Windowed aggregation of an *unbounded* stream. Features are computed in flight, once, rather than re-derived by every downstream query. Fixed 2-minute windows per machine. |
| **BigQuery** | One warehouse for the raw stream, the windowed features, the ground-truth failure log, the predictions, and the drift log. Also runs the BQML baseline in-place — no data movement. |
| **BigQuery ML** | A boosted-tree baseline in one `CREATE MODEL` statement. It exists so the custom model has something honest to be compared against, and it keeps scoring the fleet even when the paid endpoint is undeployed. |
| **Vertex AI custom training + Model Registry** | A portable, versioned model artifact. Every training run mints a new version; deploys and rollbacks reference versions. |
| **Vertex AI Pipelines (KFP v2)** | Train → evaluate → **conditional deploy**. The `dsl.If` gate is the guardrail that stops a bad retrain from silently replacing a good model. |
| **Vertex AI online endpoint** | Low-latency scoring for the one machine someone is asking about right now. |
| **Cloud Run (service)** | The scoring API in front of the endpoint: validates the feature vector, clamps the score, writes provenance to BigQuery, emits the alerting log line. |
| **Cloud Run (job) + Cloud Scheduler** | The drift check. A job runs to completion and exits — the right shape for a periodic task. |
| **dbt** | Turns raw predictions and drift logs into tested, versioned marts. The value isn't the SQL; it's that the SQL is reviewable and has assertions attached. |
| **Looker Studio** | One screen with the operational story (which machines need attention) beside the ML story (has the model drifted, when did it last retrain). |
| **Terraform** | The whole substrate — APIs, Pub/Sub, GCS, BigQuery, service accounts, IAM — declared once and destroyable in one command. |
| **GitHub Actions + WIF** | Deploys on push with **no service-account key anywhere**. GCP trusts GitHub's OIDC token, scoped to this exact repository. |

## Results

| Model | Split | AUC | PR-AUC |
|---|---|---|---|
| BigQuery ML boosted-tree classifier (baseline) | BQML default | `roc_auc` from `ML.EVALUATE` | — |
| Vertex AI gradient-boosted regressor (custom) | **Machine-disjoint holdout** | ~0.93–0.96 | ~0.85–0.90 |

Read the caveats before you read the numbers:

- **PR-AUC matters more than AUC here.** The positive class is ~20% of windows. AUC flatters imbalanced problems; average precision does not.
- **The split is by machine, not random.** Two windows two minutes apart from the same machine are near-duplicates. A random split puts one in train and its twin in test and scores the model on rows it has effectively already seen. Whole machines are held out, so the metric answers the question that actually matters: *does this generalize to a machine the model has never met?*
- **Two recall numbers are reported, at 0.5 and at a tuned threshold.** The deployed model is a *regressor* fit to a 0/1 target, so its output is a risk score, not a calibrated probability, and 0.5 has no special meaning to it. The honest operating point flags the riskiest windows at the base failure rate.
- The baseline is not decoration. A single accuracy number with nothing to compare it to is not a result.

## The three decisions that carry this project

**1. Labels come from a ground-truth failure log — never from thresholding a sensor reading.**
The tempting shortcut is to define failure as "vibration crossed a threshold" and label the windows before it. But `vibration_mean` is a *model input*. Defining the target by thresholding an input makes the target a near-deterministic function of that input, and the model learns "is vibration close to the threshold?" — a tautology dressed up as a prediction. The generator writes a row to `failure_events` every time a machine actually dies, exactly as a real maintenance system would, and labels are derived from that table alone.

**2. `reading_count` is deliberately excluded from the feature set.**
It looks like a feature. It is the same value in every row, because it counts `window_length ÷ publish_interval`. What it actually encodes is *the ingestion rate of the pipeline*, not the health of any machine. Train on it and nothing bad happens — until the day a worker hiccups and a window lands short, and a column the model relies on shifts under it at serving time for reasons that have nothing to do with a machine. It stays in the feature table as a data-quality signal and out of the model.

**3. Windows whose label horizon hasn't elapsed yet are dropped, not labeled zero.**
A window from four minutes ago cannot yet be known to be a negative — its 10-minute horizon hasn't finished. Labeling it `0` teaches the model that the most recent, most-degraded windows are safe, which is precisely backwards. Those rows are *censored* (excluded), which is why the labeled table always lags the live one by one full horizon.

## Repository layout

```
infra/                 Terraform: APIs, Pub/Sub, GCS, BigQuery, service accounts, IAM
generator/             Synthetic fleet — publishes telemetry, logs real failures as ground truth
dataflow/              Apache Beam streaming job: raw passthrough + 2-min windowed features
sql/
  label.sql            Horizon labeling from failure_events, with censoring
  drift.sql            Standardized mean shift (effect size) of the live sensor distribution
training/              train.py, Dockerfile, run_training.py — custom training on Vertex AI
pipeline/              KFP v2 pipeline: train → evaluate → conditional deploy
scorer/                Cloud Run service: real-time scoring in front of the Vertex endpoint
retrainer/             Cloud Run job: drift check → rebuild labels → resubmit the pipeline
maint_dbt/             dbt models + tests → staging and mart layers
.github/workflows/     Keyless CI/CD (Workload Identity Federation)
BUILD_GUIDE.md         Step-by-step build from an empty GCP project
TECHNICAL_DOCUMENTATION.md   Architecture, data contracts, IAM, failure modes, runbook
```

## Quick start

Full instructions, including every gotcha worth knowing, are in **[BUILD_GUIDE.md](BUILD_GUIDE.md)**. The short version:

```bash
# 1. Provision
cd infra && terraform init && terraform apply

# 2. Start the fleet (leave running)
PROJECT=<your-project-id> python generator/generate.py

# 3. Start the streaming feature pipeline
python dataflow/stream_features.py --project_id <your-project-id> \
  --subscription projects/<your-project-id>/subscriptions/telemetry-df-sub \
  --runner DataflowRunner --project <your-project-id> --region <your-region> ...

# 4. Label, baseline, train, deploy
bq query --use_legacy_sql=false < sql/label.sql
python training/run_training.py
python pipeline/maint_pipeline.py && python pipeline/run_pipeline.py

# 5. Analytics
cd maint_dbt && dbt build
```

## Cost

**This project is not zero-cost while idle.** Two resources bill continuously while switched on:

| Resource | Behaviour |
|---|---|
| Dataflow streaming job | Holds at least one worker VM for its entire life. Does **not** scale to zero. |
| Vertex AI online endpoint | Holds one node per deployed model. Does **not** scale to zero. |

Everything else — Cloud Run, Pub/Sub, BigQuery storage, one-shot training jobs, the pipeline when it isn't running — scales to zero or costs pennies. The discipline is: run them for a session, then drain the Dataflow job and undeploy the endpoint. A third switch is easy to forget and can quietly re-create the endpoint after teardown — the Cloud Scheduler drift job. All three off-switches are in the build guide's cost-hygiene section. Rebuilding the whole system for a demo takes about half an hour.

## Known manual steps

Honest accounting of what is *not* codified in Terraform:

- The Workload Identity Federation pool, provider, and deployer service account (bootstrapped by CLI — they're the thing that lets CI run at all).
- The Cloud Build / Cloud Run builder role grant to the Compute Engine default service account.
- The Cloud Scheduler job that triggers the drift check.
- The Vertex AI **endpoint** — created by the pipeline at deploy time, so it lives outside Terraform's state by design.
- BigQuery scheduled queries (label rebuild, batch scoring) and the log-based metric + alert policy.

## What I'd do next

- **A Custom Prediction Routine** (or a bespoke serving container) so a real classifier can expose `predict_proba`, instead of fitting a regressor to a 0/1 target to get a continuous score out of the prebuilt container's `predict()`.
- **A pipeline-controlled holdout set** for the evaluate step, rather than the training job's own test split — and a deploy gate that compares the candidate against the *currently deployed* model instead of a fixed constant.
- **Vertex AI Model Monitoring** in place of the hand-rolled SQL drift check, and a baseline pinned to the exact data snapshot the deployed model was trained on rather than "the oldest rows we happen to have."
- **RUL regression** — predict hours-to-failure rather than a binary "will it fail in the horizon" flag.
- **A dead-letter topic** on the Dataflow path, so malformed telemetry is quarantined rather than silently dropped.
- **Batch-side alerting** ("more than N machines above threshold"), which the current log-based metric doesn't cover.

## License

[MIT](LICENSE).
