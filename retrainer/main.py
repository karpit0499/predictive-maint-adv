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