import logging
import os
import sys
from datetime import datetime, timezone

from flask import Flask, request
from google.cloud import aiplatform, bigquery

# Cloud Run captures whatever the container writes to stdout and ships it to Cloud
# Logging as textPayload. Two things have to be true for that to work:
#   1. stdout must be unbuffered (see PYTHONUNBUFFERED=1 in the Dockerfile).
#   2. We must write PLAIN TEXT to stdout — not use google-cloud-logging's
#      setup_logging(), which would move the message into jsonPayload.message and
#      break the Phase 9 log-based metric filter (textPayload=~"HIGH_RISK").
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
        _endpoint = eps[0]
        dms = _endpoint.gca_resource.deployed_models
        if not dms:
            raise RuntimeError(f"endpoint {ENDPOINT_NAME} exists but has no model deployed "
                               f"— did you undeploy it for cost reasons?")
        # Record WHICH MODEL produced each score. Logging the endpoint's name
        # here (as an earlier version of this guide did) tells you nothing when
        # you later ask "which model version made this prediction?"
        _version = f"{dms[0].model.split('/')[-1]}@{dms[0].model_version_id or '1'}"
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
        # Phase 9's log-based metric counts this exact line. Plain stdout, no
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