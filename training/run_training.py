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