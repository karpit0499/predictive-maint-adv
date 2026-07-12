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