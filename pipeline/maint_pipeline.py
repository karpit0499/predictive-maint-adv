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