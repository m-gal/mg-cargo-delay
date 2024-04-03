"""
    * Apply after gcp_import_model.py

    Script to deploy to endpoint the model which have been imported
    into Google Cloud Vertex AI Model Registry.

    @author: mikhail.galkin
"""

GCP_PROJECT_ID = "PROJECT_ID"
GCP_LOCATION = "LOCATION"
GCP_MODEL_DISPLAY_NAME = "ft-model-bol-us-delay"
GCP_MODEL_ENDPOINT_NAME = "ft-model-bol-us-delay"

# %% Load libraries
from google.cloud import aiplatform


def deploy_model_to_endpoint(
    project_id: str,
    location: str,
    model_display_name: str,
    model_endpoint_name: str,
    machine_type: str = "n1-standard-2",
):
    aiplatform.init(project=project_id, location=location)

    print(f"Get the Model resource ...")
    filter = f"display_name={model_display_name}"
    models = aiplatform.Model.list(filter=filter)
    print("Number of models:", len(models))
    print("Version ID:", models[0].version_id)
    print("Model name:", models[0].display_name)
    print("Model name:", models[0].description)
    model = models[0]

    print("Creating an Endpoint resource ...")
    endpoint = aiplatform.Endpoint.create(
        display_name=model_endpoint_name,
        project=project_id,
        location=location,
    )

    print("Deploying a single Endpoint resource ...")
    response = endpoint.deploy(
        model=model,
        deployed_model_display_name=model_display_name,
        machine_type=machine_type,
        traffic_percentage=100,
    )

    return (model, endpoint, response)


# %% ----------------------------------------------------------------------------
if __name__ == "__main__":
    _ = deploy_model_to_endpoint(
        project_id=GCP_PROJECT_ID,
        location=GCP_LOCATION,
        model_display_name=GCP_MODEL_DISPLAY_NAME,
        model_endpoint_name=GCP_MODEL_ENDPOINT_NAME,
    )
    print("Done.")
