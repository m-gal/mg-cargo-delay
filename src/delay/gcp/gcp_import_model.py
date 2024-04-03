"""
    * Apply after gcp_upload_model.py

    Script to import uploaded model (into Google Cloud Storage)
    into Google Cloud Vertex AI.

    https://cloud.google.com/vertex-ai/docs/model-registry/import-model#aiplatform_upload_model_sample-python

    ^ The parameter serving_container_image_uri is used to specify
    which pre-built container we want to use for our model.
    You can see the list of available pre-built container in this link:
    https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers
    For example, if you want to use scikit-learn 1.0 pre-built container
    for Asia region, you will pass the parameter as
    serving_container_image_uri =
    'asia-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest'.

    @author: mikhail.galkin
"""

GCP_PROJECT_ID = "PROJECT_ID"
GCP_LOCATION = "LOCATION"
GCP_MODEL_DISPLAY_NAME = "ft-model-bol-us-delay"
GCP_MODEL_ARTIFACT_URI = "./bol-us-delay/v20230205/artifacts/model/data/model"
GCP_MODEL_DESCRIPTION = "BoL US Import Delivery Delay Risk prediction"
GCP_MODEL_SERVING_CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-9:latest"

# %% Load libraries
from google.cloud import aiplatform


def import_model_to_registry(
    project_id: str,
    location: str,
    model_display_name: str,
    description: str,
    serving_container_image_uri: str,
    artifact_uri: str,
    parent_model: str = None,
    is_default_version: bool = True,
    version_description: str = None,
    sync: bool = True,
):
    aiplatform.init(project=project_id, location=location)

    # Check if there is already a parent model
    filter = f"display_name={model_display_name}"
    models = aiplatform.Model.list(filter=filter)

    if len(models) == 0:
        model = aiplatform.Model.upload(
            project=project_id,
            location=location,
            display_name=model_display_name,
            description=description,
            serving_container_image_uri=serving_container_image_uri,
            artifact_uri=artifact_uri,
        )
    else:
        print("Importing model as a next version ...")
        parent_model = models[0].resource_name
        model = aiplatform.Model.upload(
            project=project_id,
            location=location,
            display_name=model_display_name,
            description=description,
            serving_container_image_uri=serving_container_image_uri,
            artifact_uri=artifact_uri,
            parent_model=parent_model,
            is_default_version=False,
        )

    model.wait()

    print(model.display_name)
    print(model.resource_name)
    return model


# %% ----------------------------------------------------------------------------
if __name__ == "__main__":
    model = import_model_to_registry(
        project_id=GCP_PROJECT_ID,
        location=GCP_LOCATION,
        model_display_name=GCP_MODEL_DISPLAY_NAME,
        description=GCP_MODEL_DESCRIPTION,
        serving_container_image_uri=GCP_MODEL_SERVING_CONTAINER_URI,
        artifact_uri=GCP_MODEL_ARTIFACT_URI,
    )
    print("Done.")
