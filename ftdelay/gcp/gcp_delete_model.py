"""
    * Apply after gsp_delete_endpoint.py

    Script to delete model from
    Google Vertex AI: Model Registry.

    @author: mikhail.galkin
"""

GCP_MODEL_ID = "1668110871940825088"
GCP_PROJECT_ID = "PROJECT_ID"
GCP_LOCATION = "LOCATION"

# %% Load libraries
from google.cloud import aiplatform


def delete_model(
    model_id: str,
    project_id: str,
    location: str = "LOCATION",
):
    """
    Delete a Model resource.
    Args:
        model_id: The ID of the model to delete.
            Parent resource name of the model is also accepted.
        project: The project.
        location: The region name.
    Returns
        None.
    """
    # Initialize the client.
    aiplatform.init(project=project_id, location=location)

    # Get the model with the ID 'model_id'. The parent_name of Model resource can be also
    # 'projects/<your-project-id>/locations/<your-region>/models/<your-model-id>'
    model = aiplatform.Model(model_name=model_id)

    # Delete the model.
    model.delete()


# %% ----------------------------------------------------------------------------
if __name__ == "__main__":
    delete_model(
        model_id=GCP_MODEL_ID,
        project_id=GCP_PROJECT_ID,
        location=GCP_LOCATION,
    )
    print("Done.")
