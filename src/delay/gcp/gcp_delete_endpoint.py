"""
    * Apply after gsp_undeploy_model.py

    Script to delete endpoint in the
    Google Vertex AI: Endpoints.

    @author: mikhail.galkin
"""

GCP_MODEL_ENDPOINT_ID = "ENDPOINT_ID"

# %% Load libraries
import sys
from google.cloud import aiplatform

sys.path.extend([".", "./.", "././.", "../..", "../../.."])
from src/delay.config import GCP_PROJECT_ID
from src/delay.config import GCP_LOCATION


def delete_endpoint(
    endpoint_id: str,
    project_id: str,
    location: str = "LOCATION",
    api_endpoint: str = "ENDPOINT",
    timeout: int = 300,
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.EndpointServiceClient(client_options=client_options)
    name = client.endpoint_path(project=project_id, location=location, endpoint=endpoint_id)
    response = client.delete_endpoint(name=name)
    print("Long running operation:", response.operation.name)
    delete_endpoint_response = response.result(timeout=timeout)
    print("delete_endpoint_response:", delete_endpoint_response)


# %% ----------------------------------------------------------------------------
if __name__ == "__main__":
    delete_endpoint(
        endpoint_id=GCP_MODEL_ENDPOINT_ID,
        project_id=GCP_PROJECT_ID,
        location=GCP_LOCATION,
    )
    print("Done.")
