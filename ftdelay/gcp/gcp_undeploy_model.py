"""
    * Apply first

    Script to undeploy model from the endpoint in the
    Google Vertex AI: Endpoints.

    @author: mikhail.galkin
"""

GCP_MODEL_ENDPOINT_NAME = "ft-bol-us-delay-model"
GCP_PROJECT_ID = "PROJECT_ID"
GCP_LOCATION = "LOCATION"

# %% Load libraries
from google.cloud import aiplatform


def undeploy_model_from_endpoint(
    project_id: str,
    location: str,
    model_endpoint_name: str,
):
    aiplatform.init(project=project_id, location=location)

    print(f"Get the Endpoint resource ...")
    filter = f"display_name={model_endpoint_name}"
    endpoints = aiplatform.Endpoint.list(filter=filter)
    print("Number of endpoints:", len(endpoints))
    print("Endpoint name:", endpoints[0].display_name)
    print("Endpoint resource name:", endpoints[0].resource_name)
    print("Endpoint was created:", endpoints[0].create_time)
    endpoint = endpoints[0]

    print(f"\nGet info for deployed models ...")
    deployed_models = endpoint.list_models()
    deployed_model_id = deployed_models[0].id
    model_display_name = deployed_models[0].display_name
    print("Number of deployed models:", len(deployed_models))
    print("Model ID:", deployed_model_id)
    print("Model name:", model_display_name)
    print("Model resource name:", deployed_models[0].model)
    print("Model was created:", deployed_models[0].create_time)

    print(f"\nUndeploy model '{model_display_name}' from endpoint ...")
    endpoint.undeploy(deployed_model_id=deployed_model_id)


# %% ----------------------------------------------------------------------------
if __name__ == "__main__":
    undeploy_model_from_endpoint(
        project_id=GCP_PROJECT_ID,
        location=GCP_LOCATION,
        model_endpoint_name=GCP_MODEL_ENDPOINT_NAME,
    )
    print("Done.")
