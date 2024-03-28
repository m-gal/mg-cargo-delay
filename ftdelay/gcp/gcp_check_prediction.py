"""
    Stub to do checking one-off prediction using the model deployed on GCP Vertex AI.
"""

# %% Test samples
# We can use only numpy/tensor input using TF serving's "instances" format:
test_sample = {
    "instances": [
        {
            "weight_kg": [850.0],
            "ade_month": ["FEB"],
            "carrier_code": ["UASI"],
            "container_id_prefix": ["MATU"],
            "container_type_of_service": ["CS"],
            "place_of_receipt": ["SHANGHAI,CN"],
            "port_of_lading": ["SHANGHAI"],
            "port_of_unlading": ["LONG BEACH CA"],
            "vessel_name": ["MANULANI"],
        },
        {
            "weight_kg": [13255.1],
            "ade_month": ["JAN"],
            "carrier_code": ["ZIMU"],
            "container_id_prefix": ["ZCSU"],
            "container_type_of_service": ["PP"],
            "place_of_receipt": ["NINGBO (ZJ)"],
            "port_of_lading": ["NINGPO"],
            "port_of_unlading": ["SAVANNAH GA"],
            "vessel_name": ["TIANJIN"],
        },
    ]
}

# ## or it might be a list of instances:
# test_sample = [
#     {
#         "weight_kg": [850.0],
#         "ade_month": ["FEB"],
#         "carrier_code": ["UASI"],
#         "container_id_prefix": ["MATU"],
#         "container_type_of_service": ["CS"],
#         "place_of_receipt": ["SHANGHAI,CN"],
#         "port_of_lading": ["SHANGHAI"],
#         "port_of_unlading": ["LONG BEACH CA"],
#         "vessel_name": ["MANULANI"],
#     },
#     {
#         "weight_kg": [13255.1],
#         "ade_month": ["JAN"],
#         "carrier_code": ["ZIMU"],
#         "container_id_prefix": ["ZCSU"],
#         "container_type_of_service": ["PP"],
#         "place_of_receipt": ["NINGBO (ZJ)"],
#         "port_of_lading": ["NINGPO"],
#         "port_of_unlading": ["SAVANNAH GA"],
#         "vessel_name": ["TIANJIN"],
#     },
# ]

# %% GCP Model
GCP_MODEL_ENDPOINT_ID = "ENDPOINT_ID"
GCP_PROJECT_ID = "PROJECT_ID"
GCP_LOCATION = "LOCATION"

# %% Load libraries
from typing import Dict, List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


def predict_deployed_model(
    gsp_project_id: str,
    gcp_model_endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    gcp_location: str = "LOCATION",
    gcp_model_endpoint_api: str = "ENDPOINT",
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    print(f"send request to the {gcp_model_endpoint_id} endpoint")
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": gcp_model_endpoint_api}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = instances if type(instances) == list else instances["instances"]
    instances = [json_format.ParseDict(instance_dict, Value()) for instance_dict in instances]

    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=gsp_project_id,
        location=gcp_location,
        endpoint=gcp_model_endpoint_id,
    )
    response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))


# %% ----------------------------------------------------------------------------
if __name__ == "__main__":
    predict_deployed_model(
        gsp_project_id=GCP_PROJECT_ID,
        gcp_model_endpoint_id=GCP_MODEL_ENDPOINT_ID,
        instances=test_sample,
    )
