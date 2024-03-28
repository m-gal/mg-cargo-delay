"""
    Script to make predictions using the model deployed on GCP Vertex AI.
"""

# %% GCP Model
GCP_MODEL_ENDPOINT_ID = "ENDPOINT_ID"
GCP_PROJECT_ID = "PROJECT_ID"
GCP_LOCATION = "LOCATION"

# %% Load libraries
import json
from typing import Dict, List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


def parse_args():
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(description="Get predictions from model deployed locally")
    # Add the arguments
    parser.add_argument(
        "-m",
        "--modelid",
        type=str,
        default=GCP_MODEL_ENDPOINT_ID,
        # required=True,
        help="deployed model's id",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="../../data/final/test_sample_1.json",
        help="file name with input data in json format",
    )
    # Parse arguments
    args, unknown = parser.parse_known_args()
    return args


# %% Function to upload folder on GCP Storage -----------------------------------
def predict_deployed_model(
    gsp_project_id: str,
    gcp_model_endpoint_id: str,
    input_data: Union[Dict, List[Dict]],
    gcp_location: str = "LOCATION",
    gcp_model_endpoint_api: str = "ENDPOINT",
):
    """
    `instances` can be either single instance of type dict or a list of instances.
    """
    print(f"send request to the {gcp_model_endpoint_id} endpoint")
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": gcp_model_endpoint_api}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    # The format of each instance should conform to the deployed model's prediction input schema.
    if type(input_data) == list:
        instances = input_data
    else:
        instances = input_data["instances"]

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
    prediction = []
    predictions = response.predictions
    for pred in predictions:
        print(" prediction:", dict(pred))
        prediction.append(dict(pred))

    # Get output dictionary
    output = {
        "target": {
            "proba_delay": ["proba"],
            "dpd_delay": ["[00]", "[01-03]", "[04-07]", "[08-14]", "[15-21]", "[22+]"],
        },
        "prediction": prediction,
    }
    return output


def main():
    # Parse arguments
    args = parse_args()
    modelid = args.modelid
    file = args.file

    # Load data for prediction
    print(f"File with input data: {file}")
    with open(file, "r") as f:
        input_data = json.load(f)
    # Cast data into Tensor input format
    # (https://www.tensorflow.org/tfx/serving/api_rest#request_format_2)
    # instances = json.dumps(input_data)

    output = predict_deployed_model(
        gsp_project_id=GCP_PROJECT_ID,
        gcp_model_endpoint_id=GCP_MODEL_ENDPOINT_ID,
        input_data=input_data,
    )

    return output


# %% ----------------------------------------------------------------------------
if __name__ == "__main__":
    output = main()
