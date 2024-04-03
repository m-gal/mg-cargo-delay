"""
    Script to make predictions for MLflow Model served locally.

* MLflow can deploy models locally as local REST API endpoints.
You deploy MLflow model locally.
For Keras models the REST API server accepts the following data formats
as POST input to the /invocations path:

Tensor input formatted as described in TF Serving"s API docs
where the provided inputs will be cast to Numpy arrays.
This format is specified using a Content-Type request header value
of application/json and the instances or inputs key in the request body dictionary.
https://mlflow.org/docs/latest/models.html#deploy-mlflow-models

* There are four input formats of inputs for deployed Mlflow Model:

# split-oriented DataFrame input:
input_data = {
    "dataframe_split": {
        "columns": ["a", "b", "c"],
        "data": [[1, 2, 3], [4, 5, 6]],
    }
}

# record-oriented DataFrame input:
input_data = {
    "dataframe_records": [{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}]
}

# * numpy/tensor input using TF serving's "instances" format:
# * only this format is aplicable for model deployed on GCP Vertex AI
input_data = {
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
            "vessel_name": ["MANULANI"]
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
            "vessel_name": ["TIANJIN"]
        }
    ]
}

# numpy/tensor input using TF serving's "inputs" format:
input_data = {
    "inputs": {
        "weight_kg": [850.0, 13255.1],
        "ade_month": ["FEB", "JAN"],
        "carrier_code": ["UASI", "ZIMU"],
        "container_id_prefix": ["MATU", "ZCSU"],
        "container_type_of_service": ["CS", "PP"],
        "place_of_receipt": ["SHANGHAI,CN", "NINGBO (ZJ)"],
        "port_of_lading": ["SHANGHAI", "NINGPO"],
        "port_of_unlading": ["LONG BEACH CA", "SAVANNAH GA"],
        "vessel_name": ["MANULANI", "TIANJIN"]
    }
}

To create the input_data sample files:
with open("test_sample.json", "w") as f:
    json.dump(input_data, f)

    @author: mikhail.galkin
"""

#%%
import requests
import json
import pandas as pd

MCLASSES = ["[00]", "[01-03]", "[04-07]", "[08-14]", "[15-21]", "[22+]"]
TARGETS = ["proba_delay", "dpd_delay"]
CSV_FILE_PATH = "prediction.csv"


def parse_args():
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(
        description="Get predictions from model deployed locally"
    )
    # Add the arguments
    parser.add_argument(
        "-p",
        "--port",
        type=str,
        default="8080",
        # required=True,
        help="local port for deployed model",
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


def main():
    # Parse arguments
    args = parse_args()
    port = args.port
    file = args.file

    url = f"http://127.0.0.1:{port}/invocations"
    print(f"Model deployed url: {url}")

    # Load data for prediction
    print(f"File with input data: {file}")
    with open(file, "r") as f:
        input_data = json.load(f)
    # Cast data into Tensor input format
    # (https://www.tensorflow.org/tfx/serving/api_rest#request_format_2)
    input_json = json.dumps(input_data)

    # Make prediction
    try:
        response = requests.post(
            url=url,
            data=input_json,
            headers={"Content-Type": "application/json"},
        )
        if response:
            print(f"---- Success! Model response is:")
            print(response.json())
    except Exception as ex:
        raise (ex)

    # Save prediction as pd.Dataframe into CSV
    prediction = dict(response.json())["predictions"]
    key = list(input_data.keys())[0]
    df_input = pd.DataFrame(input_data[key])
    df_prediction = pd.DataFrame(prediction, index=TARGETS).T
    df_classes = pd.DataFrame({"dpd_classes": [MCLASSES] * len(df_input)})
    df = pd.concat([df_input, df_prediction, df_classes], axis=1)
    df.to_csv(CSV_FILE_PATH)

    # Convert the input to a JSON string
    # df_input.to_json(orient="records")

    return df

#%% Run ------------------------------------------------------------------------
if __name__ == "__main__":
    _ = main()
