"""
    Script to make predictions with model deployed on AWS SageMaker.

    Run Terminal in the [./src/delay/aws]
    > python aws_predict.py -f test_sample_2.json
        -f, --file : file name in the [./predict] folder with the input data saved in json format

    @author: mikhail.galkin
"""

import boto3
import json
import pandas as pd


AWS_APP_NAME = "ft-bol-us-delay"
AWS_REGION = "AWS_REGION "
MCLASSES = ["[00]", "[01-03]", "[04-07]", "[08-14]", "[15-21]", "[22+]"]
TARGETS = ["y_be_delayed", "y_dpd_delayed"]
CSV_FILE_NAME = "aws_prediction.csv"
TEST_SAMPLE_DIR = "../../data/final/"


def parse_args():
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(
        description="Get predictions from model deployed AWS SageMaker"
    )
    # Add the arguments
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="test_sample_1.json",
        help="file name with input data in json format",
    )
    # Parse arguments
    args, unknown = parser.parse_known_args()
    return args


def check_status(app_name, region):
    sm = boto3.client("sagemaker", region_name=region)
    endpoint_description = sm.describe_endpoint(EndpointName=app_name)
    endpoint_status = endpoint_description["EndpointStatus"]
    return endpoint_status


def query_endpoint(app_name, region, input_json):
    runtime = boto3.client("runtime.sagemaker", region_name=region)

    try:
        response = runtime.invoke_endpoint(
            EndpointName=app_name,
            ContentType="application/json",
            Body=input_json,
        )
        if response:
            print(f"\n---- Success! Application response is:")
            prediction = response["Body"].read().decode("ascii")
            print(prediction)
    except Exception as ex:
        raise (ex)

    prediction = json.loads(prediction)
    return prediction


def main():
    # Parse arguments
    args = parse_args()
    file = "../../data/final/" + args.file

    # Load data for prediction
    print(f"File with input data: {file}")
    with open(file, "r") as f:
        input_data = json.load(f)
    # Cast data into Tensor input format
    input_json = json.dumps(input_data)

    # Check endpoint status
    status = check_status(AWS_APP_NAME, AWS_REGION)
    print(f"Application status is: {status}")

    # Make prediction
    prediction = query_endpoint(AWS_APP_NAME, AWS_REGION, input_json)

    # Save prediction as pd.Dataframe into CSV
    df_input = pd.DataFrame(input_data["inputs"])
    df_prediction = pd.DataFrame(prediction, index=TARGETS).T
    df_classes = pd.DataFrame({"dpd_classes": [MCLASSES] * len(df_input)})
    df = pd.concat([df_input, df_prediction, df_classes], axis=1)
    df.to_csv(CSV_FILE_NAME)


# %% Run ------------------------------------------------------------------------
if __name__ == "__main__":
    main()
