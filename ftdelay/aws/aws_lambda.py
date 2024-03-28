"""
    Script for AWS Lambda function.

    When a new Lambda Function w/ a new Role will have been created
    go to the IAM Management console, find the role and add next permissions:
        AWSLambdaExecute
        AWSLambdaBasicExecutionRole
        AWSLambdaRole

    If it will be spit error:
        (AccessDeniedException) when calling the InvokeEndpoint operation:......
    create as JSON a new inline policy named LambdaSagemakerInvokeEndpointPolicy
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "VisualEditor0",
                "Effect": "Allow",
                "Action": [
                    "lambda:InvokeAsync",
                    "lambda:InvokeFunction",
                    "sagemaker:InvokeEndpoint"
                ],
                "Resource": "*"
            }
        ]
    }

    @author: mikhail.galkin
"""

import boto3
import json


BUCKET_NAME = "ft-bol"
MCLASSES = ["[00]", "[01-03]", "[04-07]", "[08-14]", "[15-21]", "[22+]"]
TARGETS = ["y_be_delayed", "y_dpd_delayed"]

ENDPOINT_NAME = "ft-bol-us-delay"
runtime = boto3.client("runtime.sagemaker")


def lambda_handler(event, context):

    input_json = json.dumps(event)
    try:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=input_json,
        )
        prediction = response["Body"].read().decode()
        print(prediction)
    except Exception as e:
        raise IOError(e)

    prediction = json.loads(prediction)

    output = {
        "targets": {TARGETS[0]: [], TARGETS[1]: MCLASSES},
        "predictions": dict(zip(TARGETS, prediction)),
    }

    return {"statusCode": 200, "body": output}
