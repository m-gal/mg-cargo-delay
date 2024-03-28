"""
    Script to deploy MLFlow Model in the Docker Container on the AWS SageMaker.

    Run Terminal in the [./ftdelay/aws]
    > python aws_deploy.py

    @author: mikhail.galkin
"""

if __name__ == "__main__":
    import mlflow.sagemaker as mfs

    EXPERIMENT_ID = "8"
    RUN_ID = "5ba3fe1a46ac44c886c7debac90860a9"
    MLFLOW_TAG_ID = "1.28.0"
    AWS_ID = "AWS_ID "
    AWS_REGION = "AWS_REGION "
    AWS_ROLE = "AWS_ROLE"
    AWS_APP_NAME = "ft-bol-us-delay"
    AWS_BUCKET = "ft-bol-us-delay-sagemaker"

    model_uri = f"../../mlflow/mlruns/{EXPERIMENT_ID}/{RUN_ID}/artifacts/model"
    image_url = f"{AWS_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/mlflow-pyfunc:{MLFLOW_TAG_ID}"

    mfs.deploy(
        app_name=AWS_APP_NAME,
        bucket=AWS_BUCKET,
        region_name=AWS_REGION,
        execution_role_arn=AWS_ROLE,
        mode="create",
        model_uri=model_uri,
        image_url=image_url,
    )
