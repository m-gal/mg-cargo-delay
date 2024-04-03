"""
    Script to upload trained model's folder with artifacts
    into Google Cloud Storage.

    @author: mikhail.galkin
"""

# TODO: Upload 'data' folder (train+test sets) inside model folder
# TODO: Check the gs:// paths correctness


MODEL_RUN_ID = "8/7e70881aaf294ca6ba6fe991ce60ee90"
LOCAL_MODEL_FOLDER_PATH = f"{mlflow_tracking_uri}/{MODEL_RUN_ID}"

GCP_MODEL_NAME = "v"
GCP_MODEL_TIMESTAMP = "20230205"
GCP_MODEL_FOLDER_NAME = f"bol-us-delay/{GCP_MODEL_NAME}{GCP_MODEL_TIMESTAMP}"
GCP_MODEL_BUCKET_NAME = "ft-model/bol-us-delay"


#%% Load libraries -------------------------------------------------------------
import sys
from pathlib import Path
from google.cloud import storage

sys.path.extend([".", "./.", "././.", "../..", "../../.."])
from src/delay.config import mlflow_tracking_uri

# Instantiates a client
storage_client = storage.Client()
# print(f"Google Cloud Storage buckets:")
# buckets = list(storage_client.list_buckets())
# for bucket in buckets:
#     print(bucket)

#%% Function to upload folder on GCP Storage -----------------------------------
def upload_folder_to_gcs(
    local_folder_path: str,
    gcp_folder_name: str,
    gcp_bucket_name: str = "ft-model",
):
    """Upload local model's folder & files to GCP bucket.

    Args:
        local_folder_path (str): The absolute path to your folder to upload
        gcp_folder_name (str): The model's folder name on GCP bucket
        gcp_bucket_name (str, optional): The  GCP bucket's name. Defaults to "ft-bol-us-delay-model".
    """

    local_folder_path = Path(local_folder_path)

    local_folder_name = local_folder_path.name
    local_files = [f for f in local_folder_path.rglob("*") if f.is_file()]
    # File names on gcs: use .as_posix() for matching GCStorage path
    blob_files = [f.as_posix().split(local_folder_name)[1] for f in local_files]
    bucket = storage_client.get_bucket(gcp_bucket_name)

    print(f"Uploading files to gs://{gcp_bucket_name}:")
    for local_file, blob_file in zip(local_files, blob_files):
        blob_path = f"{gcp_folder_name}{blob_file}"
        blob = bucket.blob(blob_path)
        print(f"\tUploading: {blob_path} ...")
        blob.upload_from_filename(local_file)
    print(f"\nSuccessfully uploaded.")

#%% ----------------------------------------------------------------------------
if __name__ == "__main__":
    upload_folder_to_gcs(
        LOCAL_MODEL_FOLDER_PATH,
        GCP_MODEL_FOLDER_NAME,
    )
    print("Done.")
