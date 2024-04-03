"""
    @author: mikhail.galkin
"""

# %% Import libraries
from pathlib import Path


# %% Define the project's paths -------------------------------------------------
project_path = Path(__file__).parent.resolve()
project_dir = project_path.parent.resolve()

data_analyze_dir = project_dir / "data/analyze"
data_external_dir = project_dir / "data/external"
data_final_dir = project_dir / "data/final"
data_processed_dir = project_dir / "data/processed"
data_raw_dir = project_dir / "data/raw"

docs_dir = project_dir / "docs"
models_dir = project_dir / "models"
temp_dir = project_dir / "temp"
tensorboard_dir = project_dir / "tensorboard"

mlflow_dir = project_dir / "mlflow"
mlflow_tracking_uri = project_dir / "mlflow/mlruns"

processed_data_folder_path = Path("z:/fishtailS3/ft-bol-data/us/data/processed/xpm")
final_data_folder_path = Path("z:/fishtailS3/ft-model/bol-us-delay/data")
reports__folder_path = Path("z:/fishtailS3/ft-bol/us/reports")

# %% Training (Trained) model stuff ---------------------------------------------
DATA_PERIOD = "201901-202212"
LOGGED_MODEL = "mlruns/8/7e70881aaf294ca6ba6fe991ce60ee90/artifacts/model"

# %% Google Cloud Platform stuff ------------------------------------------------
GCP_LOCATION = ""
GCP_PROJECT_ID = ""
GCP_PROJECT_NAME = ""

GCP_MODEL_BUCKET_NAME = ""
GCP_MODEL_ARTIFACT_URI = ""
GCP_MODEL_DISPLAY_NAME = ""
GCP_MODEL_DESCRIPTION = ""
GCP_MODEL_ENDPOINT_API = ""
GCP_MODEL_ENDPOINT_NAME = ""
GCP_MODEL_SERVING_CONTAINER_URI = ""
