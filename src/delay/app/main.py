"""
    RESTful API with FastAPI package

    You can run the application server using
        >>> uvicorn app.main:app --reload
    Here app.main indicates you use main.py file inside the 'app' directory
    and :app indicates our FastAPI instance name.
            command: uvicorn {folder}.{module}:app --reload

    open Terminal in the [./src/delay] and run:
        >>> uvicorn app.main:app --reload
        or in the [./src/delay/app] and run:
            >>> uvicorn main:app --reload
    open http://127.0.0.1:8000/delay in browser
    open http://127.0.0.1:8000/docs in browser

    @author: mikhail.galkin
"""

# TODO [ ] Change the inputs formats to be consistent with GCP Vertex AI

"""
instances = [
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
    }
]

instances = [
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
"""

# %% Load libraries
import json
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import tensorflow as tf

# Model logged in TensorFlow flavor
MODEL_RUN_ID = "8/7e70881aaf294ca6ba6fe991ce60ee90"
MODEL_PATH = f"mlflow/mlruns/{MODEL_RUN_ID}/artifacts/model/data/model"

TARGETS = ["proba_delay", "dpd_delay"]
BINCLASSES = ["proba"]
MCLASSES = ["[00]", "[01-03]", "[04-07]", "[08-14]", "[15-21]", "[22+]"]


# Load model
print(f"\nLoad logged model as a TensorFlow Model...")
print(f"{MODEL_PATH}")
try:
    model = tf.keras.models.load_model("./././" + MODEL_PATH)
except:
    model = tf.keras.models.load_model("../../../" + MODEL_PATH)

# Initializing a FastAPI App Instance
app = FastAPI()


# Define request body for input data
class RequestBody(BaseModel):
    weight_kg: list[float] = []
    ade_month: list[str] = []
    carrier_code: list[str] = []
    container_id_prefix: list[str] = []
    container_type_of_service: list[str] = []
    place_of_receipt: list[str] = []
    port_of_lading: list[str] = []
    port_of_unlading: list[str] = []
    vessel_name: list[str] = []


# Define request body for input data
class Instance(BaseModel):
    weight_kg: list[float] = []
    ade_month: list[str] = []
    carrier_code: list[str] = []
    container_id_prefix: list[str] = []
    container_type_of_service: list[str] = []
    place_of_receipt: list[str] = []
    port_of_lading: list[str] = []
    port_of_unlading: list[str] = []
    vessel_name: list[str] = []


class InstanceList(BaseModel):
    __root__: List[Instance]

    def __iter__(self):
        return iter(self.__root__)

    def __getitem__(self, item):
        return self.__root__[item]


# Defining a Simple GET Request
@app.get("/us_delay")
def get_root():
    return {"Welcome": "US Import Delivery Delay Risk prediction"}


# Creating an Endpoint to recieve the data to make prediction on.
@app.post("/us_delay/predict")
async def get_prediction(instances: InstanceList):
    """
    FastAPI also validates the request body against the model we have defined
    and returns an appropriate error response.
    """

    input_instances = {
        "weight_kg": [],
        "ade_month": [],
        "carrier_code": [],
        "container_id_prefix": [],
        "container_type_of_service": [],
        "place_of_receipt": [],
        "port_of_lading": [],
        "port_of_unlading": [],
        "vessel_name": [],
    }

    # Validate data & create inputts for model
    for instance in instances:
        # print(instance)
        input_instances["weight_kg"].append(instance.weight_kg)
        input_instances["ade_month"].append(instance.ade_month)
        input_instances["carrier_code"].append(instance.carrier_code)
        input_instances["container_id_prefix"].append(instance.container_id_prefix)
        input_instances["container_type_of_service"].append(instance.container_type_of_service)
        input_instances["place_of_receipt"].append(instance.place_of_receipt)
        input_instances["port_of_lading"].append(instance.port_of_lading)
        input_instances["port_of_unlading"].append(instance.port_of_unlading)
        input_instances["vessel_name"].append(instance.vessel_name)

    # Cast input data into TensorFlow Dataset
    input_tensor = tf.data.Dataset.from_tensor_slices(input_instances).batch(8184)

    print(f"Predictioning ...")
    pred = model.predict(input_tensor)

    pred_tuples = list(zip(*pred))
    pred_list_t = [list(y.tolist() for y in x) for x in pred_tuples]
    target_dict = dict(zip(TARGETS, [["proba"], MCLASSES]))
    pred_dicts = [dict(zip(TARGETS, x)) for x in pred_list_t]

    # Get output dictionary
    output = {"target": target_dict, "prediction": pred_dicts}
    # print(output["prediction"])

    return output


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
