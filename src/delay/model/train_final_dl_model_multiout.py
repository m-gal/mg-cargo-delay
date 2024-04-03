"""
    Train a final DL model with 2 outputs:
    - for binary ckassification
    - for multiclass classification.

    @author: mikhail.galkin
"""
#### WARNING: the code that follows would make you cry.
#### A Safety Pig is provided below for your benefit
#                            _
#    _._ _..._ .-',     _.._(`))
#   '-. `     '  /-._.-'    ',/
#      )         \            '.
#     / _    _    |             \
#    |  a    a    /              |
#    \   .-.                     ;
#     '-('' ).-'       ,'       ;
#        '-;           |      .'
#           \           \    /
#           | 7  .__  _.-\   \
#           | |  |  ``/  /`  /
#          /,_|  |   /,_/   /
#            /,_/      '`-'

#%% Setup ----------------------------------------------------------------------
import sys
import time

import json
import inspect  # for model inspection
import mlflow
import numpy as np
import pandas as pd

from sklearn import model_selection
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from pprint import pprint
from IPython.display import display

#%% Setup the Tensorflow -------------------------------------------------------
import tensorflow as tf

print("tensorflow ver.:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        cuda = tf.test.is_built_with_cuda()
        gpu_support = tf.test.is_built_with_gpu_support()
        print(
            f"\tPhysical GPUs: {len(gpus)}\n\tLogical GPUs: {len(logical_gpus)}"
        )
        print(f"\tIs built with GPU support: {gpu_support}")
        print(f"\tIs built with CUDA: {cuda}")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

#%% Load project's stuff -------------------------------------------------------
sys.path.extend([".", "./.", "././.", "../..", "../../.."])

from src/delay.config import DATA_PERIOD
from src/delay.config import final_data_folder_path
from src/delay.config import data_final_dir
from src/delay.config import docs_dir
from src/delay.config import mlflow_tracking_uri
from src/delay.config import tensorboard_dir
from src/delay.config import temp_dir

from src/delay.utils import timing
from src/delay.model._dl_utils import df_to_ds
from src/delay.model._dl_utils import create_model_inputs_and_features

# from src/delay.model.dl_models import vanilla_multioutput
from src/delay.model._dl_models import looped_multioutput

from src/delay.plots import plot_history_metrics
from src/delay.plots import plot_confusion_matrix
from src/delay.plots import plot_density
from src/delay.plots import plot_3roc
from src/delay.plots import plot_multiclass_roc

#%% Custom funcs ---------------------------------------------------------------
def save_df_test(tuple_of_dfs, path_to_save):
    df_test = tuple_of_dfs[1]
    print(f"Save test data as CSV.GZ: {path_to_save}")
    tic = time.time()
    df_test.to_csv(
        path_to_save,
        index=False,
        encoding="utf-8-sig",
        compression="gzip",
    )
    print(f"Test data saved {timing(tic)}")
    return df_test


#%% MAIN -----------------------------------------------------------------------
def main(
    data_income_file_name,
    json_features_file_name,
    mlflow_experiment_name,
):
    print("mlflow ver.:", mlflow.__version__)

    RND_STATE = 42
    MLFLOW_EXPERIMENT_NAME = mlflow_experiment_name

    # ! Metadata for incoming data ---------------------------------------------
    DATA_INCOME_FOLDER_PATH = final_data_folder_path
    JSON_INCOME_FOLDER_PATH = data_final_dir
    DATA_INCOME_FILE_NAME = data_income_file_name
    JSON_FEATURES_FILE_NAME = json_features_file_name
    path_to_data = DATA_INCOME_FOLDER_PATH / DATA_INCOME_FILE_NAME
    path_to_json = JSON_INCOME_FOLDER_PATH / JSON_FEATURES_FILE_NAME

    COLS_TO_READ = [
        "report_month",
        "ade_month",
        "carrier_code",
        "container_id_prefix",
        "container_type_of_service",
        "place_of_receipt",
        "port_of_lading",
        "port_of_unlading",
        "vessel_name",
        "weight_kg",
        "y_be_delayed",
        "y_dpd_delayed",
    ]

    # ! Metadata for the neural net --------------------------------------------
    PARAMS = {
        "_batch_size": 8184,
        "_neurons": "1024-512-512",
        "_epochs": 8,
        "_optimizer": tf.keras.optimizers.RMSprop(learning_rate=0.001),
        "_loss": {
            "proba_delay": tf.keras.losses.BinaryCrossentropy(),
            "dpd_delay": tf.keras.losses.CategoricalCrossentropy(),
        },
        "_metrics": {
            "proba_delay": tf.keras.metrics.BinaryAccuracy(name="bin_acc"),
            "dpd_delay": tf.keras.metrics.Accuracy(name="acc"),
        },
    }

    print(f"\nRead data from {path_to_data} \n\t {path_to_json}")
    df = pd.read_csv(path_to_data, usecols=None)
    # Get the dictionary of all the input features for NN.
    with open(path_to_json, "r") as f:
        set_of_features = json.load(f)

    # Get the targets features & classes
    y_01 = set_of_features["type_y"]["binary"][0]
    y_mc = list(set_of_features["type_y"]["multiclass"].keys())[0]
    vocab = set_of_features["type_y"]["multiclass"].get(y_mc)

    # Encode the target
    print(f"One-Hot Encoding the multiclass target: <{y_mc}>")
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(vocab)

    y_mc_ohe = label_binarizer.transform(df[y_mc])
    df[y_mc] = pd.Series(y_mc_ohe.tolist())

    # Split the data with stratified target
    print(f"\nSplit whole data on train & test sets ...")
    df_train, df_test = model_selection.train_test_split(
        df,
        train_size=0.99,
        random_state=RND_STATE,
        shuffle=True,
        stratify=df[y_mc],
    )
    print(f"Dev set: {df.shape[1]} vars & {df.shape[0]:,} rows.")
    print(f"\tdf_train: {df_train.shape[1]} vars & {df_train.shape[0]:,} rows.")
    print(f"\tdf_test: {df_test.shape[1]} vars & {df_test.shape[0]:,} rows.")

    # Convert pd.DataFrame to tf.Dataset
    #! Do not use the shuffle=True! It leads to wrong accuracy on the test set
    ds_train, y_true_train = df_to_ds(
        df_train,
        set_of_features,
        PARAMS["_batch_size"],
        shuffle=False,
    )
    ds_test, y_true_test = df_to_ds(
        df_test,
        set_of_features,
        PARAMS["_batch_size"],
        shuffle=False,
    )

    y_true_train_binary = y_true_train[0]
    y_true_test_binary = y_true_test[0]

    y_true_train_mclass = y_true_train[1]
    y_true_test_mclass = y_true_test[1]

    print(f"\nInspect the train dataset's elements ...")
    pprint(ds_train.element_spec)

    # Create inputs and encoded features
    inputs, encoded_features = create_model_inputs_and_features(
        set_of_features,
        ds_train,
    )

    # * Build the model --------------------------------------------------------
    # Get & save model's code
    print(
        inspect.getsource(looped_multioutput),
        file=open(temp_dir / "model_dl_code.txt", "w"),
    )
    # Build & compile model
    model = looped_multioutput(
        inputs,
        encoded_features,
        optimizer=PARAMS["_optimizer"],
        loss=PARAMS["_loss"],
        metrics=PARAMS["_metrics"],
        print_summary=False,
    )
    # Plot the model
    fig_model = tf.keras.utils.plot_model(
        model,
        to_file=docs_dir / f"pics/{model.name}.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        expand_nested=False,
        dpi=100,
        rankdir="TB",
        show_layer_activations=True,
    )
    display(fig_model)

    # * Train the model ---------------------------------------------------------
    # for reproducubility
    tf.keras.utils.set_random_seed(RND_STATE)

    # * MLflow: Setup tracking
    run_name = f"{model.name}"

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    experiment_id = mlflow.get_experiment_by_name(
        MLFLOW_EXPERIMENT_NAME
    ).experiment_id

    # * MLflow: Enable autologging
    mlflow.tensorflow.autolog(log_models=True)
    """
        If you use mlflow.tensorflow.autolog() you will get the Tensorboard logs
        in the mlruns/<ExperimentFolder>/<ModelFolder>/artifacts/tensorboard_logs
        To run Tensorboard:
            - Run terminal in the mlruns/<ExperimentFolder>/<ModelFolder>
            >>>tensorboard --logdir artifacts
    """
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=run_name
    ) as run:
        run_id = run.info.run_id

        print(f"\nMLFLOW_TRACKING_URI: {mlflow.get_tracking_uri()}")
        print(f"MLFLOW_ARTIFACT_URI: {mlflow.get_artifact_uri()}")
        print(f"MLFLOW_EXPERIMENT_NAME: {MLFLOW_EXPERIMENT_NAME}")
        print(f"EXPERIMENT_ID: {experiment_id}")
        print(f"RUN_NAME: {run_name}")
        print(f"RUN_ID: {run_id}\n")

        # Fit the model
        tic_fit = time.time()
        history = model.fit(
            ds_train,
            batch_size=PARAMS["_batch_size"],
            epochs=PARAMS["_epochs"],
            verbose=1,
            use_multiprocessing=True,
        )

        # * Disable autologging
        mlflow.tensorflow.autolog(disable=True)
        print(f"Model fitted {timing(tic_fit)}")

        # * Make prediction & define true values
        print(f"\nMake predictions for performance calculating ...")
        y_pred_train = model.predict(ds_train)
        y_pred_test = model.predict(ds_test)

        # Get predictions
        y_pred_train_binary = y_pred_train[0]
        y_pred_test_binary = y_pred_test[0]

        y_pred_train_mclass = y_pred_train[1]
        y_pred_test_mclass = y_pred_test[1]

        # Cast prediction into 1D array
        y_true_train_mclass_1d = np.argmax(y_true_train_mclass, axis=1)
        y_true_test_mclass_1d = np.argmax(y_true_test_mclass, axis=1)

        y_pred_train_mclass_1d = np.argmax(y_pred_train_mclass, axis=1)
        y_pred_test_mclass_1d = np.argmax(y_pred_test_mclass, axis=1)

        # # * Log params
        mlflow.log_params(
            {
                # "_batch_size": PARAMS["_batch_size"],
                "_neurons": PARAMS["_neurons"],
                "_loss": PARAMS["_loss"],
            }
        )

        # * Log model's plot & code
        mlflow.log_artifact(docs_dir / f"pics/{model.name}.png")
        mlflow.log_artifact(temp_dir / "model_dl_code.txt")

        # * Log results for binary classification ------------------------------
        # Log the confusion matrixes
        print(f"Calculate & log Confusion matrix for binary classification...")
        P = 0.5
        cm_train = confusion_matrix(
            y_true_train_binary, y_pred_train_binary > P
        )
        cm_test = confusion_matrix(y_true_test_binary, y_pred_test_binary > P)
        fig_cm_train = plot_confusion_matrix(
            cm_train,
            title=f"{model.name}: Train set: threshold: {P:.2f}",
        )
        fig_cm_test = plot_confusion_matrix(
            cm_test,
            title=f"{model.name}: Test set: threshold: {P:.2f}",
        )
        mlflow.log_figure(fig_cm_train, "./plots/binary_cm_train.png")
        mlflow.log_figure(fig_cm_test, "./plots/binary_cm_test.png")
        display(fig_cm_test)

        # Log the density
        print(f"Calculate & log density of scores' distribution ...")
        fig_dens = plot_density(
            y_true_test_binary,
            np.rint(y_pred_test_binary),
            "Test",
            model.name,
        )
        mlflow.log_figure(fig_dens, "./plots/binary_density_test.png")
        display(fig_dens)

        # * Log results for multiclass classification --------------------------
        # Log the Confusion Matrix
        print(f"Calculate & log confusion matrixes for multiclasses ...")
        cm_train_mclass = confusion_matrix(
            y_true_train_mclass_1d,
            y_pred_train_mclass_1d,
        )
        cm_test_mclass = confusion_matrix(
            y_true_test_mclass_1d,
            y_pred_test_mclass_1d,
        )

        fig_cm_train_mclass = plot_confusion_matrix(
            cm_train_mclass,
            categories=vocab,
            title=f"{model.name}: Train set",
        )
        fig_cm_test_mclass = plot_confusion_matrix(
            cm_test_mclass,
            categories=vocab,
            title=f"{model.name}: Test set",
        )
        mlflow.log_figure(fig_cm_train_mclass, "./plots/mclass_cm_train.png")
        mlflow.log_figure(fig_cm_test_mclass, "./plots/mclass_cm_test.png")
        display(fig_cm_test_mclass)

        # Log the Classification reports
        print(f"Classification Reports for Train set:")
        cr_train = classification_report(
            y_true_train_mclass_1d, y_pred_train_mclass_1d, target_names=vocab
        )
        print(cr_train)
        with open(temp_dir / "mclass_report_train.txt", "w") as f:
            f.write(cr_train)

        print(f"Classification Report for Test set:")
        cr_test = classification_report(
            y_true_test_mclass_1d, y_pred_test_mclass_1d, target_names=vocab
        )
        print(cr_test)
        with open(temp_dir / "mclass_report_test.txt", "w") as f:
            f.write(cr_test)

        mlflow.log_artifact(temp_dir / "mclass_report_train.txt")
        mlflow.log_artifact(temp_dir / "mclass_report_test.txt")

        # Log the ROC curves
        fig_roc_train = plot_multiclass_roc(
            y_true_train_mclass_1d,
            y_pred_train_mclass_1d,
            vocab,
            "Train",
            model.name,
        )
        fig_roc_test = plot_multiclass_roc(
            y_true_test_mclass_1d,
            y_pred_test_mclass_1d,
            vocab,
            "Test",
            model.name,
        )
        mlflow.log_figure(fig_roc_train, "./plots/mclass_roc_train.png")
        mlflow.log_figure(fig_roc_test, "./plots/mclass_roc_test.png")
        display(fig_roc_test)

        model.reset_states()

    tf.keras.backend.clear_session()
    print(f"\nModel fitted {timing(tic_fit)}")
    print("Done.")

    return model, (df_train, df_test), (ds_train, ds_test)


#%% RUN ========================================================================
if __name__ == "__main__":
    DATA_INCOME_FILE_NAME = f"xpm_us_delay_dev_{DATA_PERIOD}.csv.gz"
    JSON_FEATURES_FILE_NAME = f"xpm_us_delay_set_of_features_{DATA_PERIOD}.json"
    FILE_TO_SAVE_DF_TEST = f"xpm_us_delay_test_{DATA_PERIOD}.csv.gz"
    DIR_TO_SAVE_DF_TEST = final_data_folder_path

    model, dfs, dss = main(
        data_income_file_name=DATA_INCOME_FILE_NAME,
        json_features_file_name=JSON_FEATURES_FILE_NAME,
        mlflow_experiment_name="multioutput: final",
    )
    _ = save_df_test(dfs, DIR_TO_SAVE_DF_TEST / FILE_TO_SAVE_DF_TEST)

    print("DONE.")

#%% Evaluate model -------------------------------------------------------------
# eval_scores_train = eval_model(model, ds_train, "train")
# eval_scores_test = eval_model(model, ds_test, "test")
