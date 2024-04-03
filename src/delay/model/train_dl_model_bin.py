"""
    Train a DL model for binary classification.

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
import mlflow

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import confusion_matrix

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

from src/delay.config import final_data_folder_path
from src/delay.config import data_final_dir
from src/delay.config import docs_dir
from src/delay.config import mlflow_tracking_uri
from src/delay.config import tensorboard_dir

from src/delay.utils import timing
from src/delay.model._dl_utils import df_to_ds
from src/delay.model._dl_utils import create_model_inputs_and_features

from src/delay.model._dl_models import vanilla_binary
from src/delay.model._dl_models import vanilla_binary_normalized
from src/delay.model._dl_models import eval_model

from src/delay.plots import plot_history_metrics
from src/delay.plots import plot_confusion_matrix
from src/delay.plots import plot_density
from src/delay.plots import plot_3roc

#%% Custom funcs ---------------------------------------------------------------

#%% MAIN -----------------------------------------------------------------------
def main(
    data_income_file_name,
    json_features_file_name,
    mlflow_experiment_name,
):

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
        "_batch_size": 8184,  # * should be optimized for machine specs
        "_neurons": "1024-512",
        "_epochs": 20,
        "_optimizer": tf.keras.optimizers.RMSprop(learning_rate=0.001),
        # "_optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
        "_loss": "binary_crossentropy",
        "_metrics": [
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="auc"),
            # tf.keras.metrics.TruePositives(name="tp"),
            # tf.keras.metrics.FalsePositives(name="fp"),
            # tf.keras.metrics.TrueNegatives(name="tn"),
            # tf.keras.metrics.FalseNegatives(name="fn"),
            # tf.keras.metrics.Precision(name="precision"),
            # tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    }

    print(f"\nRead data from {path_to_data} \n\t {path_to_json}")
    df = pd.read_csv(path_to_data, usecols=COLS_TO_READ)
    # Get the dictionary of all the input features for NN.
    with open(path_to_json, "r") as f:
        set_of_features = json.load(f)

    # Get the targets features
    y_01 = set_of_features["type_y"]["binary"][0]

    # Split the data with stratified target
    # # * To val & test model on latest months - uncomment this block
    # df_test = df[df["report_month"]==202207]
    # df_val = df[df["report_month"]==202206]
    # df_train = df[df["report_month"] < 202206]

    # * To val & test model on latest months - comment this block
    print(f"\nSplit whole data on train & test sets ...")
    df_train, df_test = model_selection.train_test_split(
        df,
        train_size=0.9,
        random_state=RND_STATE,
        shuffle=True,
        stratify=df[y_01],
    )
    # * To val & test model on latest months - comment this block
    print(f"Split train data on train & val sets ...")
    df_train, df_val = model_selection.train_test_split(
        df_train,
        train_size=0.9,
        random_state=RND_STATE,
        shuffle=True,
        stratify=df_train[y_01],
    )
    print(f"Dev set: {df.shape[1]} vars & {df.shape[0]:,} rows.")
    print(f"\tdf_train: {df_train.shape[1]} vars & {df_train.shape[0]:,} rows.")
    print(f"\tdf_val: {df_val.shape[1]} vars & {df_val.shape[0]:,} rows.")
    print(f"\tdf_test: {df_test.shape[1]} vars & {df_test.shape[0]:,} rows.")

    # # ^ Analyze class imbalance in the targets
    # neg, pos = np.bincount(df_train["y_be_delayed"])
    # total = neg + pos
    # # Scaling by (total/2) helps keep the loss to a similar magnitude.
    # # The sum of the weights of all examples stays the same.
    # # Variant#1 from https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    # weight_1_for_0 = (1 / neg) * (total) / 2.0
    # weight_1_for_1 = (1 / pos) * (total) / 2.0
    # # ! THIS APPROACH IS BETTER:
    # weights_1 = {0: weight_1_for_0, 1: weight_1_for_1}

    # # Variant#2 from https://keras.io/examples/structured_data/imbalanced_classification/
    # weight_2_for_0 = 1 / neg
    # weight_2_for_1 = 1 / pos
    # # THIS IS NUMBER #2 AT APPROACHES:
    # weights_2 = {0: weight_2_for_0, 1: weight_2_for_1}
    # print(
    #     f"""
    #     # of (0): {neg:,}: {neg / total}
    #     # of (1): {pos:,}: {pos / total}
    #     """
    # )

    # Convert pd.DataFrame to tf.Dataset
    set_of_features["y"] = [
        "y_be_delayed"
    ]  # for correct converting in df_to_ds
    #! Do not use the shuffle=True! It leads to wrong accuracy on the test set
    ds_train, y_true_train = df_to_ds(
        df_train,
        set_of_features,
        PARAMS["_batch_size"],
        shuffle=False,
    )
    ds_val, y_true_val = df_to_ds(
        df_val,
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
    # # Alternative way - use Test set w/o target
    # ds_test = tf.data.Dataset.from_tensor_slices(
    #     dict(df_test[set_of_features["x"]].copy())
    # ).batch(PARAMS["_batch_size"])

    print(f"\nInspect the train dataset's elements ...")
    pprint(ds_train.element_spec)
    # # View first 3 samples
    # for x, y in ds_train.take(3):
    #     print("Input:", x)
    #     print("Target:", y)

    # Create inputs and encoded features
    inputs, encoded_features = create_model_inputs_and_features(
        set_of_features,
        ds_train,
    )

    # * Build the model --------------------------------------------------------
    model = vanilla_binary_normalized(
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

    # * Train the model --------------------------------------------------------
    #! for reproducubility
    tf.keras.utils.set_random_seed(RND_STATE)

    # * MLflow: Setup tracking
    run_name = f"{model.name}"

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    experiment_id = mlflow.get_experiment_by_name(
        MLFLOW_EXPERIMENT_NAME
    ).experiment_id

    # * MLflow: Enable autologging
    mlflow.tensorflow.autolog(log_models=False)
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
            validation_data=ds_val,
            use_multiprocessing=True,
            # class_weight=weights_1,
            callbacks=[
                # #root project folder [ft-bol-us-delay] > Open in Integrated Terminal
                # #>>>tensorboard --logdir tensorboard
                # tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir / fit_time),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=4,
                    verbose=1,
                    mode="max",
                    # restore_best_weights=True, # better should be commented
                ),
                # tf.keras.callbacks.ModelCheckpoint(
                #     filepath=FILEPATH.as_posix(),
                #     monitor="val_auc",
                #     verbose=1,
                #     mode="max",
                #     save_best_only=True,
                # ),
                # tf.keras.callbacks.LearningRateScheduler(
                #     scheduler_exp, verbose=1,
                # ),
            ],
        )

        # * Disable autologging
        mlflow.tensorflow.autolog(disable=True)
        print(f"Model fitted {timing(tic_fit)}")

        # Make prediction
        print(f"\nMake predictions for performance calculating ...")
        y_pred_train = model.predict(ds_train)
        y_pred_val = model.predict(ds_val)
        y_pred_test = model.predict(ds_test)

        # # * Log params
        mlflow.log_params(
            {
                # "_batch_size": PARAMS["_batch_size"],
                "_neurons": PARAMS["_neurons"],
                "_loss": PARAMS["_loss"],
            }
        )

        # * Log plots
        mlflow.log_artifact(docs_dir / f"pics/{model.name}.png")

        # Log the model's loss
        print(f"Calculate & log losses ...")
        fig_losses = plot_history_metrics(history, model.name)
        mlflow.log_figure(fig_losses, "./plots/losses.png")
        display(fig_losses)

        # Log the confusion matrixes
        print(f"Calculate & log Confusion matrix ...")
        P = 0.5
        cm_train = confusion_matrix(y_true_train[0], y_pred_train > P)
        cm_val = confusion_matrix(y_true_val[0], y_pred_val > P)
        cm_test = confusion_matrix(y_true_test[0], y_pred_test > P)
        fig_cm_train = plot_confusion_matrix(
            cm_train,
            title=f"{model.name}: Train set: threshold: {P:.2f}",
        )
        fig_cm_val = plot_confusion_matrix(
            cm_val,
            title=f"{model.name}: Val set: threshold: {P:.2f}",
        )
        fig_cm_test = plot_confusion_matrix(
            cm_test,
            title=f"{model.name}: Test set: threshold: {P:.2f}",
        )
        mlflow.log_figure(fig_cm_train, "./plots/cm_train.png")
        mlflow.log_figure(fig_cm_val, "./plots/cm_val.png")
        mlflow.log_figure(fig_cm_test, "./plots/cm_test.png")
        display(fig_cm_val)

        # Log the ROC curves
        print(f"Calculate & log ROC curves ...")
        fig_3roc = plot_3roc(
            y_true_train[0],
            np.rint(y_pred_train),
            y_true_val[0],
            np.rint(y_pred_val),
            y_true_test[0],
            np.rint(y_pred_test),
            model.name,
        )
        mlflow.log_figure(fig_3roc, "./plots/3roc.png")
        display(fig_3roc)

        # Log the density
        print(f"Calculate & log density of scores' distribution ...")
        fig_dens = plot_density(
            y_true_test[0],
            np.rint(y_pred_test),
            "Test",
            model.name,
        )
        mlflow.log_figure(fig_dens, "./plots/density_test.png")
        display(fig_dens)

        model.reset_states()

    tf.keras.backend.clear_session()
    print(f"\nModel fitted {timing(tic_fit)}")
    print("Done.")

    return model, (df_train, df_val, df_test), (ds_train, ds_val, ds_test)


#%% RUN ========================================================================
if __name__ == "__main__":
    model, _, dss = main(
        data_income_file_name="xpm_us_delay_dev_2022.csv.gz",
        json_features_file_name="xpm_us_delay_set_of_features_2022.json",
        mlflow_experiment_name="binary: 2022",
    )

#%% Evaluate model -------------------------------------------------------------
# ds_train = dss[0]
# ds_val = dss[1]
# ds_test = dss[2]

# eval_scores_train = eval_model(model, ds_train, "train")
# eval_scores_val = eval_model(model, ds_val, "val")
# eval_scores_test = eval_model(model, ds_test, "test")
