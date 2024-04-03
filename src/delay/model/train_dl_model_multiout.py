"""
    Train a DL model with 2 outputs:
    - for binary ckassification
    - for multiclass classification.

    @author: mikhail.galkin
"""

# TODO: For categorical features with vocab, try to use not the vocab based on whole DF
# TODO: but on train\val\test sets accordinly. To generalize the model

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

# %% Setup ----------------------------------------------------------------------
import sys
import time

import json
import mlflow

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from pprint import pprint
from IPython.display import display

# %% Setup the Tensorflow -------------------------------------------------------
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
        print(f"\tPhysical GPUs: {len(gpus)}\n\tLogical GPUs: {len(logical_gpus)}")
        print(f"\tIs built with GPU support: {gpu_support}")
        print(f"\tIs built with CUDA: {cuda}")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# %% Load project's stuff -------------------------------------------------------
sys.path.extend([".", "./.", "././.", "../..", "../../.."])

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

# %% Custom funcs ---------------------------------------------------------------


# %% MAIN -----------------------------------------------------------------------
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
        "_batch_size": 8184,
        "_neurons": "1024-512-512",
        "_epochs": 20,
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
    df = pd.read_csv(path_to_data, usecols=COLS_TO_READ)
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
        train_size=0.9,
        random_state=RND_STATE,
        shuffle=True,
        stratify=df[y_mc],
    )
    print(f"Split train data on train & val sets ...")
    df_train, df_val = model_selection.train_test_split(
        df_train,
        train_size=0.9,
        random_state=RND_STATE,
        shuffle=True,
        stratify=df_train[y_mc],
    )
    print(f"Dev set: {df.shape[1]} vars & {df.shape[0]:,} rows.")
    print(f"\tdf_train: {df_train.shape[1]} vars & {df_train.shape[0]:,} rows.")
    print(f"\tdf_val: {df_val.shape[1]} vars & {df_val.shape[0]:,} rows.")
    print(f"\tdf_test: {df_test.shape[1]} vars & {df_test.shape[0]:,} rows.")

    # Convert pd.DataFrame to tf.Dataset
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

    y_true_train_binary = y_true_train[0]
    y_true_val_binary = y_true_val[0]
    y_true_test_binary = y_true_test[0]

    y_true_train_mclass = y_true_train[1]
    y_true_val_mclass = y_true_val[1]
    y_true_test_mclass = y_true_test[1]

    print(f"\nInspect the train dataset's elements ...")
    pprint(ds_train.element_spec)

    # Create inputs and encoded features
    inputs, encoded_features = create_model_inputs_and_features(
        set_of_features,
        ds_train,
    )

    # * Build the model --------------------------------------------------------
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
    experiment_id = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME).experiment_id

    # * MLflow: Enable autologging
    mlflow.tensorflow.autolog(log_models=False)
    """
        If you use mlflow.tensorflow.autolog() you will get the Tensorboard logs
        in the mlruns/<ExperimentFolder>/<ModelFolder>/artifacts/tensorboard_logs
        To run Tensorboard:
            - Run terminal in the mlruns/<ExperimentFolder>/<ModelFolder>
            >>>tensorboard --logdir artifacts
    """
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
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
                    # restore_best_weights=True, # for better should be commented
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

        # * Make prediction & define true values
        print(f"\nMake predictions for performance calculating ...")
        y_pred_train = model.predict(ds_train)
        y_pred_val = model.predict(ds_val)
        y_pred_test = model.predict(ds_test)

        # Get predictions
        y_pred_train_binary = y_pred_train[0]
        y_pred_val_binary = y_pred_val[0]
        y_pred_test_binary = y_pred_test[0]

        y_pred_train_mclass = y_pred_train[1]
        y_pred_val_mclass = y_pred_val[1]
        y_pred_test_mclass = y_pred_test[1]

        # Cast prediction into 1D array
        y_true_train_mclass_1d = np.argmax(y_true_train_mclass, axis=1)
        y_true_val_mclass_1d = np.argmax(y_true_val_mclass, axis=1)
        y_true_test_mclass_1d = np.argmax(y_true_test_mclass, axis=1)

        y_pred_train_mclass_1d = np.argmax(y_pred_train_mclass, axis=1)
        y_pred_val_mclass_1d = np.argmax(y_pred_val_mclass, axis=1)
        y_pred_test_mclass_1d = np.argmax(y_pred_test_mclass, axis=1)

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
        display(fig_losses)
        mlflow.log_figure(fig_losses, "./plots/losses.png")

        # * Log results for binary classification ------------------------------
        # Log the confusion matrixes
        print(f"Calculate & log Confusion matrix for binary classification...")
        P = 0.5
        cm_train = confusion_matrix(y_true_train_binary, y_pred_train_binary > P)
        cm_val = confusion_matrix(y_true_val_binary, y_pred_val_binary > P)
        cm_test = confusion_matrix(y_true_test_binary, y_pred_test_binary > P)
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
        mlflow.log_figure(fig_cm_train, "./plots/binary_cm_train.png")
        mlflow.log_figure(fig_cm_val, "./plots/binary_cm_val.png")
        mlflow.log_figure(fig_cm_test, "./plots/binary_cm_test.png")
        display(fig_cm_val)

        # Log the ROC curves
        print(f"Calculate & log ROC curves ...")
        fig_3roc = plot_3roc(
            y_true_train_binary,
            np.rint(y_pred_train_binary),
            y_true_val_binary,
            np.rint(y_pred_val_binary),
            y_true_test_binary,
            np.rint(y_pred_test_binary),
            model.name,
        )
        mlflow.log_figure(fig_3roc, "./plots/binary_3roc.png")
        display(fig_3roc)

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
        cm_val_mclass = confusion_matrix(
            y_true_val_mclass_1d,
            y_pred_val_mclass_1d,
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
        fig_cm_val_mclass = plot_confusion_matrix(
            cm_val_mclass,
            categories=vocab,
            title=f"{model.name}: Val set",
        )
        fig_cm_test_mclass = plot_confusion_matrix(
            cm_test_mclass,
            categories=vocab,
            title=f"{model.name}: Test set",
        )
        mlflow.log_figure(fig_cm_train_mclass, "./plots/mclass_cm_train.png")
        mlflow.log_figure(fig_cm_val_mclass, "./plots/mclass_cm_val.png")
        mlflow.log_figure(fig_cm_test_mclass, "./plots/mclass_cm_test.png")
        display(fig_cm_val_mclass)
        display(fig_cm_test_mclass)

        # Log the Classification reports
        print(f"Classification Reports for Train set:")
        cr_train = classification_report(
            y_true_train_mclass_1d, y_pred_train_mclass_1d, target_names=vocab
        )
        print(cr_train)
        with open(temp_dir / "mclass_report_train.txt", "w") as f:
            f.write(cr_train)

        print(f"Classification Report for Val set:")
        cr_val = classification_report(
            y_true_val_mclass_1d, y_pred_val_mclass_1d, target_names=vocab
        )
        print(cr_val)
        with open(temp_dir / "mclass_report_val.txt", "w") as f:
            f.write(cr_val)

        print(f"Classification Report for Test set:")
        cr_test = classification_report(
            y_true_test_mclass_1d, y_pred_test_mclass_1d, target_names=vocab
        )
        print(cr_test)
        with open(temp_dir / "mclass_report_test.txt", "w") as f:
            f.write(cr_test)

        mlflow.log_artifact(temp_dir / "mclass_report_train.txt")
        mlflow.log_artifact(temp_dir / "mclass_report_val.txt")
        mlflow.log_artifact(temp_dir / "mclass_report_test.txt")

        # Log the ROC curves
        fig_roc_train = plot_multiclass_roc(
            y_true_train_mclass_1d,
            y_pred_train_mclass_1d,
            vocab,
            "Train",
            model.name,
        )
        fig_roc_val = plot_multiclass_roc(
            y_true_val_mclass_1d,
            y_pred_val_mclass_1d,
            vocab,
            "Val",
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
        mlflow.log_figure(fig_roc_val, "./plots/mclass_roc_val.png")
        mlflow.log_figure(fig_roc_test, "./plots/mclass_roc_test.png")
        display(fig_roc_val)
        display(fig_roc_test)

        model.reset_states()

    tf.keras.backend.clear_session()
    print(f"\nModel fitted {timing(tic_fit)}")
    print("Done.")

    return model, (df_train, df_val, df_test), (ds_train, ds_val, ds_test)


# %% RUN ========================================================================
if __name__ == "__main__":
    model, _, dss = main(
        data_income_file_name="xpm_us_delay_dev_201901-202212.csv.gz",
        json_features_file_name="xpm_us_delay_set_of_features_201901-202212.json",
        mlflow_experiment_name="multioutput: 2019-22",
    )

# %% Evaluate model -------------------------------------------------------------
# ds_train = dss[0]
# ds_val = dss[1]
# ds_test = dss[2]

# eval_scores_train = eval_model(model, ds_train, "train")
# eval_scores_val = eval_model(model, ds_val, "val")
# eval_scores_test = eval_model(model, ds_test, "test")
