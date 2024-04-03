"""
    Load preprocessed Xportmine BoL US Import data
    Process data for modeling

    @author: mikhail.galkin
"""

#%% Setup ----------------------------------------------------------------------
import sys
import time

import winsound
import json

import pandas as pd
from pprint import pprint
from IPython.display import display

#%% Load project's stuff -------------------------------------------------------
sys.path.extend([".", "./.", "././.", "../..", "../../.."])

from src/delay.config import DATA_PERIOD
from src/delay.config import final_data_folder_path
from src/delay.config import data_final_dir
from src/delay.utils import timing


#%% Custom funcs ---------------------------------------------------------------
def read_xpm_processed_data_parquet(
    processed_folder_path,
    processed_files_names: list,  # or None
    cols_to_read: list,  # or None
    drop_dupes=True,
    **kwargs,
):
    """Read parquet-files w/ processed Xportmine BoL data
    and concatenate the data into one DataFrame

    Args:
        processed_folder_path (Path): Path to folder with data processed parquets.
        processed_files_names (list of str or None): files' names w/o extension.
        cols_to_read (list of str or None): Columns' names to read.
        drop_dupes (bool): Either check for duplicates
        **kwargs: kwargs for pandas.read_parquet()
    Returns:
        Pandas DataFrame : Data combined into one DF
    """

    print(f"\nRead the Xportmine processed data ..............................")
    tic_main = time.time()

    df = pd.DataFrame()  # empty DF for interim result

    print(f"---- Get processed data for the columns:")
    pprint("All columns..." if cols_to_read is None else cols_to_read)

    for file in processed_files_names:
        print(f"---- Read processed data from the <{file}> ...")

        _df = pd.read_parquet(
            path=processed_folder_path / f"{file}.parquet",
            columns=cols_to_read,
            engine="fastparquet",
        )

        # Concatenate the data
        df = pd.concat([df, _df], axis=0, ignore_index=True)
        del _df

    if drop_dupes:
        df.drop_duplicates(inplace=True)

    df.reset_index(inplace=True, drop=True)
    print(f"\nLoaded dataset has # {len(df):,} records ...")
    # display(df.info(show_counts=True))

    winsound.Beep(frequency=2000, duration=200)

    min, sec = divmod(time.time() - tic_main, 60)
    print(f"All read for: {int(min)}min {int(sec)}sec")

    return df


#%% MAIN -----------------------------------------------------------------------
def main():
    FILE_TO_SAVE = f"xpm_us_delay_dev_{DATA_PERIOD}.csv.gz"
    JSON_TO_SAVE = f"xpm_us_delay_set_of_features_{DATA_PERIOD}.json"

    #! Metadata for incoming data ----------------------------------------------
    DATA_INCOME_FOLDER_PATH = final_data_folder_path
    DATA_INCOME_FILES_NAMES = [
        "xpm_processed_US-2019",
        "xpm_processed_US-2020",
        "xpm_processed_US-2021",
        "xpm_processed_US-2022",
    ]
    COLS_TO_READ = [
        "report_month",
        "arrival_date_estimate",
        "arrival_date_delay",
        "carrier_code",
        # "carrier_name",
        # "consignee_address", # ? should be ZIPs
        # "consignee_name",  # ???
        "container_id",
        "container_type_of_service",
        # "notify_party_address", # ? should be ZIPs
        # "notify_party_name",  # ???
        "place_of_receipt",
        "port_of_lading",
        "port_of_unlading",
        # "shipper_address", # ? should be ZIPs
        # "shipper_name",  # ???
        "vessel_name",
        "vessel_type",
        # "weight_kg",
        "weight_kg_outliers_off",
    ]

    #! Metadata for features & targets -----------------------------------------
    CATEGORICAL_STR_FEATURE_MISSING_STUB = "(UNKNOWN)"
    NEW_CONTAINER_FEATURES = ["container_id_prefix", "container_id_number"]

    # A lists of the binary feature names. Feature like "is_animal"
    BINARY_FEATURE_NAMES = []  # ["is_animal"]

    # A list of the numerical feature names.
    NUMERIC_INTEGER_FEATURE_NAMES = []  # i.e ["age"]
    NUMERIC_FLOAT_FEATURE_NAMES = ["weight_kg"]

    # A list of the categorical features we want be one-hot encoding.
    CATEGORICAL_INTEGER_FEATURE_NAMES_OHE = []
    CATEGORICAL_STRING_FEATURE_NAMES_OHE = []

    # A list of the categorical features we want be embedded.
    CATEGORICAL_INTEGER_FEATURE_NAMES = []
    CATEGORICAL_STRING_FEATURE_NAMES = [
        "ade_month",
        "carrier_code",
        "container_id_prefix",
        "container_type_of_service",
        "place_of_receipt",
        "port_of_lading",
        "port_of_unlading",
        "vessel_name",
    ]

    # Textual features we want be embedded
    TEXT_FEATURE_NAMES = []

    # Targets' names and types
    BINARY_TARGET_NAME = "y_be_delayed"
    MULTICLASS_TARGET_NAME = "y_dpd_delayed"
    MULTILABEL_TARGET_NAME = []

    # A list of all the input features.
    cols_x = (
        BINARY_FEATURE_NAMES
        + NUMERIC_INTEGER_FEATURE_NAMES
        + NUMERIC_FLOAT_FEATURE_NAMES
        + CATEGORICAL_INTEGER_FEATURE_NAMES_OHE
        + CATEGORICAL_STRING_FEATURE_NAMES_OHE
        + CATEGORICAL_INTEGER_FEATURE_NAMES
        + CATEGORICAL_STRING_FEATURE_NAMES
        + TEXT_FEATURE_NAMES
    )

    # The names of the target features.
    cols_y = (
        [BINARY_TARGET_NAME] + [MULTICLASS_TARGET_NAME] + MULTILABEL_TARGET_NAME
    )

    cols_dev = ["report_month"] + cols_x + cols_y

    tic = time.time()
    print(f"\nRead data from {DATA_INCOME_FOLDER_PATH} .......................")
    df = read_xpm_processed_data_parquet(
        processed_folder_path=DATA_INCOME_FOLDER_PATH,
        processed_files_names=DATA_INCOME_FILES_NAMES,
        cols_to_read=COLS_TO_READ,
        drop_dupes=True,
    )

    print(f"Filter the data ..................................................")
    df = df[df["vessel_type"] == "container_ship"]
    if "weight_kg_outliers_off" in df.columns.to_list():
        df.rename(columns={"weight_kg_outliers_off": "weight_kg"}, inplace=True)

    print(f"Get estimated month of arrival number ............................")
    df["ade_month"] = (
        df["arrival_date_estimate"].dt.month_name().str[:3].str.upper()
    )
    # df.drop(columns=["arrival_date_estimate"], inplace=True)

    print(f"Unwind \ Explode the 'container_id' ..............................")
    df["container_id"] = df["container_id"].fillna(
        CATEGORICAL_STR_FEATURE_MISSING_STUB
    )
    df["container_id"] = df["container_id"].str.split(", ").to_list()
    df = df.explode("container_id")
    # Split containers' IDs
    df[NEW_CONTAINER_FEATURES] = df["container_id"].str.extract(
        "([A-Za-z]*)(\d*)"
    )
    # Replace wrong strings with stub for missing
    df[NEW_CONTAINER_FEATURES] = df[NEW_CONTAINER_FEATURES].replace(
        "", CATEGORICAL_STR_FEATURE_MISSING_STUB
    )
    df[NEW_CONTAINER_FEATURES] = df[NEW_CONTAINER_FEATURES].replace(
        "NA", CATEGORICAL_STR_FEATURE_MISSING_STUB
    )
    df[NEW_CONTAINER_FEATURES] = df[NEW_CONTAINER_FEATURES].replace(
        "0", CATEGORICAL_STR_FEATURE_MISSING_STUB
    )

    print(f"Binning the 'arriaval_date_delay' ................................")
    bins = [-360, 0, 3, 7, 14, 21, 360]
    classes = ["[00]", "[01-03]", "[04-07]", "[08-14]", "[15-21]", "[22+]"]
    df[MULTICLASS_TARGET_NAME] = pd.cut(
        df["arrival_date_delay"], bins=bins, labels=classes
    )
    # Create binary variable
    df[BINARY_TARGET_NAME] = df["arrival_date_delay"].apply(
        lambda x: 1 if x > 0 else 0
    )

    print(f"Handle missing values ............................................")
    df[CATEGORICAL_STRING_FEATURE_NAMES] = df[
        CATEGORICAL_STRING_FEATURE_NAMES
    ].fillna(CATEGORICAL_STR_FEATURE_MISSING_STUB)
    df[NUMERIC_FLOAT_FEATURE_NAMES] = df[NUMERIC_FLOAT_FEATURE_NAMES].fillna(0)
    df[NUMERIC_INTEGER_FEATURE_NAMES] = df[
        NUMERIC_INTEGER_FEATURE_NAMES
    ].fillna(0)
    # print("Sum of missing values :")
    # display(df.isna().sum())

    print(f"Drop redundant columns & wrong data ..............................")
    df.drop(columns=[col for col in df if col not in cols_dev], inplace=True)
    df = df[sorted(df.columns)]

    print(f"Strip all strings in the dataset .................................")
    df_obj = df.select_dtypes(["object"])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    del df_obj

    print(f"Create features dictionary .......................................")
    # A dictionary of the categorical features and their vocabulary for NN
    categorical_int_feature_names_w_vocab = {}
    if CATEGORICAL_INTEGER_FEATURE_NAMES:
        for feature in CATEGORICAL_INTEGER_FEATURE_NAMES:
            categorical_int_feature_names_w_vocab[feature] = sorted(
                list(df[feature].unique())
            )
    # A dictionary of the categorical features and their vocabulary for NN.
    categorical_str_feature_names_w_vocab = {}
    if CATEGORICAL_STRING_FEATURE_NAMES:
        for feature in CATEGORICAL_STRING_FEATURE_NAMES:
            vocab = sorted(list(df[feature].unique()))
            if CATEGORICAL_STR_FEATURE_MISSING_STUB not in vocab:
                vocab = [CATEGORICAL_STR_FEATURE_MISSING_STUB] + vocab

            categorical_str_feature_names_w_vocab[feature] = vocab

    multiclass_target_name_w_labels = {MULTICLASS_TARGET_NAME: classes}

    # A dictionary of all the input columns\features.
    set_of_features = {
        "x": cols_x,
        "y": cols_y,
        "type_x": {
            "binary": BINARY_FEATURE_NAMES,
            "numeric_float": NUMERIC_FLOAT_FEATURE_NAMES,
            "numeric_int": NUMERIC_INTEGER_FEATURE_NAMES,
            "categorical_int_w_vocab": categorical_int_feature_names_w_vocab,
            "categorical_str_w_vocab": categorical_str_feature_names_w_vocab,
            "text": TEXT_FEATURE_NAMES,
        },
        "type_y": {
            "binary": [BINARY_TARGET_NAME],
            "multiclass": multiclass_target_name_w_labels,
            "multilabel": MULTILABEL_TARGET_NAME,
        },
    }

    winsound.Beep(frequency=2000, duration=200)
    print(f"Data processed {timing(tic)}")

    # Save processed dataframe
    dir_to_save_data = final_data_folder_path
    path_to_save = dir_to_save_data / FILE_TO_SAVE
    print(f"Save data as CSV.GZ: {path_to_save}")
    tic = time.time()
    df.to_csv(
        path_to_save,
        index=False,
        encoding="utf-8-sig",
        compression="gzip",
    )
    print(f"Data saved {timing(tic)}")

    # Save dictionary of features
    dir_to_save_json = data_final_dir
    json_path_to_save = dir_to_save_json / JSON_TO_SAVE
    print(f"Save features metadata as JSON: {json_path_to_save}")
    with open(json_path_to_save, "w") as fp:
        json.dump(set_of_features, fp)

    winsound.Beep(frequency=2000, duration=200)
    print(f"Final dataset:")
    display(df.info())
    print("\nDone.")

    return df, set_of_features


#%% RUN ========================================================================
if __name__ == "__main__":
    df, sof = main()
