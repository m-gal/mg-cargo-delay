""" Contains the functions used for
    features encoding \ embedding and creating DL models' inputs.

    @author: mikhail.galkin
"""

#%% Import needed python libraryies and project config info
import tensorflow as tf
import math
import numpy as np

# import pandas as pd

from sklearn import model_selection
from IPython.display import display
from pprint import pprint

#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ U T I L S ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def df_split_data(df, rnd_state, **kwargs):
    print(f"\nSplit data on train & test sets ...")
    df_train_val, df_test = model_selection.train_test_split(
        df,
        train_size=0.8,
        random_state=rnd_state,
        shuffle=True,
        **kwargs,
    )

    df_train, df_val = model_selection.train_test_split(
        df_train_val,
        train_size=0.85,
        random_state=rnd_state,
        shuffle=True,
        **kwargs,
    )

    print(f"Dev set: {df.shape[1]} vars & {df.shape[0]:,} rows.")
    print(f"\tdf_train: {df_train.shape[1]} vars & {df_train.shape[0]:,} rows.")
    print(f"\tdf_val: {df_val.shape[1]} vars & {df_val.shape[0]:,} rows.")
    print(f"\tdf_test: {df_test.shape[1]} vars & {df_test.shape[0]:,} rows.")
    return df_train, df_val, df_test


def df_to_ds(df, set_of_features, batch_size=None, shuffle=False):
    print(f"\nConverting Pandas DataFrame to TF Dataset ...")
    df_X = df[set_of_features["x"]].copy()

    # To use the tf.data.Dataset.from_tensor_slices()
    # we should to have tuple of np.arrayes
    # due this we convert pd.series into 2d array and put them into list
    labels = []
    for y in set_of_features["y"]:
        labels.append(np.array(df[y].values.tolist()))

    # The given tensors are sliced along their first dimension.
    # This operation preserves the structure of the input tensors,
    # removing the first dimension of each tensor and using it as the dataset dims.
    # All input tensors must have the same size in their first dimensions.
    ds_X = tf.data.Dataset.from_tensor_slices(dict(df_X))
    ds_Y = tf.data.Dataset.from_tensor_slices(tuple(labels))
    ds = tf.data.Dataset.zip((ds_X, ds_Y))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df_X), seed=42)

    # Combines consecutive elements of this dataset into batches.
    if batch_size is not None:
        ds = ds.batch(batch_size)
    return ds, labels


#! -----------------------------------------------------------------------------
#! ------------------------- E N C O D I N G  ----------------------------------
#! -----------------------------------------------------------------------------
""" Function for demonstrate several types of feature column -------------------
    TensorFlow provides many types of feature columns.
    In this section, we will create several types of feature columns,
    and demonstrate how they transform a column from the dataframe.
    We will use this batch to demonstrate several types of feature columns
        example_batch = next(iter(ds_val))[0]
    A utility method to create a feature column
    and to transform a batch of data
    def demo(feature_column):
        feature_layer = keras.layers.DenseFeatures(feature_column)
        print(feature_layer(example_batch).numpy())
"""


def encode_numeric_float_feature(feature_name, dataset):
    # Create model input for feature
    input = tf.keras.layers.Input(shape=(1,), name=feature_name)
    # Create a Normalization layer for our feature
    normalizer = tf.keras.layers.Normalization(
        name=f"normalized_{feature_name}"
    )
    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[feature_name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    print(f"\tLearn the statistics of the data < {feature_name} > ...")
    normalizer.adapt(feature_ds)
    print(f"\tNormalize the input feature ...")
    encoded_feature = normalizer(input)
    #! It is possible to not use the Normalizing:
    # encoded_feature = input
    return input, encoded_feature


def encode_categorical_feature_with_defined_vocab(
    feature_name, vocab, dataset, is_string
):
    if is_string:
        lookup_class = tf.keras.layers.StringLookup
        dtype_class = "string"
    else:
        lookup_class = tf.keras.layers.IntegerLookup
        dtype_class = "int32"

    # Create model input for feature
    input = tf.keras.layers.Input(
        shape=(1,),
        name=feature_name,
        dtype=dtype_class,
    )

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[feature_name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Since we are not using a mask token we set mask_token to None,
    # and possible expecting any out of vocabulary (oov) token - set num_oov_indices to 1.
    lookup = lookup_class(vocabulary=vocab, name=f"lookup_{feature_name}")
    print(f"\tTurn the categorical input into integer indices ...")
    encoded_feature = lookup(input)

    print(f"\tCreate an embedding layer with the specified dimensions ...")
    #! It is possible to not use the embedding. Just comment the below.
    embedding_dims = int(math.sqrt(lookup.vocabulary_size()))
    embedding = tf.keras.layers.Embedding(
        input_dim=lookup.vocabulary_size(),
        output_dim=embedding_dims,
        name=f"lookuped_{feature_name}",
    )
    embedded_feature = embedding(encoded_feature)
    flatten = tf.keras.layers.Flatten(name=f"embedded_{feature_name}")
    encoded_feature = flatten(embedded_feature)
    return input, encoded_feature


def create_model_inputs_and_features(
    set_of_features,
    dataset,
    print_result=True,
):
    print(f"\nCreate inputs & Encode features ...")
    inputs = {}
    encoded_features = {}
    type_x = set_of_features["type_x"]
    for feature_name in set_of_features["x"]:
        print(f"Feature: < {feature_name} > :")

        if feature_name in type_x["numeric_float"]:
            input, encoded_feature = encode_numeric_float_feature(
                feature_name, dataset
            )

        if feature_name in type_x["categorical_int_w_vocab"]:
            vocabulary = type_x["categorical_int_w_vocab"][feature_name]
            (
                input,
                encoded_feature,
            ) = encode_categorical_feature_with_defined_vocab(
                feature_name, vocabulary, dataset, is_string=False
            )

        if feature_name in type_x["categorical_str_w_vocab"]:
            vocabulary = type_x["categorical_str_w_vocab"][feature_name]
            (
                input,
                encoded_feature,
            ) = encode_categorical_feature_with_defined_vocab(
                feature_name, vocabulary, dataset, is_string=True
            )

        inputs[feature_name] = input
        encoded_features[feature_name] = encoded_feature

    if print_result:
        print(f"\nPrepared Inputs:")
        display(list(inputs.values()))
        print(f"Encoded features:")
        display(list(encoded_features.values()))
    return inputs, encoded_features
