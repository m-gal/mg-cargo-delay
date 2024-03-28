""" Contains the DL models' architectures
    and functions used for models' creating, training and evaluating.

    @author: mikhail.galkin
"""

#%% Import needed python libraryies and project config info
import tensorflow as tf

#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ U T I L S ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def eval_model(model, dataset, dataset_name):
    print(f"\nModel evaluation for: {dataset_name} ...")
    evals = model.evaluate(dataset, verbose=1)
    metrics_names = [name + f"_{dataset_name}" for name in model.metrics_names]
    eval_scores = dict(zip(metrics_names, evals))
    return eval_scores


def predict_model(model, dataset, dataset_name):
    print("\nModel prediction for: " + dataset_name)
    prediction = model.predict(dataset)
    print(dataset_name + " predicted")
    return prediction


#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~ M O D E L S ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def vanilla_binary(
    inputs,
    encoded_features,
    optimizer,
    loss,
    metrics,
    print_summary=True,
):
    # You needs 'input' and 'encoded_features' as a lists
    inputs_list = list(inputs.values())
    encoded_features_list = list(encoded_features.values())

    all = tf.keras.layers.concatenate(encoded_features_list, name="all")
    x = tf.keras.layers.Dense(
        units=512,
        activation="relu",
        name="dense1",
    )(all)
    output = tf.keras.layers.Dense(
        units=1,
        activation="sigmoid",
        name="proba_delay",
    )(x)

    model = tf.keras.Model(
        inputs_list,
        output,
        name="vanilla_binary",
    )
    model.compile(optimizer, loss, metrics)

    if print_summary:
        print(model.summary(line_length=150))

    model.reset_states()
    return model


def vanilla_binary_normalized(
    inputs,
    encoded_features,
    optimizer,
    loss,
    metrics,
    print_summary=True,
):
    # You needs 'input' and 'encoded_features' as a lists
    inputs_list = list(inputs.values())
    encoded_features_list = list(encoded_features.values())

    all = tf.keras.layers.concatenate(encoded_features_list, name="all")
    x = tf.keras.layers.BatchNormalization(name="batchNorm_1")(all)
    x = tf.keras.layers.Dense(
        units=1024,
        activation="relu",
        name="dense1",
    )(x)
    x = tf.keras.layers.Dense(
        units=512,
        activation="relu",
        name="dense2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="batchNorm_2")(x)
    output = tf.keras.layers.Dense(
        units=1,
        activation="sigmoid",
        name="proba_delay",
    )(x)

    model = tf.keras.Model(
        inputs_list,
        output,
        name="vanilla_binary_normalized",
    )
    model.compile(optimizer, loss, metrics)

    if print_summary:
        print(model.summary(line_length=150))

    model.reset_states()
    return model


def vanilla_multiclass(
    inputs,
    encoded_features,
    optimizer,
    loss,
    metrics,
    print_summary=True,
):
    # You needs 'input' and 'encoded_features' as a lists
    inputs_list = list(inputs.values())
    encoded_features_list = list(encoded_features.values())

    # Combine input features into a single tensor
    all = tf.keras.layers.concatenate(encoded_features_list, name="all")
    batch_1 = tf.keras.layers.BatchNormalization(name="batchNorm_1")(all)

    # One shared lauers for two branches
    dense_1 = tf.keras.layers.Dense(
        units=1024,
        activation="relu",
        name="dense_1",
    )(batch_1)
    dense_2 = tf.keras.layers.Dense(
        units=512,
        activation="relu",
        name="dense_2",
    )(dense_1)
    batch_2 = tf.keras.layers.BatchNormalization(name="batchNorm_2")(dense_2)

    # Define model outputs
    output = tf.keras.layers.Dense(
        units=6,
        activation="softmax",
        name="dpd_delay",
    )(batch_2)

    model = tf.keras.Model(
        inputs_list,
        output,
        name="vanilla_multiclass",
    )
    model.compile(optimizer, loss, metrics)

    if print_summary:
        print(model.summary(line_length=150))

    model.reset_states()
    return model


def vanilla_multioutput(
    inputs,
    encoded_features,
    optimizer,
    loss,
    metrics,
    print_summary=True,
):
    # You needs 'input' and 'encoded_features' as a lists
    inputs_list = list(inputs.values())
    encoded_features_list = list(encoded_features.values())

    # Combine input features into a single tensor
    all = tf.keras.layers.concatenate(encoded_features_list, name="all")
    batch_1 = tf.keras.layers.BatchNormalization(name="batchNorm_1")(all)

    # One shared lauers for two branches
    dense_1 = tf.keras.layers.Dense(
        units=1024,
        activation="relu",
        name="dense_1",
    )(batch_1)
    dense_2 = tf.keras.layers.Dense(
        units=512,
        activation="relu",
        name="dense_2",
    )(dense_1)
    batch_2 = tf.keras.layers.BatchNormalization(name="batchNorm_2")(dense_2)

    # Binary classification branch
    out_proba_delay = tf.keras.layers.Dense(
        units=1,
        activation="sigmoid",
        name="proba_delay",
    )(batch_2)

    # Multiclass classification branch
    out_dpd_delay = tf.keras.layers.Dense(
        units=6,
        activation="softmax",
        name="dpd_delay",
    )(batch_2)

    outputs = [out_proba_delay, out_dpd_delay]
    model = tf.keras.Model(
        inputs_list,
        outputs,
        name="vanilla_multioutput",
    )
    model.compile(optimizer, loss, metrics)

    if print_summary:
        print(model.summary(line_length=150))

    model.reset_states()
    return model


def looped_multioutput(
    inputs,
    encoded_features,
    optimizer,
    loss,
    metrics,
    print_summary=True,
):
    # You needs 'input' and 'encoded_features' as a lists
    inputs_list = list(inputs.values())
    encoded_features_list = list(encoded_features.values())

    # Combine input features into a single tensor
    all = tf.keras.layers.concatenate(encoded_features_list, name="all")
    batch_1 = tf.keras.layers.BatchNormalization(name="btchNrm_1")(all)

    # One shared lauers for two branches
    dense_1 = tf.keras.layers.Dense(
        units=1024,
        activation="relu",
        name="dense_1",
    )(batch_1)
    dense_2 = tf.keras.layers.Dense(
        units=512,
        activation="relu",
        name="dense_2",
    )(dense_1)
    batch_2 = tf.keras.layers.BatchNormalization(name="btchNrm_2")(dense_2)

    # Binary classification branch
    out_proba_delay = tf.keras.layers.Dense(
        units=1,
        activation="sigmoid",
        name="proba_delay",
    )(batch_2)

    # Multiclass classification branch
    dense_3 = tf.keras.layers.Dense(
        units=512,
        activation="relu",
        name="dense_3",
    )(batch_2)
    concat = tf.keras.layers.concatenate(
        [dense_3, out_proba_delay], name="concat"
    )
    batch_3 = tf.keras.layers.BatchNormalization(name="btchNrm_3")(concat)
    out_dpd_delay = tf.keras.layers.Dense(
        units=6,
        activation="softmax",
        name="dpd_delay",
    )(batch_3)

    outputs = [out_proba_delay, out_dpd_delay]
    model = tf.keras.Model(
        inputs_list,
        outputs,
        name="looped_multioutput",
    )
    model.compile(optimizer, loss, metrics)

    if print_summary:
        print(model.summary(line_length=150))

    model.reset_states()
    return model
