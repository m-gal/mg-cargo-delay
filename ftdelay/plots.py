""" Contains the functions used for plotting.

    @author: mikhail.galkin
"""

#%% Import needed python libraryies and project config info
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import ConfusionMatrixDisplay

#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~ P L O T S ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def plot_history_loss(history, nn_name, loss="loss"):
    plt.figure(figsize=(12, 8))
    # * Var#1: Use a log scale to show the wide range of values.
    # plt.semilogy(
    #     history.epoch, history.history[loss], label="Train: " + nn_name
    # )
    # plt.semilogy(
    #     history.epoch,
    #     history.history[f"val_{loss}"],
    #     label="Val: " + nn_name,
    #     linestyle="--",
    # )
    # * Var#2: W/o a log scaling.
    plt.plot(history.epoch, history.history[loss], label="Train: " + nn_name)
    plt.plot(
        history.epoch,
        history.history[f"val_{loss}"],
        label="Val: " + nn_name,
        linestyle="--",
    )

    plt.title(nn_name + f": model {loss}")
    plt.xlabel("epoch")
    plt.ylabel(loss)
    plt.legend()
    # get current figure
    fig = plt.gcf()
    plt.close()
    return fig


def plot_history_metrics(history, nn_name):
    metrics = [x for x in history.history.keys() if x[:3] != "val"]
    nrows = math.ceil(len(metrics) / 2)
    ncols = 2

    plt.figure(figsize=(16, 6 * nrows))
    for n, metric in enumerate(metrics):
        print
        plt.subplot(nrows, ncols, n + 1)
        plt.plot(
            history.epoch,
            history.history[metric],
            label="Train",
        )
        plt.plot(
            history.epoch,
            history.history["val_" + metric],
            linestyle="--",
            label="Val",
        )
        plt.xlabel("epoch")
        plt.ylabel(metric)
        y_min = 0.99 * min(
            min(history.history[metric]), min(history.history["val_" + metric])
        )
        y_max = 1.01 * max(
            max(history.history[metric]), max(history.history["val_" + metric])
        )
        plt.ylim([y_min, y_max])
        plt.legend()

    # get current figure
    fig = plt.gcf()

    # create main title
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.suptitle(nn_name, fontsize="x-large")

    plt.close()
    return fig


def plot_cm(target, prediction, data_name, nn_name, p=0.5):
    ConfusionMatrixDisplay.from_predictions(
        y_true=target,
        y_pred=prediction > p,
        normalize="true",
    )
    plt.title(
        nn_name + ": " + data_name + ": confusion matrix @{:.2f}".format(p)
    )
    plt.grid(False)
    # get current figure
    fig = plt.gcf()
    plt.close()
    return fig


def plot_confusion_matrix(
    cm,
    group_names=None,
    categories="auto",
    count=True,
    percent="by_true",
    cbar=True,
    xyticks=True,
    xyplotlabels=True,
    sum_stats=True,
    figsize=(9, 9),
    cmap="viridis",
    title=None,
):
    """
    This function will make a pretty plot of an sklearn Confusion Matrix cm
    using a Seaborn heatmap visualization.
    Arguments
    ---------
    cm:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    percent:       {'by_true', 'by_pred', 'by_all'}, default=None.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                    Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                    See http://matplotlib.org/examples/color/colormaps_reference.html
    title:         Title for the heatmap. Default is None.
    """

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cm) / float(np.sum(cm))

        # if it is a binary confusion matrix, show some more stats
        if len(cm) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cm[1, 1] / sum(cm[:, 1])
            recall = cm[1, 1] / sum(cm[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = f"\n\nAccuracy={accuracy:0.4f}\nPrecision={precision:0.4f}\nRecall={recall:0.4f}\nF1 Score={f1_score:0.4f}"
        else:
            stats_text = f"\n\nAccuracy={accuracy:0.3f}"
    else:
        stats_text = ""

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ["" for i in range(cm.size)]

    if group_names and len(group_names) == cm.size:
        group_labels = [f"{value}\n" for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = [f"{value:,}\n" for value in cm.flatten()]
    else:
        group_counts = blanks

    if percent == "by_true":
        group_prc = (cm / cm.sum(axis=1, keepdims=True)).flatten().tolist()
        group_prc = np.nan_to_num(group_prc)
        group_prc = [f"{value:.2%}" for value in group_prc]
        cm = cm / cm.sum(axis=1, keepdims=True)
    elif percent == "by_pred":
        group_prc = (cm / cm.sum(axis=0, keepdims=True)).flatten().tolist()
        group_prc = np.nan_to_num(group_prc)
        group_prc = [f"{value:.2%}" for value in group_prc]
        cm = cm / cm.sum(axis=0, keepdims=True)
    elif percent == "by_all":
        group_prc = (cm / cm.sum()).flatten().tolist()
        group_prc = np.nan_to_num(group_prc)
        group_prc = [f"{value:.2%}" for value in group_prc]
        cm = cm / cm.sum()
    else:
        group_prc = blanks

    box_labels = [
        f"{v1}{v2}{v3}".strip()
        for v1, v2, v3 in zip(group_labels, group_counts, group_prc)
    ]
    box_labels = np.asarray(box_labels).reshape(cm.shape[0], cm.shape[1])

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get("figure.figsize")

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=box_labels,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        xticklabels=categories,
        yticklabels=categories,
    )

    if xyplotlabels:
        plt.ylabel("True label")
        plt.xlabel("Predicted label" + stats_text)
    else:
        plt.xlabel(stats_text)

    if sum_stats:
        plt.subplots_adjust(bottom=0.2)

    if title:
        plt.title(title)

    fig = plt.gcf()
    plt.close()
    return fig


def plot_3roc(
    train_target,
    train_prediction,
    val_target,
    val_prediction,
    test_target,
    test_prediction,
    nn_name,
):

    train_fp, train_tp, _ = roc_curve(train_target, train_prediction)
    train_auc = round(roc_auc_score(train_target, train_prediction), 5)
    val_fp, val_tp, _ = roc_curve(val_target, val_prediction)
    val_auc = round(roc_auc_score(val_target, val_prediction), 5)
    test_fp, test_tp, _ = roc_curve(test_target, test_prediction)
    test_auc = round(roc_auc_score(test_target, test_prediction), 5)

    plt.figure(figsize=(9, 9))
    plt.plot(
        100 * train_fp,
        100 * train_tp,
        label="train: auc " + str(train_auc),
        linewidth=2,
    )
    plt.plot(
        100 * val_fp,
        100 * val_tp,
        label="val: auc " + str(val_auc),
        linewidth=2,
    )
    plt.plot(
        100 * test_fp,
        100 * test_tp,
        label="test: auc " + str(test_auc),
        linewidth=2,
    )
    plt.plot([0, 100], [0, 100], "r--")
    plt.title(nn_name + ": ROC-AUC")
    plt.xlabel("false positives [%]")
    plt.ylabel("true positives [%]")
    plt.xlim([-0.5, 100])
    plt.ylim([0, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.legend(loc="lower right")
    # get current figure
    fig = plt.gcf()
    plt.close()
    return fig


def plot_multiclass_roc(
    y_true,
    y_pred,
    labels,
    data_name,
    nn_name,
    average="macro",
):
    lb = LabelBinarizer()
    lb.fit(y_true)
    y_true = lb.transform(y_true)
    y_pred = lb.transform(y_pred)

    fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))
    for (idx, c_label) in enumerate(labels):
        fpr, tpr, thresholds = roc_curve(
            y_true[:, idx].astype(int),
            y_pred[:, idx],
        )
        label = f"{c_label}: auc {auc(fpr, tpr):0.5f}"
        c_ax.plot(fpr, tpr, label=label)
    c_ax.plot(fpr, fpr, "r--")
    c_ax.legend()
    c_ax.set_xlabel("false positives [%]")
    c_ax.set_ylabel("true positives [%]")
    auc_score = roc_auc_score(y_true, y_pred, average=average)
    plt.title(f"{nn_name}: {data_name}: ROC-AUC: {auc_score:0.6f}")
    # plt.show()
    fig = plt.gcf()
    plt.close()

    return fig


def plot_density_train_test(
    train_target,
    train_prediction,
    test_target,
    test_prediction,
    nn_name,
):
    fig = plt.figure(figsize=(12, 6))
    d = pd.concat(
        [
            pd.DataFrame(
                {
                    "target": train_target.ravel(),
                    "prediction": train_prediction.ravel(),
                    "data": "train",
                }
            ),
            pd.DataFrame(
                {
                    "target": test_target.ravel(),
                    "prediction": test_prediction.ravel(),
                    "data": "test",
                }
            ),
        ]
    ).groupby(["target", "data"])

    q50_test_0 = d.quantile(q=0.50).values[0]
    q50_test_1 = d.quantile(q=0.50).values[2]
    q50_train_0 = d.quantile(q=0.50).values[1]
    q50_train_1 = d.quantile(q=0.50).values[3]

    d.prediction.plot.kde()
    plt.title(nn_name + ": density chart")
    plt.xlabel("predicted probability")
    plt.legend()

    plt.axvline(x=q50_test_0, ls="--", lw=1, c="tab:blue")
    plt.text(q50_test_0, 0, "Q50=" + str(q50_test_0.round(4)), rotation=90)
    plt.axvline(x=q50_train_0, ls="--", lw=1, c="tab:orange")

    plt.axvline(x=q50_test_1, ls="--", lw=1, c="tab:green")
    plt.text(
        q50_test_1 - 0.02, 0, "Q50=" + str(q50_test_1.round(4)), rotation=90
    )
    plt.axvline(x=q50_train_1, ls="--", lw=1, c="tab:red")

    # get current figure
    fig = plt.gcf()
    plt.close()
    return fig


def plot_density(target, prediction, nn_name, ds_name="Test"):
    """_summary_

    Args:
        target (np.array): _description_
        prediction (np.array): _description_
        nn_name (_type_): _description_
        ds_name (str, optional): _description_. Defaults to "Test".

    Returns:
        _type_: _description_
    """
    fig = plt.figure(figsize=(12, 6))
    d = pd.DataFrame(
        {
            "target": target.ravel(),
            "prediction": prediction.ravel(),
            "data": ds_name,
        }
    ).groupby(["target", "data"])

    q50_test_0 = d.quantile(q=0.50).values[0]
    q50_test_1 = d.quantile(q=0.50).values[1]

    d.prediction.plot.kde()
    plt.title(f"{nn_name}: density chart: {ds_name}")
    plt.xlabel("predicted probability")
    plt.legend()

    plt.axvline(x=q50_test_0, ls="--", lw=1, c="tab:blue")
    plt.text(q50_test_0, 0, "Q50=" + str(q50_test_0.round(4)), rotation=90)
    plt.axvline(x=q50_test_1, ls="--", lw=1, c="tab:green")
    plt.text(
        q50_test_1 - 0.02, 0, "Q50=" + str(q50_test_1.round(4)), rotation=90
    )

    # get current figure
    fig = plt.gcf()
    plt.close()
    return fig
