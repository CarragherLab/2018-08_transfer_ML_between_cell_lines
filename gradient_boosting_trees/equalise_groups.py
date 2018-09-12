"""
Predict moa and create confusion matrix for single cell-line with
a Gradient Boosting Classifier (GBC), taking into account unequal class
sizes by down-sampling over-represented classes.
"""

import os
import sys
from collections import namedtuple
import pandas as pd
import numpy as np
import sklearn.metrics
import sklearn.model_selection
from sklearn.ensemble import GradientBoostingClassifier
import morar
from moa_dict import moa_dict
import matplotlib.pyplot as plt


plt.style.use(["seaborn-paper", "seaborn-ticks"])

data_dir = "/home/scott/2018-08-01_random_forest_moa"
data_path = os.path.join(data_dir, "single_cell_data.hdf")
data = pd.read_hdf(data_path, "data_norm_agg")


def find_smallest_group(dataframe):
    """
    return the number of samples in the smallest moa class
    Returns: int
    """
    return dataframe.groupby("Metadata_moa_class").size().min()


def preprocess_data(data, cell_line):
    data_cl = data.query(
        "Metadata_cell_line == @cell_line and Metadata_compound_type == 'drug'"
    )
    moa_classes = [moa_dict[i] for i in data_cl.Metadata_compound]
    moa_classes = make_labels_consistent(moa_classes)
    data_cl["Metadata_moa_class"] = moa_classes
    return data_cl


def equalise_classes(dataframe, n):
    """
    Equalise moa classes so there are `n` examples of each moa
    class in `dataframe`
    """
    grouped = dataframe.groupby("Metadata_moa_class", as_index=False)
    return grouped.apply(lambda x: x.sample(n=n))


def make_labels_consistent(labels):
    label_dict = {
        "DNA damaging agent"    : "dna damaging",
        "actin disrupting"      : "actin",
        "aurora B inhibitor"    : "aurora",
        "kinase inhibitor"      : "kinase",
        "microtubule disrupting": "microtubule",
        "protein degradation"   : "protein deg",
        "protein synthesis"     : "protein synth",
        "statin"                : "statin"
    }
    return [label_dict[i] for i in labels]


def train_predict(dataframe, cell_line):
    output = namedtuple(
        "ConfusionMatrix", ["cm", "acc", "labels", "cell_line"]
    )
    featuredata = dataframe[morar.get_featuredata(dataframe)]
    labels = dataframe["Metadata_moa_class"]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        featuredata, labels
    )
    clf = GradientBoostingClassifier(n_estimators=600)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    acc = sklearn.metrics.accuracy_score(y_test, predictions)
    cm = sklearn.metrics.confusion_matrix(y_test, predictions)
    cm = norm_cm(cm)
    labels = sorted(list(set(y_train)))
    return output(cm, acc, labels, cell_line)


def norm_cm(cm):
    return cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]


def plot_result(cm):
    """
    where `cm` is the namedtuple produced by `train_predict`
    """
    plt.figure()
    plt.grid(linestyle=":")
    plt.imshow(cm.cm, vmin=0, vmax=1, cmap=plt.cm.bone_r)
    plt.xticks(range(len(cm.labels)), cm.labels, rotation=90)
    plt.yticks(range(len(cm.labels)), cm.labels)
    plt.title("{}\naccuracy = {:.2f}%".format(cm.cell_line, cm.acc * 100),
              loc="left", fontweight="bold")
    plt.tight_layout()
    plt.savefig("{}_rf_cm.pdf".format(cm.cell_line))



def main():
    assert len(sys.argv) == 2, "need to pass cell line as an argument"
    cell_line = sys.argv[1]
    data_post = preprocess_data(data, cell_line)
    smallest = find_smallest_group(data_post)
    data_equal = equalise_classes(data_post, smallest)
    result = train_predict(data_equal, cell_line)
    plot_result(result)


if __name__ == "__main__":
    main()