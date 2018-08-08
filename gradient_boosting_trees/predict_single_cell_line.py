"""
predict moa and create confusion matrix for single cell-line with
a Gradient Boosting Classifier
"""

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


def preprocess_data(data, cell_line):
    data_cl = data[data.Metadata_cell_line == cell_line]
    data_cl = data.query("Metadata_compound_type == 'drug'")
    featuredata = data_cl[morar.get_featuredata(data_cl)]
    moa_classes = [moa_dict[i] for i in data_cl.Metadata_compound]
    moa_classes = make_labels_consistent(moa_classes)
    return sklearn.model_selection.train_test_split(featuredata, moa_classes)


def make_labels_consistent(labels):
    label_dict = {
        "DNA damaging agent": "dna damaging",
        "actin disrupting": "actin",
        "aurora B inhibitor": "aurora",
        "kinase inhibitor": "kinase",
        "microtubule disrupting": "microtubule",
        "protein degradation": "protein deg",
        "protein synthesis": "protein synth",
        "statin": "statin"
    }
    return [label_dict[i] for i in labels]


def make_cm():
    output = namedtuple("ConfusionMatrix",
                        ["cm", "acc", "labels", "cell_line"])
    cell_line = sys.argv[1]
    data = pd.read_hdf("single_cell_data.hdf", "data_norm_agg")
    X_train, X_test, y_train, y_test = preprocess_data(data, cell_line)
    clf = GradientBoostingClassifier(n_estimators=600)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    acc = sklearn.metrics.accuracy_score(y_test, predictions)
    cm = sklearn.metrics.confusion_matrix(y_test, predictions)
    labels = sorted(list(set(y_train)))
    return output(cm, acc, labels, cell_line)


def norm_cm(cm):
    return cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]


def main():
    cm = make_cm()
    plt.figure()
    plt.grid(linestyle=":")
    plt.imshow(norm_cm(cm.cm), vmin=0, vmax=1, cmap=plt.cm.bone_r)
    plt.xticks(range(len(cm.labels)), cm.labels, rotation=90)
    plt.yticks(range(len(cm.labels)), cm.labels)
    plt.title("{}\naccuracy = {:.2f}%".format(cm.cell_line, cm.acc * 100),
              loc="left", fontweight="bold")
    plt.tight_layout()
    plt.savefig("{}_rf_cm.pdf".format(cm.cell_line))


if __name__ == "__main__":
    main()
