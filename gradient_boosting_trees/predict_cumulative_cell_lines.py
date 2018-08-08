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


def preprocess_data(data, cell_lines):
    data_cl = data[data.Metadata_cell_line.isin(cell_lines)]
    data_cl = data.query("Metadata_compound_type == 'drug'")
    moa_classes = [moa_dict[i] for i in data_cl.Metadata_compound]
    moa_classes = make_labels_consistent(moa_classes)
    data_cl["Metadata_moa_class"] = moa_classes
    # split MDA-231 test set
    data_no_231 = data_cl[data_cl.Metadata_cell_line != "MDA-231"]
    data_231 = data_cl[data_cl.Metadata_cell_line == "MDA-231"]
    train_231, test_231 = sklearn.model_selection.train_test_split(data_231)
    # put MDA-231 training set back in with other cell-lines
    data_no_test_231 = data_no_231.append(train_231)
    X_train = data_no_test_231[morar.get_featuredata(data_no_test_231)]
    X_test = test_231[morar.get_featuredata(test_231)]
    y_train = data_no_test_231.Metadata_moa_class
    y_test = test_231.Metadata_moa_class
    return X_train, X_test, y_train, y_test


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


def main():
    cell_lines = sys.argv[1:]
    data = pd.read_hdf("single_cell_data.hdf", "data_norm_agg")
    X_train, X_test, y_train, y_test = preprocess_data(data, cell_lines)
    clf = GradientBoostingClassifier(n_estimators=10)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    acc = sklearn.metrics.accuracy_score(y_test, predictions)
    joined_cell_lines = "_".join(cell_lines)
    output_name = "output_{}".format(joined_cell_lines)
    with open(output_name, "w") as f:
        output = "{}\t{}\t{}\n".format(len(cell_lines), acc, cell_lines)
        f.write(output)



if __name__ == "__main__":
    main()
