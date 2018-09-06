"""
Even out the number of training examples for each class, as there was some
class inbalance.
This method undersamples the over-represented classes to number of training
examples in the smallest class.
----------------------
author: Scott Warchal
date  : 2018-09-03
"""

import os
import math
import random
import shutil

BASE_PATH = "/mnt/datastore/scott/2018-04-24_nncell_data_300"

CELL_LINES = (
    "HCC1569",
    "HCC1954",
    "KPL4",
    "MCF7",
    "MDA-157",
    "MDA-231",
    "SKBR3",
    "T47D"
)

def get_path_to_training(cell_line):
    full_path = f"{BASE_PATH}_{cell_line}"
    return os.path.join(full_path, "train")


def get_path_to_test(cell_line):
    full_path = f"{BASE_PATH}_{cell_line}"
    return os.path.join(full_path, "test")


def get_lowest_count(cell_line):
    lowest = math.inf
    train_path = get_path_to_training(cell_line)
    groups = os.listdir(train_path)
    for group in groups:
        group_path = os.path.join(train_path, group)
        n_files = len(os.listdir(group_path))
        if n_files < lowest:
            lowest = n_files
    return lowest


def copy_test(cell_line):
    test_path = get_path_to_test(cell_line)
    groups = os.listdir(test_path)
    for group in groups:
        group_path = os.path.join(test_path, group)
        new_group_path = replace_dir_path(group_path)
        print(group_path, new_group_path, sep=" => ")
        shutil.copytree(group_path, new_group_path)


def replace_dir_path(path,
    orig_prefix=BASE_PATH,
    new_prefix="/mnt/datastore/scott/2018-09-04_nncell_equal_300_"):
    return path.replace(orig_prefix, new_prefix)


def main():
    for cell_line in CELL_LINES:
        lowest_n = get_lowest_count(cell_line)
        train_path = get_path_to_training(cell_line)
        copy_test(cell_line)
        for group in os.listdir(train_path):
            group_path = os.path.join(train_path, group)
            files_in_group = os.listdir(group_path)
            random.shuffle(files_in_group) # in-place shuffle
            files_in_group_subset = files_in_group[:lowest_n]
            for f in files_in_group_subset:
                indv_file_path = os.path.join(group_path, f)
                new_indv_file_path = replace_dir_path(indv_file_path)
                print(indv_file_path, new_indv_file_path, sep=" => ")
                os.makedirs(os.path.dirname(new_indv_file_path), exist_ok=True)
                shutil.copy(indv_file_path, new_indv_file_path)


if __name__ == "__main__":
    main()