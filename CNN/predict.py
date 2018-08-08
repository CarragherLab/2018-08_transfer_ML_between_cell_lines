"""
module docstring
"""

import os
import sys
from collections import OrderedDict
import torch
import resnet
import dataset
import model_utils
import data_utils

USE_GPU = torch.cuda.is_available()
NUM_CLASSES = 8


def load_model_weights(model, path_to_state_dict):
    """
    Load a model with a given state (pre-trained weights)

    Parameters:
    -----------
    model: pytorch model
    path_to_state_dict: string

    Returns:
    ---------
    pytorch model with weights loaded to state_dict
    """
    if USE_GPU:
        model_state = torch.load(path_to_state_dict)
    else:
        # need to map storage loc to cpu
        model_state = torch.load(
            path_to_state_dict, map_location=lambda storage, loc: storage
        )
    if model_utils.is_distributed_model(model_state):
        model_state = model_utils.strip_distributed_keys(model_state)
    model.load_state_dict(model_state)
    model.eval()
    if USE_GPU:
        model = model.cuda()
    return model


def make_label_dict(data_dir):
    """
    docstring
    Parameters:
    -----------
    data_dir: string
        path to directory containing sub-directories of classes and data
    
    Returns:
    ---------
    dictionary:
        {index => int: class_label => string}
    """
    cell_dataset = dataset.CellDataset(data_dir)
    # reverse {label: index} dict used within CellDataset class
    return {v: k for k, v in cell_dataset.label_dict.items()}


def parse_name(name):
    """
    extract the important stuff from the array filename.
    Returning the parent image the file is from.
    Parameters:
    ------------
    name: string
        file path of the numpy array

    Returns:
    ---------
    string:
        e.g "MCF7_img_13_90.npy"
        will return "MCF7_13"
    """
    # remove the file suffix
    assert name.endswith(".npy")
    cell_line, _, img_num, _ = name.split(".")[0].split("_")
    return "_".join([cell_line, img_num])


def main():
    data_dir, path_to_weights = sys.argv[1:]
    model = resnset.resnet18(num_classes=NUM_CLASSES)
    model = load_model_weights(model, path_to_weights)
    data = data_utils.make_datasets(data_dir)
    dataloader = data_utils.make_dataloaders(data)
    label_dict = make_label_dict(data_dir)
    # print as tab-delimited file
    print("predicted", "actual", "img_name", sep="\t")
    for batch in dataloader:
        inputs, labels, img_names = batch
        if USE_GPU:
            inputs = torch.autograd.Variable(inputs.cuda())
            labels = torch.autograd.Variable(labels.cuda())
        else:
            inputs = torch.autograd.Variable(inputs)
            labels = torch.autograd.Variable(labels)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        labels = labels.view(-1)
        parsed_img_names = [parse_name(i) for i in img_names]
        batch_actual_labels = [label_dict[i.data[0]] for i in list(labels)]
        batch_predicted_labels = [label_dict[i] for i in list(preds)]
        for predicted, actual, img_name in zip(
            batch_predicted_labels, batch_actual_labels, parsed_img_names
        ):
            print(predicted, actual, img_name, sep="\t", flush=True)


if __name__ == "__main__":
    main()

