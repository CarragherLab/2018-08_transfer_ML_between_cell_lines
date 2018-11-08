"""
author: Scott Warchal
date: 2018-04-24

Train ResNet CNN on 300px*300px*5channel numpy stacks.
Tensors are converted to 244*244 (w*h) to fit with the original ResNet
architecture.
"""

import sys
import os
import datetime

from collections import Counter

import json
import sklearn.metrics
import torch
import resnet
import pandas as pd
import dataset
import transforms
from tqdm import tqdm

BATCH_SIZE = int(32 * torch.cuda.device_count())
NUM_EPOCHS = 20
NUM_CLASSES = 8
NUM_WORKERS = 20
USE_GPU = torch.cuda.is_available()
DROPOUT = False
USER = os.environ["USER"]
DATE = str(datetime.datetime.now().date())
VERBOSE = True
SAVE_PATH = sys.argv[3]


DATA_DIR = "/exports/eddie/scratch/s1027820/SLAS/chopped"
DATAFRAME = pd.read_csv("/exports/eddie/scratch/s1027820/SLAS/train.csv")
cell_line = sys.argv[1]
drug = sys.argv[2]

TRAIN_CSV = DATAFRAME.query("cell_line == @cell_line and compound != @drug")
TEST_CSV = DATAFRAME.query("cell_line == @cell_line and compound == @drug")


def exp_lr_scheduler(optimizer, epoch, init_lr=1e-6, lr_decay_epoch=10):
    """
    Decay learning rate as epochs progress. Reduces learning rate
    by a factor of 10 every `lr_decay_epoch` number of epochs.

    Parameters:
    ------------
    optimizer: torch.optim optimizer class
    epoch: int
    init_lr: float
        initial learning rate
    lr_decay_epoch: int
        number of epochs between decay steps
    """
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print("LR is set to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return optimizer


def consensus_classify(dataframe):
    """
    Parameters:
    ----------- 
    dataframe: pandas.DataFrame
        model test output
    
    Returns:
    --------
    float: classification accuracy 
    """
    grouped = dataframe.groupby("img_id")
    predictions = []
    true_labels = []
    for name, group in grouped:
        consensus = Counter(group.predicted).most_common()[0][0]
        predictions.append(consensus)
        # sanity check that all cells in an image have the same label
        assert len(group["actual"].unique()) == 1
        actual = group["actual"].iloc[0]
        true_labels.append(actual)
    return sklearn.metrics.accuracy_score(predictions, true_labels)


def train_model(model, criterion, optimizer, lr_scheduler):
    """docstring"""
    history = {"train_acc": [], "train_loss": [],
               "test_acc": [], "test_loss": []}
    transform = transforms.RandomRotate()

    datasets = {}
    train_data = dataset.CSVDataset(
        data_dir=DATA_DIR, csv=TRAIN_CSV,
        complete_csv=DATAFRAME, transforms=transform
    )
    train_data.equalise_groups("MoA", under_sample=False)
    datasets["train"] = train_data
    datasets["test"] = dataset.CSVDataset(
        data_dir=DATA_DIR, csv=TEST_CSV, complete_csv=DATAFRAME
    )

    dataloader = {}
    dataloader["train"] = torch.utils.data.DataLoader(
        datasets["train"], batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    dataloader["test"] = torch.utils.data.DataLoader(
        datasets["test"], batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    for epoch in range(NUM_EPOCHS):
        print("Epoch {}/{}".format(epoch, NUM_EPOCHS-1))
        print("="*10)
        # each epoch has training and validation phases
        optimizer = lr_scheduler(optimizer, epoch)
        model.train(True)

        running_loss = 0.0
        running_corrects = 0
        len_data = len(datasets["train"])

        for data in tqdm(dataloader["train"]):
            # ignore parent_img labels during training, these are only needed in testing
            inputs, labels, _ = data
            if USE_GPU:
                inputs = torch.autograd.Variable(inputs).cuda()
                labels = torch.autograd.Variable(labels).cuda()
            else:
                inputs = torch.autograd.Variable(inputs)
                labels = torch.autograd.Variable(labels)

            # zero the parameter gradients before the forward pass
            optimizer.zero_grad()

            # forward pass
            outputs  = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            labels   = labels.view(-1)
            loss     = criterion(outputs, labels)

            # backprop if in the training phase
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data)
        print("\n")

        # epoch stats for train and validation phases
        epoch_loss = running_loss / len_data
        epoch_acc  = running_corrects / len_data
        history["train_acc"].append(epoch_acc)
        history["train_loss"].append(epoch_loss)
        print("Loss: {:.4f} | Acc: {:.4f}".format(epoch_loss, epoch_acc))

        # write history dict as a JSON file at each epoch
        history_path = "{}_history".format(SAVE_PATH)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)

    # Convert model to test mode and make predictions on test set
    # these need to be recorded so that the aggregate prediction
    # on all the cells in an image can be calculated.
    #
    # This can be done by grouping on the 'img_id' column in the .csv file
    print("\n")
    print("=" * 10)
    print("Testing")
    print("=" * 10)
    model.eval()
    parent_imgs = []
    predictions_list = []
    actual_vals = []
    for data in tqdm(dataloader["test"]):
        test_inputs, test_labels, parent_img = data
        test_labels = list(test_labels.cpu().numpy())
        if USE_GPU:
            test_inputs = torch.autograd.Variable(test_inputs).cuda()
        test_outputs = model(test_inputs)
        _, predictions = torch.max(test_outputs.data, 1)
        predictions = list(predictions.cpu().numpy())
        parent_imgs.append(parent_img)
        predictions_list.extend(predictions)
        actual_vals.extend(test_labels)
    for i, j, k in zip(parent_imgs, predictions_list, actual_vals):
        print(i, j, k)



def main():
    model_ft = resnet.resnet18(num_classes=NUM_CLASSES, dropout=DROPOUT)
    if USE_GPU:
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            # parallelise across multiple GPUs
            print("Multiple GPUs detected:")
            print("\tsplitting batches across {} GPUs".format(n_gpus))
            model_ft = torch.nn.DataParallel(model_ft)
        model_ft = model_ft.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=1e-6)
    train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler)


if __name__ == "__main__":
    main()

