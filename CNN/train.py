"""
author: Scott Warchal
date: 2018-04-24

Train ResNet CNN on 300px*300px*5channel numpy stacks.
Tensors are converted to 244*244 (w*h) to fit with the original ResNet
architecture.

command line args:
               1 => path to directory containing training test sets
    (optional) 2 => path to save location,
                    defaults to "/exports/eddie/scratch/$USER_$DATE"
"""

import sys
import os
import time
import datetime
import copy
import json
import torch
import torchvision
import resnet
import dataset
import transforms

BATCH_SIZE = 32
NUM_EPOCHS = 20
NUM_CLASSES = 8
NUM_WORKERS = 8
USE_GPU = True
DROPOUT = False
USER = os.environ["USER"]
DATE = str(datetime.datetime.now().date())
SAVE_PATH = sys.argv[2] if len(sys.argv) == 3 else "/exports/eddie/scratch/{}/{}".format(USER, DATE)
VERBOSE = True


def make_datasets(top_level_data_dir, transforms=None):
    """
    Parameters:
    -----------
    top_level_data_dir: string
        directory path which contains train and test sub-directories
    transforms: transformation dictionary (default is None)
        if not None, then should be a dictionary of transformations, with
        an entry for training transforms and testing transforms.

    Returns:
    --------
    Dictionary containing
        {"train": CellDataset,
         "test" : CellDataset}
    """
    dataset_dict = {}
    for phase in ["train", "test"]:
        dataset_path = os.path.join(top_level_data_dir, phase)
        if transforms is not None:
            print("INFO: images will be randomly rotated in training")
            dataset_dict[phase] = dataset.CellDataset(
                dataset_path,
                transforms=transforms if phase == "train" else None
            )
        else:
            dataset_dict[phase] = dataset.CellDataset(dataset_path)
    return dataset_dict


def make_dataloaders(datasets_dict):
    """
    Parameters:
    -----------
    datasets_dict: dictionary
        dictionary created from make_datasets()

    Returns:
    --------
    Dictionary of DataLoaders
        {"train": DataLoader,
         "test" : DataLoader}
    """
    dataloader_dict = {}
    for phase in ["train", "test"]:
        dataloader_dict[phase] = torch.utils.data.DataLoader(
            datasets_dict[phase], batch_size=BATCH_SIZE,
            shuffle=True if phase == "train" else False,
            num_workers=NUM_WORKERS
        )
    return dataloader_dict


def exp_lr_schedular(optimizer, epoch, init_lr=0.001, lr_decay_epoch=10):
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


def print_model_stats(index, batch_size, len_data, loss):
    """print running model statistics"""
    x = "    {n}/{d}: loss = {loss:.4f}          "
    print(x.format(n=index*batch_size,
                   d=len_data,
                   loss=loss),
          end="\r", flush=True)


def train_model(model, criterion, optimizer, lr_scheduler):
    """docstring"""
    time_at_start = time.time()
    best_model = model
    best_acc = 0.0
    history = {"train_acc": [], "train_loss": [],
               "test_acc": [], "test_loss":[]}
    transform = transforms.RandomRotate()
    datasets = make_datasets(sys.argv[1], transforms=transform)
    ##
    dataloader = make_dataloaders(datasets)

    for epoch in range(NUM_EPOCHS):
        print("Epoch {}/{}".format(epoch, NUM_EPOCHS-1))
        print("="*10)
        # each epoch has training and validation phases
        for phase in ["train", "test"]:
            if phase == "train":
                optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)
            else:
                model.train(False)
            print("Setting model to {phase} mode".format(phase=phase))

            running_loss = 0.0
            running_corrects = 0
            len_data = len(datasets[phase])

            for index, data in enumerate(dataloader[phase]):
                inputs, labels = data
                if USE_GPU:
                    inputs = torch.autograd.Variable(inputs.cuda())
                    labels = torch.autograd.Variable(labels.cuda())
                else:
                    inputs = torch.autograd.Variable(inputs)
                    labels = torch.autograd.Variable(labels)

                # zero the parameter gradients before the forward pass
                optimizer.zero_grad()

                # forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                labels = labels.view(-1)
                loss = criterion(outputs, labels)

                # backprop if in the training phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                if VERBOSE:
                    print_model_stats(index, BATCH_SIZE, len_data, loss.data[0])
            print("\n")

            # epoch stats for train and validation phases
            epoch_loss = running_loss / len_data
            epoch_acc  = running_corrects / len_data
            history["{}_acc".format(phase)].append(epoch_acc)
            history["{}_loss".format(phase)].append(epoch_loss)
            print("{} Loss: {:.4f} | Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)
                model_path = "{}_checkpoint".format(SAVE_PATH)
                print("checkpointing model at {}".format(model_path))
                torch.save(model.state_dict(), model_path)

            # write history dict as a JSON file at each epoch
            history_path = "{}_history".format(SAVE_PATH)
            with open(history_path, "w") as f:
                json.dump(history, f, indent=4)

    time_elapsed = time.time() - time_at_start
    print("Training complete in {:.0f}m {:.0f}s".format(
          time_elapsed // 60, time_elapsed % 60))
    print("Best validation accuracy: {:.4f}".format(best_acc))


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
    optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.005)
    train_model(model_ft, criterion, optimizer_ft, exp_lr_schedular)


if __name__ == "__main__":
    main()

