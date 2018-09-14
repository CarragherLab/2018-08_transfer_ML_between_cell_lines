"""
Transfer learning test

See if a model trained on 7 cell-lines improves accuracy when
fine-tuned with a few training examples of the new cell-line

Will do this by fixing all the layers except the final classification layer
"""

import os
import sys
import time
import json
import copy
import torch
import resnet
from predict import load_model_weights
import model_utils
import dataset
import transforms


WEIGHTS  = sys.argv[1]
DATA_DIR = sys.argv[2]
SAVE_PATH = sys.argv[3] if len(sys.argv) == 4 else "/home/scott/transfer_learning"
NUM_EPOCHS = 20
NUM_WORKERS = 8
N_GPU = torch.cuda.device_count()
USE_GPU = torch.cuda.is_available()
BATCH_SIZE = 32 * N_GPU
LEARNING_RATE = 1e-5
VERBOSE = True


def load_model(weights):
    """
    load resnet18 model with pre-trained weights

    Parameters:
    ---------------
    weights: path to resnet18 state_dict

    Returns:
    --------
    model with loaded state-dicti
 
 
    """
    model = resnet.resnet18(num_classes=8)
    return model_utils.load_model_weights(model, weights, strip_keys=False)


def freeze_model_weights(model):
    """
    1. freeze model weights to stop convolutional layers from training
    2. replace final classification layer with randomised weights and
       and allow training of this final layer
    """
    for param in model.parameters():
        param.requires_grad = False
    # replace final classification layer
    # replaced with the same, but removes trained weights from the pre-trained
    # model and sets requires_grad = True
    input_dimension  = 2048
    output_dimension = 8
    model.fc = torch.nn.Linear(input_dimension, output_dimension)
    return model


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
            num_workers=NUM_WORKERS, pin_memory=True
        )
    return dataloader_dict


def exp_lr_schedular(optimizer, epoch, init_lr, lr_decay_epoch=30):
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

def train(model, criterion, optimizer, lr_scheduler, data_dir, lr):
    """
    docstring

    Parameters:
    -----------
    model: pytorch model
    criterion:
    optimizer:
    lr_scheduler:
    data_dir:
    """
    time_at_start = time.time()
    best_acc = 0.0
    history = {"train_acc" : [],
               "train_loss": [],
               "test_acc"  : [],
               "test_loss" : []}
    datasets = make_datasets(data_dir, transforms=transforms.RandomRotate())
    dataloader = make_dataloaders(datasets)
    for epoch in range(NUM_EPOCHS):
        print("Epoch {}/{}".format(epoch, NUM_EPOCHS-1))
        print("="*10)
        for phase in ["train", "test"]:
            if phase == "train":
                optimizer = lr_scheduler(optimizer, epoch, lr)
                model.train(True)
            else:
                model.train(False)
            print("Setting model to {} mode".format(phase))
        
            running_loss = 0.0
            running_corrects = 0
            len_data = len(datasets[phase])

            for index, data in enumerate(dataloader[phase]):
                inputs, labels, _ = data
                if USE_GPU:
                    inputs = torch.autograd.Variable(inputs.cuda())
                    labels = torch.autograd.Variable(labels.cuda())
                else:
                    inputs = torch.autograd.Variable(inputs)
                    labels = torch.autograd.Variable(labels)
                
                # zero parameter gradients before the forward pass
                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                labels = labels.view(-1)
                loss = criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                if VERBOSE:
                    print_model_stats(index, BATCH_SIZE, len_data, loss.data[0])
            print("\n")

            # epoch stats
            epoch_loss = running_loss / len_data
            epoch_acc = running_corrects / len_data
            history["{}_acc".format(phase)].append(epoch_acc)
            history["{}_loss".format(phase)].append(epoch_loss)
            print("{} Loss: {:.4f} | Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # checkpoint the best model based on validation accuracy
            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                model_path = "{}_checkpoint".format(SAVE_PATH)
                print("checkpointing model at {}".format(model_path))
                torch.save(model.state_dict(), model_path)

            # write history to JSON file
            history_path = "{}_history".format(SAVE_PATH)
            with open(history_path, "w") as f:
                json.dump(history, f, indent=4)
            
    time_elapsed = time.time() - time_at_start
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best validation accuracy: {:.4f}".format(best_acc))


def finetune_model(model, lr=1e-5):
    criterion = torch.nn.CrossEntropyLoss()
    # set optimizer to only optimize parameters where requires_grad = True
    # otherwise pytorch raises an error that cannot optimize parameters where requires_grad = False
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    train(model, criterion, optimizer, exp_lr_schedular, data_dir=DATA_DIR, lr=lr)



def main():
    model = load_model(weights=WEIGHTS)
    model = freeze_model_weights(model)
    if USE_GPU:
        print("GPU detected")
        model = model.cuda()
        if N_GPU > 1:
            print("Multiple GPUs detected, parallelising batches...")
            model = torch.nn.DataParallel(model)
    finetune_model(model, lr=LEARNING_RATE)


if __name__ == "__main__":
    main()

