import os
import torch
import dataset


def make_datasets(dataset_path, transforms=None, **kwargs):
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
    return dataset.CellDataset(dataset_path, **kwargs)


def make_dataloaders(datasets, batch_size=32, num_workers=8):
    """
    Parameters:
    -----------
    datasets_dict: CellDataset
    batch_size: int
        number of images per batch
    num_workers: int
        number of sub-processes used to pre-load the images
    Returns:
    --------
    DataLoader
    """
    return torch.utils.data.DataLoader(datasets, batch_size=batch_size,
                                       num_workers=num_workers)