"""
author: Scott Warchal
date: 2018-04-24

"""


import glob
import os
import numpy as np
from skimage import transform as ski_transform
import torch
from torch import Tensor
from torch.utils.data import Dataset
import pandas as pd


class CellDataset(Dataset):
    """
    Custom Dataset for structured directory of numpy arrays to work
    with torch.utils.DataLoader

    Directory structure should mirror that of ImageFolder. i.e:

        all_data
        ├── test
        │   ├── actin
        │   ├── aurora
        │   ├── dna_damaging
        │   ├── kinase
        │   ├── microtubule
        │   ├── protein_deg
        │   ├── protein_synth
        │   └── statin
        └── train
            ├── actin
            ├── aurora
            ├── dna_damaging
            ├── kinase
            ├── microtubule
            ├── protein_deg
            ├── protein_synth
            └── statin

    So you would have a CellDataset for train and test separately.
    Storing these in a dicionary would be the sensible thing to do. e.g:

        path = "/path/to/all_data"
        datasets = {x: CellDataset(os.path.join(path, x) for x in ["train", "test])}
    """

    def __init__(self, data_dir, transforms=None, model="resnet",
                 normalise=False):
        self.data_dir = data_dir
        self.image_list = self.get_image_list(data_dir)
        self.label_dict = self.generate_label_dict(self.image_list)
        self.labels = self.get_unique_labels(self.image_list)
        self.transforms = transforms
        self.normalise = normalise
        assert model in ["resnet", "alexnet"]
        self.model = model

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img = np.load(self.image_list[index])
        img = self.reshape(img)
        if self.transforms is not None:
            img = self.transforms(img)
        label_name = self.get_class_label(self.image_list[index])
        label_index = torch.LongTensor([self.label_dict[label_name]])
        name = self.image_list[index].split(os.sep)[-1]
        return img, label_index, name

    @staticmethod
    def get_image_list(data_dir):
        """generate list of numpy arrays from the data directory"""
        all_images = glob.glob(os.path.join(data_dir, "*/*"))
        return [i for i in all_images if i.endswith(".npy")]

    def get_unique_labels(self, img_list):
        """return a list that just contains strings of all the class labels
           in a sorted order"""
        all_labels = [self.get_class_label(i) for i in img_list]
        return sorted(list(set(all_labels)))

    def generate_label_dict(self, image_list):
        all_labels = [self.get_class_label(i) for i in image_list]
        unique_sorted_labels = sorted(list(set(all_labels)))
        return {label: int(i) for i, label in enumerate(unique_sorted_labels)}

    def get_class_label(self, img_path):
        """get MOA label from the file path
        return this as an integer index"""
        img_path = os.path.abspath(img_path)
        return img_path.split(os.sep)[-2]

    def reshape(self, img):
        """reshape 300x300x5 numpy array into
        a 1x5x244x244 torch Tensor"""
        # reshape for a particular model as the
        # models have different expected image dimensions
        if self.model == "resnet":
            img_size = (244, 244, 5)
        elif self.model == "alexnet":
            img_size = (224, 224, 5)
        # resize image to from 300*300*5 => img_size, also converts to float
        img = ski_transform.resize(img, img_size, mode="reflect")
        if self.normalise:
            img = (img - img.mean()) / img.std()
        # reshape from width*height*channel => channel*width*height
        return Tensor(img).permute(2, 0, 1)




class CSVDataset(Dataset):
    """
    Dataset from CSV file
    """

    def __init__(self, data_dir, csv_path, model="resnet", transforms=None):
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.model = model
        self.transforms = transforms
        self.dataframe = pd.read_csv(csv_path)

    def __len__(self):
        return self.csv_path.shape[0]

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_id = row["id"]
        image = self.read_image(image_id)
        if self.transforms is not None:
            image = self.transforms(image)
        image = self.reshape(image)
        label = row["MoA"]
        return image, label

    def read_image(self, img_id):
        img_path = os.path.join(self.data_dir, "{}.npy".format(img_id))
        return np.load(img_path)

    def reshape(self, image):
        # reshape for a particular model as the
        # models have different expected image dimensions
        if self.model == "resnet":
            image_size = (244, 244, 5)
        elif self.model == "alexnet":
            image_size = (224, 224, 5)
        # resize image to from 300*300*5 => image_size, also converts to float
        image = ski_transform.resize(image, image_size, mode="reflect")
        # reshape from width*height*channel => channel*width*height
        return Tensor(image).permute(2, 0, 1)

