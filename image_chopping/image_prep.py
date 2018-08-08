"""
Prep Images into RGB .png images for keras ImageDataGenerator.
Need to split these by directory into the different classes.
Also need to split into a test and a training directory
"""

import os
import random
import uuid
import numpy as np
import pandas as pd
from parserix import parse
import skimage
from joblib import Parallel, delayed
import multiprocessing
from skimage import io
from nncell import utils
from nncell import chop




class Prepper(object):
    """
    Abstract class for ImagePrep and ArrayPrep to create directories of images
    or arrays for keras DataGenerators

    Input:
    -------
    input dictionary from ImageDict:

        train:
            class1 : [image_list]
            class2 : [image_list]
        test:
            class1 : [image_list]
            class2 : [image_list]
    """

    def __init__(self, img_dict):
        if isinstance(img_dict, dict):
            self.img_dict = img_dict
        else:
            raise ValueError("input needs to be a dictionary")


    @staticmethod
    def convert_to_rgb(img_channels):
        """read in three channels and merge to an 8-bit RGB array"""
        image_collection = io.imread_collection(img_channels)
        img_ubyte = skimage.img_as_ubyte(image_collection)
        return np.dstack(img_ubyte)


    def _check_dict(self):
        """check validity of input dict"""
        # make sure it has train and test sub-dictionaries
        if len(self.img_dict.keys()) != 2:
            raise ValueError("input dict has too few keys")
        # make sure it has train and test sub-dictionaries
        if ("train" or "test") not in self.img_dict.keys():
            raise ValueError("missing train or test keys in input dictionary")
        # TODO more checks, check we have lists of strings in sub-directories
        # check we have the same classes in train and test directories
        train_classes = self.img_dict["train"].keys()
        test_classes = self.img_dict["test"].keys()
        if sorted(train_classes) != sorted(test_classes):
            raise ValueError("train and test sets contain differing classes")








class ImagePrep(Prepper):
    """
    Prepare images for Keras.preprocessing.image.ImageDataGenerator.
    Creating directories split into test, training, and within those
    images are in directories named after their class.
    Images stored as three channel (RGB) .png files

    Input:
    -------
    input dictionary from ImageDict:

        train:
            class1 : [image_list]
            class2 : [image_list]
        test:
            class1 : [image_list]
            class2 : [image_list]
    """

    def __init__(self, img_dict):
        super().__init__(img_dict)


    @staticmethod
    def write_img_to_disk(img, name, path, extension=".png"):
        """
        write image to disk

        Arguments:
        ----------
        img : numpy array
            numpy array  created by ImageDict.convert_to_rgb()
        name : (string)
            what the image file should be called
        path : (string)
            path to save location
        extension : (string, default=".png")
            file extension for image, can either be saved as .png or .jpg
        """
        assert isinstance(img, np.ndarray)
        full_path = os.path.join(path, name + extension)
        io.imsave(fname=full_path, arr=img)


    def create_directories(self, base_dir):
        """
        create directory structure for prepared images

        Parameters:
        -----------
        base_dir : string
            Path to directory in which to hold training and test datasets
            A directory will be created if it does not already exist
        """
        utils.make_dir(base_dir)
        # create training and test directories
        for group in self.img_dict.keys():
            for key, img_list in self.img_dict[group].items():
                # create directory item/key from key
                dir_path = os.path.join(os.path.abspath(base_dir), group, key)
                utils.make_dir(dir_path)
                # create and save images in dir_path
                for i, img in enumerate(img_list, 1):
                    # need to load images and merge
                    rgb_img = self.convert_to_rgb(img)
                    self.write_img_to_disk(img=rgb_img, name="img_{}".format(i),
                                           path=dir_path)


    def create_directories_chop(self, base_dir, prefix="", as_array=False,
                                **kwargs):
        """
        create directory structure for prepared images, and chop each image
        into an image per cell.

        Parameters:
        -----------
        base_dir : string
            Path to directory in which to hold training and test datasets.
            A directory will be created if it does not already exist
        prefix: string
            prefix for image file names
        as_array: Boolean
            if True will save as a numpy array. If False, then images are saved
            as RGB .png files.
        **kwargs: additional arguments to chop functions
        """
        utils.make_dir(base_dir)
        for group in self.img_dict.keys():
            for key, img_list in self.img_dict[group].items():
                # create directory item/key from key
                dir_path = os.path.join(os.path.abspath(base_dir), group, key)
                utils.make_dir(dir_path)
                # create and save images in dir_path
                for i, img in enumerate(img_list, 1):
                    rgb_img = self.convert_to_rgb(img)
                    # convert_to_rgb is a bit of a misnomer, actually just stacks
                    # an image collection to a numpy array, can work with more
                    # than three channels
                    #
                    # chop image into sub-img per cell
                    # sometimes there is an error where we don't have all the
                    # channel to stack into an array, not sure what is causing
                    # this though we can skip any errors and carry on processing
                    # the images rather than crash the entire session
                    #
                    # probably a much better way to handle this, but screw it
                    try:
                        sub_img_array = chop.chop_nuclei(rgb_img, **kwargs)
                        for j, sub_img in enumerate(sub_img_array, 1):
                            if as_array: # save as numpy array
                                img_name = "{}_img_{}_{}.npy".format(prefix, i, j)
                                full_path = os.path.join(os.path.abspath(dir_path), img_name)
                                np.save(file=full_path, arr=sub_img,
                                        allow_pickle=False)
                            else: # save as .png (has to be RGB)
                                img_name = "{}_img_{}_{}.png".format(prefix, i, j)
                                full_path = os.path.join(os.path.abspath(dir_path), img_name)
                                io.imsave(fname=full_path, arr=sub_img)
                    except ValueError:
                        pass

    def create_directories_chop_par(self, base_dir, n_jobs=-1, size=200):
        """
        create directory structure for prepared images, and chop each image
        into an image per cell.

        Parameters:
        -----------
        base_dir : string
            Path to directory in which to hold training and test datasets.
            A directory will be created if it does not already exist
        **kwargs: additional arguments to chop functions
        """

        if n_jobs < 1:
            n_jobs = multiprocessing.cpu_count()

        utils.make_dir(base_dir)
        for group in self.img_dict.keys():
            for key, img_list in self.img_dict[group].items():
                # create directory item/key from key
                dir_path = os.path.join(os.path.abspath(base_dir), group, key)
                utils.make_dir(dir_path)
                Parallel(n_jobs=n_jobs)(delayed(chopper)(img, dir_path, size) for img in img_list)





class ArrayPrep(Prepper):
    """
    Prepare numpy arrays for Keras.preprocessing.image.ImageDataGenerator.
    Creating directories split into test, training, and within those
    Arrays are in directories named after their class.
    Arrays are stored as .npy files

    Input:
    -------
    input dictionary from ImageDict:
    e.g
        train:
            class1 : [image_list]
            class2 : [image_list]
        test:
            class1 : [image_list]
            class2 : [image_list]
    """

    def __init__(self, img_dict):
        super().__init__(img_dict)


    @staticmethod
    def write_array_to_disk(img, name, path, extension=".npy"):
        """
        write array to disk

        Arguments:
        ----------
        img : numpy array
            numpy array  created by ImageDict.convert_to_rgb()
        name : (string)
            what the image file should be called
        path : (string)
            path to save location
        extension : (string, default=".npy")
            file extension for array
        """
        assert isinstance(img, np.ndarray)
        full_path = os.path.join(path, name + extension)
        np.save(file=full_path, arr=img)


    def create_directories(self, base_dir):
        """
        create directory structure for prepared images

        Parameters:
        -----------
        base_dir : string
            Path to directory in which to hold training and test datasets
            A directory will be created if it does not already exist
        """
        utils.make_dir(base_dir)
        # create training and test directories
        for group in self.img_dict.keys():
            for key, img_list in self.img_dict[group].items():
                # create directory item/key from key
                dir_path = os.path.join(os.path.abspath(base_dir), group, key)
                utils.make_dir(dir_path)
                # create and save images in dir_path
                for i, img in enumerate(img_list, 1):
                    # need to load images and merge
                    rgb_img = self.convert_to_rgb(img)
                    self.write_array_to_disk(img=rgb_img, name="img_{}".format(i),
                                             path=dir_path)


    def create_directories_chop(self, base_dir, **kwargs):
        """
        create directory structure for prepared images, and chop each image
        into an image per cell.

        Parameters:
        -----------
        base_dir : string
            Path to directory in which to hold training and test datasets.
            A directory will be created if it does not already exist
        **kwargs: additional arguments to chop functions
        """
        utils.make_dir(base_dir)
        for group in self.img_dict.keys():
            for key, img_list in self.img_dict[group].items():
                # create directory item/key from key
                dir_path = os.path.join(os.path.abspath(base_dir), group, key)
                utils.make_dir(dir_path)
                # create and save images in dir_path
                for i, img in enumerate(img_list, 1):
                    rgb_img = self.convert_to_rgb(img)
                    # chop image into sub-img per cell
                    # sometimes there is an error where we don't have all the
                    # channel to stack into an array, not sure what is causing
                    # this though we can skip any errors and carry on processing
                    # the images rather than crash the entire session
                    #
                    # probably a much better way to handle this, but screw it
                    try:
                        sub_img_array = chop.chop_nuclei(rgb_img, **kwargs)
                        for j, sub_img in enumerate(sub_img_array, 1):
                            img_name = "img_{}_{}.npy".format(i, j)
                            full_path = os.path.join(os.path.abspath(dir_path), img_name)
                            np.save(file=full_path, arr=sub_img)
                    except ValueError:
                        pass









class ImageDict(object):
    """
    Class to make an image dictionary for ImagePrep()

    No idea on how to handle this yet
        - Interactively add image lists that are appended to a dictionary?
        - Construct a directory system and give the directory path to ImageDict?
        - Give plate, wells etc for various classes?
    """

    def __init__(self, train_test_sets=False):
        self.train_test_sets = train_test_sets
        self.grouped = False
        self.parent_dict = dict()
        self.train_test_dict = dict()


    @staticmethod
    def _group_channels(url_list, order):
        """
        given a list of img urls, this will group them into the same well and
        site, per plate
        Arguments:
        -----------
        order : boolean
            sort channel numbers into numerical order
        """
        grouped_list = []
        urls = [parse.img_filename(i) for i in url_list]
        tmp_df = pd.DataFrame(list(url_list), columns=["img_url"])
        tmp_df["plate_name"] = [parse.plate_name(i) for i in url_list]
        tmp_df["plate_num"] = [parse.plate_num(i) for i in url_list]
        # get_well and get_site use the image URL rather than the full path
        tmp_df["well"] = [parse.img_well(i) for i in urls]
        tmp_df["site"] = [parse.img_site(i) for i in urls]
        grouped_df = tmp_df.groupby(["plate_name", "plate_num", "well", "site"])
        if order is True:
            # order by channel
            for _, group in grouped_df:
                grouped = list(group["img_url"])
                channel_nums = [parse.img_channel(i) for i in grouped]
                # create tuple(path, channel_number) and sort by channel_number
                sort_im = sorted(zip(grouped, channel_nums), key=lambda x: x[1])
                # return only the file-paths back from the list of tuples
                grouped_list.append([i[0] for i in sort_im])
        elif order is False:
            for _, group in grouped_df:
                grouped_list.append(list(group["img_url"]))
        else:
            raise ValueError("order needs to be a boolean")
        return grouped_list


    @staticmethod
    def _split_train_test(list_like, test_size):
        """
        randomly split list object into training and test set

        Parameters:
        -----------
        list_like : list-like
            list to split into training and test sets
        test_size : float
            proportion of the data to become the test set
        """
        test_n = int(round(test_size * len(list_like)))
        train_n = int(len(list_like) - test_n)
        random.shuffle(list_like)
        training = list_like[:train_n]
        test = list_like[-test_n:]
        assert len(list_like) == len(training) + len(test)
        return [training, test]


    @staticmethod
    def get_wells(img_list, wells_to_get, plate=None):
        """
        given a list of image paths, this will return the images matching
        the well or wells given in well

        Parameters:
        -----------
        img_list : list
            list of image URLs
        well : string or list of strings
            which well(s) to select
        plate: string or list of strings (default = None)
            get wells per specified plate(s)
        """
        # parse wells from metadata
        if plate is None:
            # ignore plate labels, get all matching wells
            wells = [parse.img_well(path) for path in img_list]
            combined = zip(img_list, wells)
            if isinstance(wells_to_get, list):
                wanted_images = []
                for i in wells_to_get:
                    for path, parsed_well in combined:
                        if i == parsed_well:
                            wanted_images.append(path)
            elif isinstance(wells_to_get, str):
                wanted_images = []
                for path, parsed_well in combined:
                    if wells_to_get == parsed_well:
                        wanted_images.append(path)
            return wanted_images
        else:
            # get wells per specified plate(s)
            wanted_images = []
            if isinstance(wells_to_get, str):
                wells_to_get = [wells_to_get]
            if isinstance(plate, str):
                plate = [plate]
            urls = [parse.img_filename(i) for i in img_list]
            tmp_df = pd.DataFrame(list(img_list), columns=["img_url"])
            tmp_df["plate_name"] = [parse.plate_name(i) for i in img_list]
            tmp_df["well"] = [parse.img_well(i) for i in urls]
            tmp_df["site"] = [parse.img_site(i) for i in urls]
            grouping_cols = ["plate_name"]
            grouped_df = tmp_df.groupby(grouping_cols)
            for name, group in grouped_df:
                if name in plate:
                    # get only wells that match well
                    tmp_urls = group[group.well.isin(wells_to_get)].img_url.tolist()
                    wanted_images.extend(tmp_urls)
            return wanted_images
                    



    @staticmethod
    def remove_channels(img_list, channels):
        """
        given a list of image paths, this will remove specified channel
        numbers.

        Parameters:
        -----------
        img_list : list
            list of image URLs
        channels : list of integers
            list of channel numbers to exclude

        Returns:
        --------
        list of image URLs
        """
        # find if img_urls are full paths or just filenames
        if utils.is_full_path(img_list[0]):
            just_file_path = [parse.img_filename(i) for i in img_list]
        else:
            just_file_path = img_list
        channel_nums = [parse.img_channel(i) for i in just_file_path]
        # make sure we zip the original img_list, *not* just_file_path
        ch_img_tup = zip(channel_nums, img_list)
        filtered_tup = [i for i in ch_img_tup if i[0] not in channels]
        _, img_urls = zip(*filtered_tup)
        return img_urls


    @staticmethod
    def keep_channels(img_list, channels):
        """
        given a list of image paths, this will keep specified channel numbers,
        and remove all others.

        Parameters:
        -----------
        img_list : list
            list of image URLs
        channels : list of integers
            list of channel numbers to keep

        Returns:
        --------
        list of image URLs
        """
        # find if img_urls are full paths or just filenames
        if utils.is_full_path(img_list[0]):
            just_file_path = [parse.img_filename(i) for i in img_list]
        else:
            just_file_path = img_list
        channel_nums = [parse.img_channel(i) for i in just_file_path]
        # make sure we zip the original img_list, *not* just_file_path
        ch_img_tup = zip(channel_nums, img_list)
        filtered_tup = [i for i in ch_img_tup if i[0] in channels]
        _, img_urls = zip(*filtered_tup)
        return img_urls


    def group_image_channels(self, order=True):
        """group each image list into RGB channels"""
        if self.train_test_sets is True:
            raise AttributeError("already formed training and test sets")
        for key, img_list in self.parent_dict.items():
            self.parent_dict[key] = self._group_channels(img_list, order)
        self.grouped = True


    def train_test_split(self, test_size=0.3):
        """
        split into train and test sets
        these are stored in separate dictionary keys
        """
        if self.grouped is False:
            raise AttributeError("image channels not grouped")
        # create train and test sub-dictionaries
        self.train_test_dict["test"] = dict()
        self.train_test_dict["train"] = dict()
        # loop through class lists
        # split into training and test, place in approp dicts under the same key
        for key, img_list in self.parent_dict.items():
            train, test = self._split_train_test(img_list, test_size)
            self.train_test_dict["train"][key] = train
            self.train_test_dict["test"][key] = test
        # once finished, indicate we have created training and test sets
        self.train_test_sets = True


    def add_class(self, class_name, url_list, extend=True):
        """
        add a new class of images to the url_list

        Arguments:
        ----------
        class_name : string
            name of the class to be stored in the dictionary
        url_list : list of strings
            image paths that belong to `class_name`
        extend : Boolean
            If True will extend the existing class, if False it will
            overwrite the data in that class
        """
        if self.grouped is True:
            raise Warning("channels already grouped, this will need to be " +
                          "called again to group the new class")
        if self.train_test_sets is True:
            msg = "cannot add new class once training and test sets have been sampled"
            raise AttributeError(msg)
        # if we already have that class
        if class_name in self.parent_dict.keys():
            if extend is True:
                self.parent_dict[class_name].extend(url_list)
            elif extend is False:
                msg = "'{}' already exists and extend is False, overwriting class".format(
                    class_name)
                print(msg)
                self.parent_dict[class_name] = url_list
            else:
                raise ValueError("append expects a Boolean")
        else:
            self.parent_dict[class_name] = url_list


    def make_dict(self):
        """return image dictionary"""
        return self.train_test_dict




def _convert_to_rgb(img_channels):
    """read in three channels and merge to an 8-bit RGB array"""
    image_collection = io.imread_collection(img_channels)
    img_ubyte = skimage.img_as_ubyte(image_collection)
    return np.dstack(img_ubyte)


def chopper(img, dir_path, size):
    """wrapper round chop.chop_nuclei for joblib parallelism"""
    try:
        img = _convert_to_rgb(img)
        sub_img_array = chop.chop_nuclei(img, size)
        for sub_img in sub_img_array:
            img_name = "img_{}.png".format(uuid.uuid4().hex)
            full_path = os.path.join(os.path.abspath(dir_path), img_name)
            io.imsave(fname=full_path, arr=sub_img)
    except ValueError:
        # numpy stack error for empty channels, skip image
        pass

