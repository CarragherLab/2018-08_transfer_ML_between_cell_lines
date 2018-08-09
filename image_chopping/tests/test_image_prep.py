"""
tests for nncell.image_prep
"""
import os
from nncell import image_prep
from parserix import parse
from parserix import clean
import numpy as np

# sort out data for tests
TEST_DIR = os.path.abspath("tests")
PATH_TO_IMG_URLS = os.path.join(TEST_DIR, "test_images/images.txt")
IMG_URLS = clean.clean([i.strip() for i in open(PATH_TO_IMG_URLS).readlines()])


#####################################
# ImageDict tests
#####################################

def test_ImageDict_remove_channels():
    channels_to_remove = [4, 5]
    ImgDict = image_prep.ImageDict()
    ans = ImgDict.remove_channels(IMG_URLS, channels_to_remove)
    # parse channel numbers out of ans
    img_names = [parse.img_filename(f) for f in ans]
    img_channels = [parse.img_channel(name) for name in img_names]
    for channel in img_channels:
        assert channel not in channels_to_remove


def test_ImageDict_keep_channels():
    channels_to_keep = [1, 2, 3]
    ImgDict = image_prep.ImageDict()
    ans = ImgDict.keep_channels(IMG_URLS, channels_to_keep)
    # parse channel numbers out of ans
    img_names = [parse.img_filename(f) for f in ans]
    img_channels = [parse.img_channel(name) for name in img_names]
    for channel in img_channels:
        assert channel in channels_to_keep


def test_ImageDict_add_class():
    ImgDict = image_prep.ImageDict()
    ImgDict.add_class("test", IMG_URLS)
    out = ImgDict.parent_dict
    assert isinstance(out, dict)
    assert list(out.keys()) == ["test"]


def test_ImageDict_group_channels():
    ImgDict = image_prep.ImageDict()
    ImgDict.add_class("test", IMG_URLS)
    ImgDict.group_image_channels()
    out = ImgDict.parent_dict
    assert isinstance(out, dict)


def test_ImageDict_make_dict():
    ImgDict = image_prep.ImageDict()
    ImgDict.add_class("test", IMG_URLS)
    ImgDict.group_image_channels()
    ImgDict.train_test_split()
    out = ImgDict.make_dict()
    assert isinstance(out, dict)


def test_ImageDict_train_test_split():
    ImgDict = image_prep.ImageDict()
    ImgDict.add_class("test", IMG_URLS)
    ImgDict.group_image_channels()
    ImgDict.train_test_split()
    out = ImgDict.make_dict()
    assert isinstance(out, dict)
    assert set(out.keys()) == set(["train", "test"])
    # train test split doesn't lose any images
    n_train = len(out["train"]["test"])
    n_test = len(out["test"]["test"])
    n_images = len(ImgDict._group_channels(IMG_URLS, order=False))
    assert n_train + n_test == n_images


def test_ImageDict_sort_channels():
    ImgDict = image_prep.ImageDict()
    # un-sorted channels
    # reverse channels as already sorted
    rev_img_urls = IMG_URLS[::-1]
    ImgDict.add_class("foo", rev_img_urls)
    ImgDict.group_image_channels(order=False)
    order_false_dict = ImgDict.parent_dict
    order_false_vals = order_false_dict["foo"][0]
    order_false_chnnls = [parse.img_channel(val) for val in order_false_vals]
    assert sorted(order_false_chnnls) != order_false_chnnls
    # sort channels
    # need to create new ImageDict class otherwise we get a warning due to
    # adding a new class to already grouped data
    ImgDict2 = image_prep.ImageDict()
    ImgDict2.add_class("bar", IMG_URLS)
    ImgDict2.group_image_channels(order=True)
    order_true_dict = ImgDict2.parent_dict
    order_true_vals = order_true_dict["bar"][0]
    order_true_chnnls = [parse.img_channel(val) for val in order_true_vals]
    assert sorted(order_true_chnnls) == order_true_chnnls


def test_ImageDict_get_wells_list():
    ImgDict = image_prep.ImageDict()
    wells_to_get = ["B02", "C02"]
    wanted = ImgDict.get_wells(IMG_URLS, wells_to_get)
    # parse well info from image urls
    wanted_wells = [parse.img_well(url) for url in wanted]
    for well in wanted_wells:
        assert well in wells_to_get


def test_ImageDict_get_wells_str():
    ImgDict = image_prep.ImageDict()
    well_to_get = "B02"
    wanted = ImgDict.get_wells(IMG_URLS, well_to_get)
    # parse well info from image urls
    wanted_wells = [parse.img_well(url) for url in wanted]
    for well in wanted_wells:
        assert well in well_to_get


####################################
# ImagePrep tests
####################################

# need to create a dictionary for ImagePrep
# this can contain only a single image in three channels
TEST_IMG_DIR = os.path.join(TEST_DIR, "test_images")
IMG_NAMES = [i for i in os.listdir(TEST_IMG_DIR) if i.startswith("val screen")]
REAL_IMG_PATHS = [os.path.join(TEST_DIR, "test_images", i) for i in IMG_NAMES]
# skip using ImageDict to create dict, make it manually
img_dict = {"foo" : REAL_IMG_PATHS}

def test_ImagePrep():
    img_prep = image_prep.ImagePrep(img_dict)
    assert isinstance(img_prep.img_dict, dict)
    assert list(img_prep.img_dict.keys()) == ["foo"]
    assert list(img_prep.img_dict.values())[0] == REAL_IMG_PATHS


def test_ImagePrep_convert_to_rgb():
    img_prep = image_prep.ImagePrep(img_dict)
    out = img_prep.convert_to_rgb(REAL_IMG_PATHS)
    assert isinstance(out, np.ndarray)
    assert out.ndim == 3

# TODO sort testing with creating and tearing down directories with pytest
# def test_ImagePrep_create_directories():
#     assert 2 + 2 == 5
#
#
# def test_ImagePrep_prepare_images():
#     assert 2 + 2 == 5


####################################
# ImagePrep tests
####################################

def test_ArrayPrep():
    array_prep = image_prep.ArrayPrep(img_dict)
    assert isinstance(array_prep.img_dict, dict)
    assert list(array_prep.img_dict.keys()) == ["foo"]
    assert list(array_prep.img_dict.values()[0]) == REAL_IMG_PATHS

def test_ArrayPrep_convert_to_stack():
    array_prep = image_prep.ArrayPrep(img_dict)
    out = array_prep.convert_to_stack(REAL_IMG_PATHS)
    assert isinstance(out, np.ndarray)
    assert out.ndim == 3
