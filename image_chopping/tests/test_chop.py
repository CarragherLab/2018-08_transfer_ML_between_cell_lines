import os
import numpy as np
from nncell import chop
import pytest
from skimage import io

# load test image
TEST_PATH = os.path.abspath("tests/test_images")
IMG_NUCLEI_PATH = os.path.abspath(os.path.join(TEST_PATH, "val screen_B02_s1_w1AD0ABEBC-3BA8-4199-9431-041A4D5B8C32.tif"))
IMG_MULTI_PATH = os.path.abspath(os.path.join(TEST_PATH, "multi_array.npy"))
IMG_NUCLEI = io.imread(IMG_NUCLEI_PATH)
IMG_MULTI = np.load(IMG_MULTI_PATH)


def test_is_outside_img_finds_inside0():
    # example 10 by 10 sub-array in the middle of a 100 by 100 parent array
    ans = chop.is_outside_img(50, 50, img_size=[100, 100], size=10)
    assert ans is False


def test_is_outwide_img_finds_inside1():
    ans = chop.is_outside_img(30, 40, img_size=[100, 100], size=20)
    assert ans is False


def test_is_outside_img_finds_outside0():
    ans = chop.is_outside_img(99, 50, img_size=[100, 100], size=10)
    assert ans is True


def test_is_outside_img_finds_outside1():
    ans = chop.is_outside_img(50, 99, img_size=[100, 100], size=10)
    assert ans is True


def test_is_outside_img_finds_outside2():
    ans = chop.is_outside_img(99, 99, img_size=[100, 100], size=10)
    assert ans is True


def test_is_outside_img_finds_outside3():
    ans = chop.is_outside_img(0, 4, img_size=[100, 100], size=10)
    assert ans is True


def test_is_outside_img_finds_outside4():
    ans = chop.is_outside_img(50, 0, img_size=[100, 100], size=10)
    assert ans is True


def test_nudge_coords_inside():
    # co-ordinates within image, should just return co-ordinates
    ans = chop.nudge_coords(50, 50, img_size=[100, 100], size=10)
    assert ans == [50, 50]


def test_nudge_coords_outside0():
    ans = chop.nudge_coords(0, 0, img_size=[100, 100], size=10)
    assert ans == [5, 5]


def test_nudge_coords_outside1():
    ans = chop.nudge_coords(100, 100, img_size=[100, 100], size=10)
    assert ans == [95, 95]


def test_crop_to_box():
    # create 1000 by 1000 test array
    arr = np.zeros(1000*1000).reshape([1000, 1000])
    box = chop.crop_to_box(x=50, y=50, img=arr, size=10, edge="keep")
    assert isinstance(box, np.ndarray)
    assert box.shape == (10, 10)
    assert np.zeros(100).reshape([10, 10]).all() == 0


def test_crop_to_box_on_edge():
    arr = np.zeros(1000*1000).reshape([1000, 1000])
    box = chop.crop_to_box(x=995, y=5, img=arr, size=10, edge="keep")
    assert isinstance(box, np.ndarray)
    assert box.shape == (10, 10)
    assert np.zeros(100).reshape([10, 10]).all() == 0


def test_crop_to_box_remove():
    arr = np.zeros(1000*1000).reshape([1000, 1000])
    box = chop.crop_to_box(x=4, y=5, img=arr, size=10, edge="remove")
    assert box is None


def test_crop_to_box_error_edge():
    with pytest.raises(ValueError):
        arr = np.zeros(1000*1000).reshape([1000, 1000])
        chop.crop_to_box(x=500, y=500, img=arr, size=10, edge="error")


def test_crop_to_box_error_size_too_large():
    with pytest.raises(ValueError):
        arr = np.zeros(100*100).reshape([100, 100])
        chop.crop_to_box(x=50, y=50, img=arr, size=200)


def test_crop_to_box_error_size():
    arr = np.zeros(1000*1000).reshape([1000, 1000])
    with pytest.raises(ValueError):
        chop.crop_to_box(500, 500, arr, size=10.2)
    with pytest.raises(ValueError):
        chop.crop_to_box(500, 500, arr, size=-10)
    with pytest.raises(ValueError):
        chop.crop_to_box(500, 500, arr, size=15)


def test_chop_nuclei_single_channel():
    size = 100
    arr = chop.chop_nuclei(img=IMG_NUCLEI, size=size, edge="keep", threshold=0.1)
    assert isinstance(arr, np.ndarray)
    # this is dependent on the image and parameters
    assert len(arr) == 288
    for i in arr:
        assert i.shape == (size, size)


def test_chop_nuclei_edge_remove():
    keep = chop.chop_nuclei(img=IMG_NUCLEI, size=300, edge="keep")
    remove = chop.chop_nuclei(img=IMG_NUCLEI, size=300, edge="remove")
    assert len(keep) > len(remove)


def test_chop_nuclei_multi_channel_same_number():
    multi_keep = chop.chop_nuclei(img=IMG_MULTI, size=300, edge="keep")
    single_keep = chop.chop_nuclei(img=IMG_NUCLEI, size=300, edge="keep")
    multi_remove = chop.chop_nuclei(img=IMG_MULTI, size=300, edge="remove")
    single_remove = chop.chop_nuclei(img=IMG_NUCLEI, size=300, edge="remove")
    assert len(multi_keep) == len(single_keep)
    assert len(multi_remove) == len(single_remove)


def test_chop_nuclei_multi_keeps_dims():
    multi = chop.chop_nuclei(img=IMG_MULTI, size=300, edge="keep")
    for img in multi:
        assert img.ndim == 3
        assert img.shape == (300, 300, 3)

