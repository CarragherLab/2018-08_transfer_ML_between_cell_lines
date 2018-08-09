import os
import numpy as np
from skimage import feature
from skimage import io
from nncell import utils

"""
chop parent image into separate images for each nuclei
"""


def is_outside_img(x, y, img_size, size):
    """
    determine if bounding box around co-ordinates is outside the image limits

    Parameters:
    ------------
    x : number
        x co-ordinate
    y : number
        y co-ordinate
    img_size : list [num, num]
        image shape
    size : number
        width and height of the bounding box (in pixels)

    Returns:
    ---------
    Boolean:
        True if box exceeds image boundary
        False if box lies within image boundary
    """
    dist = int(size / 2.0)
    if (x + dist > img_size[0]) or (x - dist < 0):
        return True
    if (y + dist > img_size[1]) or (y - dist < 0):
        return True
    return False


def nudge_coords(x, y, img_size, size):
    """
    move co-ordinates to fit box within image

    Parameters:
    ------------
    x : number
        x co-ordinate
    y : number
        y co-ordinate
    img_size : [num (x), num (y)]
        image shape
    size : number
        width and height of the bounding box (in pixels)
    """
    _check_size(size)
    dist = size / 2
    # for x co-ordinate
    if (x + dist) > img_size[0]:
        # past x max limit
        diff = abs((x + dist) - img_size[0])
        x -= diff
    if (x - dist) < 0:
        # past x min limit
        diff = abs(x - dist)
        x += diff
    # for y co-ordinate
    if (y + dist) > img_size[1]:
        # past y max limit
        diff = abs((y + dist) - img_size[1])
        y -= diff
    if (y - dist) < 0:
        # past y min limit
        diff = abs(y - dist)
        y += diff
    return [x, y]



def crop_to_box(x, y, img, size, edge="keep"):
    """
    create bounding box around co-ordinates
    Parameters:
    ------------
    x : number
        x co-ordinate
    y : number
        y co-ordinate
    img : np.array
        image
    size : number
        width and height of bounding box (in pixels)
    edge : string
        options : ("keep", "remove")
            what to do for a box which would go beyond the edge of the image.
        "keep" : will keep the box within the image boundaries at the specified
            size, though the cells may not be centered within the box.
        "remove" : do not use points which will have boxes beyond the image
            boundary.
    """
    _check_size(size)
    _check_edge_args(edge)
    for dim in img.shape[:2]:
        if dim < size:
            raise ValueError("image is too small for specified box size")
    dist = int(size / 2)
    # determine if the box will be within the image
    if is_outside_img(x, y, img.shape, size):
        if edge == "keep":
            # adjust x, y co-ordinates
            x, y = nudge_coords(x, y, img.shape, size)
        if edge == "remove":
            # don't use this x,y co-ordinate
            return None
    if img.ndim == 2:
        return img[int(x - dist): int(x + dist), int(y - dist): int(y + dist)]
    elif img.ndim == 3:
        return img[int(x - dist): int(x + dist), int(y - dist): int(y + dist), :]
    else:
        raise ValueError("wrong number of dimensions in img")


def chop_nuclei(img, size=100, edge="keep", threshold=0.1, **kwargs):
    """
    Chop an image into separate images for each nuclei. Each image will be the
    same dimensions of `size`*`size` pixels. Nuclei on the edge of the image
    can either be kept or removed, though if kept they may not be centered
    within the individual image.

    Parameters:
    ------------
    img : numpy.array
        image
    size : integer
        size of image, must be a positive even integer (in pixels)
    edge : string
        what to do with nuclei on the edge of the parent image
        options:
            keep   : nuclei will be kept though they may not be centered within
                     the individual image
            remove : nuclei near the edge of the image will be ignored
    threshold : number (default = 0.1)
        threshold argument to skimage.feature.blob_dog
    **kwargs : additional arguments to skimage.feature.blob_dog to detect the
        nuclei.
    """
    _check_edge_args(edge)
    if img.ndim == 2:
        # single channel image
        # find nuclei positions within the image
        nuclei = feature.blob_dog(img, threshold=threshold, **kwargs)
    elif img.ndim == 3:
        # multi channel image, take first channel as nuclei
        nuclei = feature.blob_dog(img[:, :, 0], threshold=threshold, **kwargs)
    else:
        raise ValueError("wrong number of dimensions in img")
    # loop through x-y co-ordinates for each nucleus
    # create list of sub-arrays
    cropped_imgs = []
    for x, y, _ in nuclei:
        cropped = crop_to_box(x, y, img, size, edge)
        cropped_imgs.append(cropped)
    # remove any empty arrays
    cropped_remove_na = [i for i in cropped_imgs if i is not None]
    return np.stack(cropped_remove_na)


def save_chopped(arr, directory, prefix="img", ext=".png", save_as="img"):
    """
    Save chopped array from chop_nuclei() to a directory. Each image will be
    saved individually and consecutively numbered.

    Parameters:
    -----------
    arr : np.ndarray
        numpy array from chop_nuclei()
    directory : string
        directory in which to save the images.
    prefix : string (default : "img")
        image prefix
    ext : string (default : ".png")
        file extension. options are .png and .jpg if saving as an image.
        Otherwise recommended extension for numpy arrays is .npy
    """
    assert isinstance(arr, np.ndarray)
    _check_ext_args(ext)
    utils.make_dir(directory)
    # loop through images in array and save with consecutive numbers
    if save_as == "img":
        for i, img in enumerate(arr, 1):
            img_name = "{}_{}{}".format(prefix, i, ext)
            full_path = os.path.join(os.path.abspath(directory), img_name)
            io.imsave(fname=full_path, arr=img)
    else:
        for i, img in enumerate(arr, 1):
            arr_name = "arr_{}.npy".format(i)
            full_path = os.path.join(os.path.abspath(directory), arr_name)
            np.save(full_path, img)


def _check_size(size):
    """checks size is an even positive integer"""
    if size % 2 != 0:
        raise ValueError("size must be an even number")
    if size < 0:
        raise ValueError("size must be positive")
    if not isinstance(size, int):
        raise ValueError("size must be an integer")


def _check_edge_args(edge):
    """check edge arguments"""
    edge_args = ["keep", "remove"]
    if edge not in edge_args:
        raise ValueError("unknown edge argument. options: {}".format(edge_args))


def _check_ext_args(ext):
    """check ext arguments"""
    ext_args = [".png", ".jpg"]
    if ext not in ext_args:
        raise ValueError("unknown ext argument. options : {}".format(ext_args))
