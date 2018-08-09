import os
import numpy as np

def is_full_path(path):
    """determines if URL is a full path or just a file name"""
    num_splits = len(path.split(os.sep))
    if num_splits > 1:
        return True
    elif num_splits == 1:
        return False


def make_dir(directory):
    """sensible way to create directory"""
    try:
        os.makedirs(directory)
    except OSError:
        if os.path.isdir(directory):
            pass
        else:
            err_msg = "failed to create directory {}".format(directory)
            raise RuntimeError(err_msg)


def softmax(results):
    results = np.asarray(results)
    exp_r = np.exp(results)
    return exp_r / exp_r.sum()

