import os
from nncell import utils
import numpy as np

def test_is_full_path():
    # TODO
    pass


def test_make_dir():
    # TODO
    pass

def test_softmax():
    """utils.softmax()"""
    x = np.asarray([1, 2, 3, 4, 1, 2, 3])
    results = utils.softmax(x)
    results = [round(i, 3) for i in results]
    assert results == [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]
