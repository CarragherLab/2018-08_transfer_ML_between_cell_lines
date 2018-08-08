"""
Custom transforms which work on Tensors, rather than images.
"""

import torch
import random


class RandomRotate(object):
    """
    Random rotate which works on a tensor.
    Rotations are in 90 degree intervals.
    Will only work on square tensors (i.e width == height)
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image = sample
        n_rotations = random.choice([1, 2, 3, 4])
        if n_rotations == 4:
            # no point in tranposing 4 times, just return the un-rotated image
            return image
        else:
            # transpose the last two dimensions
            # as image dimensions might be:
            #    batch * channel * width * height
            # or
            #    channel * width * height
            n_dims = len(image.shape)
            if n_dims == 3:
                # no batch dimension
                width_index = 1
                height_index = 2
            elif n_dims == 4:
                # there is a batch dimension
                width_index = 2
                height_index = 3
            else:
                raise ValueError("unexpected number of dimensions")
            # FIXME: probably pretty inefficient
            # if 180 degree rotation should be able to flip image vertically
            for i in range(n_rotations):
                image = torch.transpose(image, width_index, height_index)
            return image

