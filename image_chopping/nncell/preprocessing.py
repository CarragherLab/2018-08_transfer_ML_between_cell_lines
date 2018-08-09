import os
import random
from functools import partial
import multiprocessing.pool
import threading
import numpy as np


class Iterator(object):
    """
    Abstract base class for data iterators

    Parameters:
    -----------
    n : Integer
        total number of samples in the dataset to loop over
    batch_size: Integer
        size of batch
    """

    def __init__(self, n, batch_size):
        self.n = n
        self.batch_size = batch_size
        self.batch_index = 0
        self.total_batches_seen = 0
        self.index_generator = self._flow_index(n, batch_size)
        self.lock = threading.Lock()


    def reset(self):
        self.batch_index = 0


    def _flow_index(self, n, batch_size=32):
        self.reset()
        while 1:
            if self.batch_index == 0:
                index_array = np.arange(n)
            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)


    def __iter__(self):
        return self


    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)




class DirectoryIterator(Iterator):
    """
    docstring

    Parameters:
    ------------
    directory: string
        Path to directory to read from. Each subdirectory in this directory
        will be considered to contain samples from one class.
    image_data_generator: Instance of ArrayDataGenerator
        To use for transformations and normalizations
    batch_size: Integer
                Size of batch
    """

    def __init__(self, directory, image_data_generator, batch_size, image_shape,
                 class_mode="categorical", follow_links=False):
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.class_mode = class_mode

        # count the number of samples and classes
        white_list_formats = ["npy"]

        self.samples = 0

        classes = []
        for subdir in sorted(os.listdir(directory)):
            if os.path.isdir(os.path.join(directory, subdir)):
                classes.append(subdir)
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))


        def _recursive_list(subpath):
            return sorted(os.walk(subpath, followlinks=follow_links),
                          key=lambda tpl: tpl[0])


        pool = multiprocessing.pool.ThreadPool()
        function_partial = partial(_count_valid_files_in_directory,
                                white_list_formats=["npy"],
                                follow_links=follow_links)
        self.samples = sum(pool.map(function_partial,
                                    (os.path.join(directory, subdir)
                                     for subdir in classes)))
        # print message like keras
        print("Found {} arrays belonging to {} classes".format(
              self.samples, self.num_classes))
        results = []
        self.filenames = []
        self.classes = np.zeros((self.samples, ), dtype="int32")
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(pool.apply_async(_list_valid_filenames_in_directory,
                                            (dirpath, white_list_formats,
                                             self.class_indices, follow_links)))
        for res in results:
            classes, filenames = res.get()
            self.classes[i : i + len(classes)] = classes
            self.filenames += filenames
            i += len(classes)
        pool.close()
        pool.join()
        super(DirectoryIterator, self).__init__(self.samples, batch_size)


    def next(self):
        """
        returns the next batch
        """
        with self.lock:
            index_array, _, current_batch_size = next(self.index_generator)
        batch_x = np.zeros((current_batch_size, ) + self.image_shape, dtype="float32")
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            arr = np.load(os.path.join(self.directory, fname))
            # transform here
            batch_x[i] = arr
        if self.class_mode == "categorical":
            batch_y = np.zeros((len(batch_x), self.num_classes), dtype="float32")
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1
        return batch_x, batch_y





class ArrayDataGenerator(Iterator):
    """
    Similar to keras.preprocessing.ImageDataGenerator but works on numpy arrays.
    """

    def __init__(self, rescale=None, horizontal_flip=False):
        self.rescale = rescale
        self.horizontal_flip = horizontal_flip


    def flow(self):
        """
        docstring
        """
        raise NotImplementedError("Not made this yet!")


    def flow_from_directory(self, directory, batch_size=32,
                            follow_links=False, image_shape=(250, 250)):
        """
        generator to return numpy arrays from a directory
        """
        return DirectoryIterator(directory, self, batch_size=batch_size,
                                 follow_links=follow_links)

        if self.horizontal_flip is True:
            # Check if array is square. If it is we can rotate by a
            # multiple of 90 degrees
            x_dim, y_dim = arr.shape[:2]
            if x_dim == y_dim:
                # number of times for 90 degree rotation
                n_rotations = random.sample([0, 1, 2, 3], 1)
                arr = np.rot90(arr, n_rotations)

                # if array is square, then random rotation between 0, 90, 180 or
                # 270 degrees
            else:
                # if not square, just flip horizontally
                arr = np.fliplr(arr)
        else:
            # yield numpy array
            pass



def _count_valid_files_in_directory(directory, white_list_formats, follow_links):
    """
    Count files with extension in white_list_formats contained in a directory

    Parameters:
    -----------
    directory: string
        absolute path to the directory containing files to be counted
    white_list_formats: list of strings
        allowed extensions for files to be counted
    
    Returns:
    --------
    count of files with extension in white_list_formats contained in directory
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links),
                      key=lambda tpl: tpl[0])

    samples = 0
    for root, _, files in _recursive_list(directory):
        for fname in files:
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith("." + extension):
                    is_valid = True
                    break
            if is_valid:
                samples += 1
    return samples


def _list_valid_filenames_in_directory(directory, white_list_formats,
                                       class_indices, follow_links):
    """
    List files with extension in white_list_formats contained in a directory

    Parameters:
    -----------
    directory: string
        absolute path to the directory containing giles to be counted
    white_list_formats: list of strings
        allowes extensions for files to be counted

    Returns:
    --------
    list of files with extension in white_list_formats contained in directory
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links),
                      key=lambda tpl: tpl[0])
    classes = []
    filenames = []
    subdir = os.path.basename(directory)
    basedir = os.path.dirname(directory)
    for root, _, files in _recursive_list(directory):
        for fname in files:
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith("." + extension):
                    is_valid = True
                    break
            if is_valid:
                classes.append(class_indices[subdir])
                # add filename relative to directory
                absolute_path = os.path.join(root, fname)
                filenames.append(os.path.relpath(absolute_path, basedir))
    return classes, filenames

