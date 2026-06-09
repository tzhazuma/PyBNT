"""Utility functions for super-resolution data preparation."""
import os


def split_train_test(imgpath, trainpath, testpath):
    """Split images into train and test directories.

    Parameters
    ----------
    imgpath : str
        Source directory of images.
    trainpath : str
        Output directory for training images.
    testpath : str
        Output directory for test images.
    """
    import DataProcess

    if not os.path.exists(trainpath):
        os.makedirs(trainpath)
    if not os.path.exists(testpath):
        os.makedirs(testpath)
    DataProcess.split_train_test(imgpath, trainpath, testpath)
