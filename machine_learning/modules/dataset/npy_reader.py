import numpy as np
import torch

from .baseset import base_set

class numpy_reader(object):
    '''
    Numpy reader that reads npy file.

    Reading Native NPY saved by np.save is much faster than pickle. ~ hdf5.
    '''
    def __init__(self):
        pass
