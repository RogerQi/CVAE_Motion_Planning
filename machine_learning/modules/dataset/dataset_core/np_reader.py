import numpy as np
import torch

class numpy_reader(object):
    '''
    Numpy reader that reads npy file.

    Reading Native NPY saved by np.save is much faster than pickle. ~ hdf5.
    '''
    def __init__(self, data_arr, label_arr):
        assert data_arr.shape[0] == label_arr.shape[0]
        self.data_arr = torch.from_numpy(data_arr).float()
        self.label_arr = torch.from_numpy(label_arr).float()

    def __getitem__(self, idx):
        return (self.data_arr[idx], self.label_arr[idx])

    def __len__(self):
        return self.data_arr.shape[0]
