import numpy as np
import torch
from .dataset_core.np_reader import numpy_reader
from .baseset import base_set

def get_train_set(cfg):
    data_npy_path = cfg.DATASET.NUMPY_READER.train_data_npy_path
    label_npy_path = cfg.DATASET.NUMPY_READER.train_label_npy_path
    if cfg.DATASET.NUMPY_READER.mmap:
        mmap_mode = "r"
    else:
        mmap_mode = None
    data_arr = np.load(data_npy_path, mmap_mode = mmap_mode, allow_pickle = False, fix_imports = False)
    label_arr = np.load(label_npy_path, mmap_mode = mmap_mode, allow_pickle = False, fix_imports = False)
    assert data_arr.shape[0] == label_arr.shape[0]
    ds = numpy_reader(data_arr, label_arr)
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    raise NotImplementedError

def get_test_set(cfg):
    data_npy_path = cfg.DATASET.NUMPY_READER.test_data_npy_path
    label_npy_path = cfg.DATASET.NUMPY_READER.test_label_npy_path
    if cfg.DATASET.NUMPY_READER.mmap:
        mmap_mode = "r"
    else:
        mmap_mode = None
    data_arr = np.load(data_npy_path, mmap_mode = mmap_mode, allow_pickle = False, fix_imports = False)
    label_arr = np.load(label_npy_path, mmap_mode = mmap_mode, allow_pickle = False, fix_imports = False)
    assert data_arr.shape[0] == label_arr.shape[0]
    ds = numpy_reader(data_arr, label_arr)
    return base_set(ds, "test", cfg)