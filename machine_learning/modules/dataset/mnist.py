import torch
from torchvision import datasets, transforms
from .baseset import base_set

def get_train_set(cfg):
    ds = datasets.MNIST('../datasets', train = True, download = True,
                       transform = transforms.ToTensor())
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    raise ValueError("MNIST does not come with validation set!")

def get_test_set(cfg):
    ds =  datasets.MNIST('../datasets', train = False, download = True,
                        transform = transforms.ToTensor())
    return base_set(ds, "test", cfg)