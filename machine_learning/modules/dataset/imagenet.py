import torch
from torchvision import datasets, transforms
from .baseset import base_set

def get_train_set(cfg):
    ds = datasets.ImageNet('~/datasets', split = 'train', download = True,
                       transform = transforms.ToTensor())
    return base_set(ds, "train", cfg)

def get_val_set(cfg):
    ds = datasets.ImageNet('~/datasets', split = 'val', download = True,
                        transform = transforms.ToTensor())
    return base_set(ds, "test", cfg)

def get_test_set(cfg):
    raise ValueError("ImageNet does not come with test set!")