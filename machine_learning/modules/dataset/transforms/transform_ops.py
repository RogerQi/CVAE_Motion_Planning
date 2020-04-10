import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import torch
import torch.nn.functional as F
from torchvision import transforms
from .transforms_registry import registry

my_transforms_registry = registry()

######################
# Crop
######################
@my_transforms_registry.register
def random_resized_crop(transforms_cfg):
    size = transforms_cfg.TRANSFORMS_DETAILS.crop_size
    scale = transforms_cfg.TRANSFORMS_DETAILS.RANDOM_RESIZED_CROP.scale
    ratio = transforms_cfg.TRANSFORMS_DETAILS.RANDOM_RESIZED_CROP.ratio
    op = transforms.RandomResizedCrop(size, scale, ratio, interpolation = Image.BILINEAR)
    return op

@my_transforms_registry.register
def random_horizontal_flip(transforms_cfg):
    op = transforms.RandomHorizontalFlip(p = 0.5)
    return op

@my_transforms_registry.register
def center_crop(transforms_cfg):
    size = transforms_cfg.TRANSFORMS_DETAILS.crop_size
    op = transforms.CenterCrop(size)
    return op

@my_transforms_registry.register
def resize(transforms_cfg):
    size = transforms_cfg.TRANSFORMS_DETAILS.resize_size
    op = transforms.Resize(size, interpolation = Image.BILINEAR)
    return op

######################
# Color
######################

######################
# Geometric Transform
######################