import numpy as np
import torch
from torchvision import transforms
from .transforms.dispatcher import dispatcher

class base_set(torch.utils.data.Dataset):
    '''
    An implementation of torch.utils.data.Dataset that supports various
    data transforms and augmentation.
    '''
    def __init__(self, dataset, split, cfg):
        '''
        Args:
            dataset: any object with __getitem__ and __len__ methods implemented.
                Object retruned from dataset[i] is expected to be (raw_tensor, label).
            split: ("train" or "test"). Specify dataset mode
            cfg: yacs root config node object.
        '''
        assert split in ["train", "test"]
        self.dataset = dataset
        if split == "train":
            transforms_config_node = cfg.DATASET.TRANSFORM.TRAIN
        else:
            transforms_config_node = cfg.DATASET.TRANSFORM.TEST
        self.transforms = self._get_all_transforms(transforms_config_node)
    
    def __getitem__(self, index):
        desired_data = self.dataset[index] # (data, label)
        return (self.apply_transforms(desired_data[0], self.transforms), desired_data[1])

    def __len__(self):
        return len(self.dataset)

    def apply_transforms(self, img, transforms):
        '''
        Args:
            img: data to be transformed
            transforms: list of transforms constructed by torchvision.transforms.Compose
        '''
        return transforms(img)

    def _get_all_transforms(self, transforms_cfg):
        transforms_list = transforms_cfg.transforms
        assert len(transforms_list) != 0
        if transforms_list == ('none',):
            return transforms.Compose([])
        if transforms_list == ('normalize'):
            return transforms.Compose([self._get_dataset_normalizer(transforms_cfg)])
        # Nontrivial transforms...
        try:
            normalize_first_occurence = transforms_list.index("normalize")
            assert normalize_first_occurence == len(transforms_list) - 1
            return transforms.Compose([transforms.ToPILImage()] + dispatcher(transforms_cfg) + [transforms.ToTensor(),
                        self._get_dataset_normalizer(transforms_cfg)])
        except ValueError:
            # Given transforms does not contain normalization
            return transforms.Compose([transforms.ToPILImage()] + dispatcher(transforms_cfg) + [transforms.ToTensor()])
    
    def _get_dataset_normalizer(self, transforms_cfg):
        return transforms.Normalize(transforms_cfg.TRANSFORMS_DETAILS.NORMALIZE.mean,
                                    transforms_cfg.TRANSFORMS_DETAILS.NORMALIZE.sd)