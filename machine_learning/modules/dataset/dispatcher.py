def dispatcher(cfg):
    dataset_name = cfg.DATASET.dataset
    if dataset_name == "mnist":
        from .mnist import get_train_set, get_test_set
        return [get_train_set(cfg), get_test_set(cfg)]
    elif dataset_name == "cifar10":
        raise NotImplementedError
    elif dataset_name == "imagenet":
        raise NotImplementedError
    elif dataset_name == "numpy":
        from .generic_np_dataset import get_train_set, get_test_set
        return [get_train_set(cfg), get_test_set(cfg)]
    else:
        raise NotImplementedError