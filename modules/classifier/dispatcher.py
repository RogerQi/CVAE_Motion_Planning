import torch
import torch.nn as nn
import torch.nn.functional as F

def dispatcher(cfg):
    classifier_name = cfg.CLASSIFIER.classifier
    assert classifier_name != "none"
    if classifier_name == "identity":
        assert cfg.task == "auto_encoder"
        def identity_func(x):
            return x
        return identity_func
    elif classifier_name == "softmax":
        def softmax_wrapper(x):
            return F.softmax(x, dim = 1)
        return softmax_wrapper
    elif classifier_name == "log_softmax":
        def log_softmax_wrapper(x):
            return F.log_softmax(x, dim = 1)
        return log_softmax_wrapper
    else:
        raise NotImplementedError