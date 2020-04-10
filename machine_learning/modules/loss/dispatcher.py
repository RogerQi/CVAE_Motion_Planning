import torch
import torch.nn as nn
import torch.nn.functional as F

def dispatcher(cfg):
    loss_name = cfg.LOSS.loss
    assert loss_name != "none"
    if loss_name == "cross_entropy":
        from .loss import cross_entropy
        return cross_entropy(cfg)
    elif loss_name == "kl_divergence":
        return F.kl_div
    elif loss_name == "naive_vae":
        from .loss import naive_VAE
        return naive_VAE(cfg)
    elif loss_name == "nll_loss":
        assert cfg.CLASSIFIER.classifier == "log_softmax" # Some math here!
        return F.nll_loss
    elif loss_name == "mse":
        return F.mse_loss
    else:
        raise NotImplementedError