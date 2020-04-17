import torch
import torch.nn as nn
import torch.nn.functional as F

class cross_entropy(nn.Module):
    '''
    (Optionally) Weighted cross entropy loss
    '''
    def __init__(self, cfg, loss_weight = None):
        super().__init__()
        self.weight = loss_weight
    
    def forward(self, output, label):
        return F.cross_entropy(output, label, weight = self.weight)

class naive_VAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.kl_div_factor = cfg.LOSS.loss_factor
    
    def forward(self, output, original_input, aux_dict):
        # Sanity check
        mu_vec = aux_dict['mean_vec']
        log_var_vec = aux_dict['log_var_vec']
        assert output.shape == original_input.shape
        assert mu_vec.shape == log_var_vec.shape
        # Calculate losses
        bce_loss = F.binary_cross_entropy(output, original_input, reduction = "sum")
        # KL Divergence between model estimated distribution and N(0, 1)
        kl_div_loss = 0.5 * torch.sum((torch.exp(log_var_vec)) + (mu_vec.pow(2)) - 1 - log_var_vec)
        return bce_loss + self.kl_div_factor * kl_div_loss