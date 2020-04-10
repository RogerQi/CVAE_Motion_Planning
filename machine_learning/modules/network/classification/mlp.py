import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class net(nn.Module):
    def __init__(self, cfg):
        super(net, self).__init__()
        input_size = np.prod(cfg.input_dim)
        output_size = cfg.num_classes
        latent_space_dim = [128, 32]
        latent_space_dim = [input_size] + latent_space_dim
        latent_space_dim = latent_space_dim + [output_size]
        net_list = []
        for i in range(len(latent_space_dim) - 1):
            net_list.append(nn.Linear(latent_space_dim[i], latent_space_dim[i + 1]))
            if i != len(latent_space_dim) - 2:
                net_list.append(nn.ReLU())
        print(net_list)
        self.net = nn.Sequential(*net_list)

    def forward(self, x):
        x = torch.flatten(x, start_dim = 1)
        return self.net(x)