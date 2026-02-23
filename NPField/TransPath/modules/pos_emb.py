import torch
from torch import nn
import numpy as np


def build_grid(resolution, max_v=1.):
    """
    :param resolution: tuple of 2 numbers
    :return: grid for positional embeddings built on input resolution
    """
    ranges = [np.linspace(0., max_v, num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return np.concatenate([grid, max_v - grid], axis=-1)


class PosEmbeds(nn.Module):
    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.linear = nn.Linear(4, hidden_size)
        self.grid = nn.Parameter(torch.Tensor(build_grid(resolution)), requires_grad=False)
        
    def forward(self, inputs):
        pos_emb = self.linear(self.grid).moveaxis(3, 1)
        return inputs + pos_emb
    
    def change_resolution(self, resolution, max_v):
        self.grid = nn.Parameter(torch.Tensor(build_grid(resolution, max_v)), requires_grad=False)
