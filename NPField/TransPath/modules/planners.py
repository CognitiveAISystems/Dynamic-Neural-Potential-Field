"""
modified code from
https://github.com/omron-sinicx/neural-astar/blob/minimal/neural_astar/planner/differentiable_astar.py 
"""

import math
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AstarOutput:
    """
    Output structure of A* search planners
    """
    histories: torch.tensor
    paths: torch.tensor
    intermediate_results: Optional[List[dict]] = None
    g: Optional[torch.tensor] = None


def get_diag_heuristic(goal_maps):
    num_samples, size = goal_maps.shape[0], goal_maps.shape[-1]
    grid = torch.meshgrid(torch.arange(0, size), torch.arange(0, size))
    loc = torch.stack(grid, dim=0).type_as(goal_maps)
    loc_expand = loc.reshape(2, -1).unsqueeze(0).expand(num_samples, 2, -1)
    goal_loc = torch.einsum("kij, bij -> bk", loc, goal_maps)
    goal_loc_expand = goal_loc.unsqueeze(-1).expand(num_samples, 2, -1)
    
    #diagonal distance
    dxdy = torch.abs(loc_expand - goal_loc_expand)
    h = dxdy.min(dim=1)[0] * (2**0.5) + torch.abs(dxdy[:, 0] - dxdy[:, 1])
    h = h.reshape_as(goal_maps)

    return h


def _st_softmax_noexp(val: torch.tensor) -> torch.tensor:
    """
    Softmax + discretized activation
    Used a detach() trick as done in straight-through softmax

    Args:
        val (torch.tensor): exponential of inputs.

    Returns:
        torch.tensor: one-hot matrices for input argmax.
    """

    val_ = val.reshape(val.shape[0], -1)
    y = val_ / (val_.sum(dim=-1, keepdim=True))
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y)
    y_hard[range(len(y_hard)), ind] = 1
    y_hard = y_hard.reshape_as(val)
    y = y.reshape_as(val)
    return (y_hard - y).detach() + y


def expand(x: torch.tensor, neighbor_filter: torch.tensor) -> torch.tensor:
    """
    Expand neighboring node 

    Args:
        x (torch.tensor): selected nodes
        neighbor_filter (torch.tensor): 3x3 filter to indicate 8 neighbors

    Returns:
        torch.tensor: neighboring nodes of x
    """

    x = x.unsqueeze(0)
    num_samples = x.shape[1]
    y = F.conv2d(x, neighbor_filter, padding=1, groups=num_samples).squeeze()
    y = y.squeeze(0)
    return y


def backtrack(start_maps: torch.tensor, goal_maps: torch.tensor,
              parents: torch.tensor, current_t: int) -> torch.tensor:
    """
    Backtrack the search results to obtain paths

    Args:
        start_maps (torch.tensor): one-hot matrices for start locations
        goal_maps (torch.tensor): one-hot matrices for goal locations
        parents (torch.tensor): parent nodes
        current_t (int): current time step

    Returns:
        torch.tensor: solution paths
    """

    num_samples = start_maps.shape[0]
    parents = parents.type(torch.long)
    goal_maps = goal_maps.type(torch.long)
    start_maps = start_maps.type(torch.long)
    path_maps = goal_maps.type(torch.long)
    num_samples = len(parents)
    loc = (parents * goal_maps.view(num_samples, -1)).sum(-1)
    for _ in range(current_t):
        path_maps.view(num_samples, -1)[range(num_samples), loc] = 1
        loc = parents[range(num_samples), loc]
    return path_maps


class DifferentiableDiagAstar(nn.Module):
    def __init__(self, g_ratio: float = 0.5, Tmax: float = 0.95, h_w=1, f_w=2, mode='default'):
        """
        Differentiable A* module

        Args:
            g_ratio (float, optional): ratio between g(v) + h(v).
            Tmax (float, optional): how much of the map the planner is able to potentially explore during training. 
        """

        super().__init__()

        neighbor_filter = torch.ones(1, 1, 3, 3)
        neighbor_filter[0, 0, 1, 1] = 0
        self.neighbor_filter = nn.Parameter(neighbor_filter,
                                            requires_grad=False)
        cost_filter = torch.ones(1, 1, 3, 3)
        cost_filter[0, 0, 1, 1] = 0
        cost_filter[0, 0, 0, 0] = 2**0.5
        cost_filter[0, 0, 0, 2] = 2**0.5
        cost_filter[0, 0, 2, 0] = 2**0.5
        cost_filter[0, 0, 2, 2] = 2**0.5
        self.cost_filter = nn.Parameter(cost_filter, requires_grad=False)
        
        self.get_heuristic = get_diag_heuristic

        self.g_ratio = g_ratio
        assert (Tmax > 0) & (Tmax <= 1), "Tmax must be within (0, 1]"
        self.Tmax = Tmax
        self.mode = mode
        self.h_w = h_w
        self.f_w = f_w

    def forward(self,
                cost_maps: torch.tensor,
                start_maps: torch.tensor,
                goal_maps: torch.tensor,
                obstacles_maps: torch.tensor) -> AstarOutput:
        assert cost_maps.ndim == 4
        assert start_maps.ndim == 4
        assert goal_maps.ndim == 4
        assert obstacles_maps.ndim == 4
        
        cost_maps = cost_maps[:, 0]
        start_maps = start_maps[:, 0]
        goal_maps = goal_maps[:, 0]
        obstacles_maps = obstacles_maps[:, 0]
        if self.mode == 'h':
            heuristic = cost_maps
        elif self.mode == 'f':
            focal_map = cost_maps
        elif self.mode == 'k':
            heuristic_koef = cost_maps
        cost_maps = torch.zeros_like(obstacles_maps)

        num_samples = start_maps.shape[0]
        neighbor_filter = self.neighbor_filter
        neighbor_filter = torch.repeat_interleave(neighbor_filter, num_samples,
                                                  0)
        cost_filter = self.cost_filter
        cost_filter = torch.repeat_interleave(cost_filter, num_samples,
                                                  0)
        size = start_maps.shape[-1]

        open_maps = start_maps
        histories = torch.zeros_like(start_maps)
        intermediate_results = []

        h = self.get_heuristic(goal_maps) * self.h_w
        if self.mode == 'h':
            h = heuristic
        elif self.mode == 'k':
            thr = torch.tensor(0.1).to(next(self.parameters()).device)
            heuristic_koef = torch.where(heuristic_koef > thr, heuristic_koef, thr)
            h = h / (heuristic_koef)
        g = torch.zeros_like(start_maps)

        parents = (
            torch.ones_like(start_maps).reshape(num_samples, -1) *
            goal_maps.reshape(num_samples, -1).max(-1, keepdim=True)[-1])

        size = cost_maps.shape[-1]
        Tmax = self.Tmax if self.training else 1.
        Tmax = int(Tmax * size * size)
        for t in range(Tmax):
            if self.mode != 'f':
                f = self.g_ratio * g + (1 - self.g_ratio + 0.001) * h
                f_exp = torch.exp(-1 * f / math.sqrt(size))
                f_exp = f_exp * open_maps
                selected_node_maps = _st_softmax_noexp(f_exp)
            else:
                f = self.g_ratio * g + (1 - self.g_ratio) * h
                f_open = (f * open_maps + (open_maps == 0) * size**2).view(num_samples, -1)
                min_values, _  = f_open.min(dim=-1)
                new_open = torch.where((f <= min_values.view(-1, 1, 1) * self.f_w) * (open_maps == 1) , 1., 0.)
                focal_exp = torch.exp(focal_map)
                focal_exp = focal_exp * new_open
                selected_node_maps = _st_softmax_noexp(focal_exp)

            dist_to_goal = (selected_node_maps * goal_maps).sum((1, 2),
                                                                keepdim=True)
            is_unsolved = (dist_to_goal < 1e-8).float()

            histories = histories + selected_node_maps
            histories = torch.clamp(histories, 0, 1)
            open_maps = open_maps - is_unsolved * selected_node_maps
            open_maps = torch.clamp(open_maps, 0, 1)

            neighbor_nodes = expand(selected_node_maps, neighbor_filter)
            neighbor_nodes = neighbor_nodes * obstacles_maps

            g2 = (g*selected_node_maps).sum((1, 2), keepdim=True) + expand(selected_node_maps, cost_filter)
            idx = (1 - open_maps) * (1 - histories) + open_maps * (g > g2)
            idx = idx * neighbor_nodes
            idx = idx.detach()
            g = g2 * idx + g * (1 - idx)
            g = g.detach()

            # update open maps
            open_maps = torch.clamp(open_maps + idx, 0, 1)
            open_maps = open_maps.detach()

            # for backtracking
            idx = idx.reshape(num_samples, -1)
            snm = selected_node_maps.reshape(num_samples, -1)
            new_parents = snm.max(-1, keepdim=True)[1]
            parents = new_parents * idx + parents * (1 - idx)

            if torch.all(is_unsolved.flatten() == 0):
                break

        # backtracking
        path_maps = backtrack(start_maps, goal_maps, parents, t)

        return AstarOutput(histories.unsqueeze(1), path_maps.unsqueeze(1),
                           intermediate_results, g.unsqueeze(1))
