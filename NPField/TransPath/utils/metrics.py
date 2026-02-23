from dataclasses import dataclass
from typing import Optional, List

import torch
import numpy as np

EPS = 1e-10


@dataclass
class Metrics:
    p_opt: float
    p_exp: float
    h_mean: float
    pcost_dif: float
    pcost_dif_list: list = None
    p_exp_list:list = None
    abs_cost: list = None
    abs_exp: list = None

    def __repr__(self):
        return f"optimality: {self.p_opt:0.3f}, efficiency: {self.p_exp:0.3f}, h_mean: {self.h_mean:0.3f}, cost_diff: {self.pcost_dif:0.3f}"


@dataclass
class AstarOutput:
    """
    Output structure of A* search planners
    """
    histories: torch.tensor
    paths: torch.tensor
    intermediate_results: Optional[List[dict]] = None
    g: Optional[torch.tensor] = None


def calc_metrics(na_outputs: AstarOutput, va_outputs: AstarOutput) -> Metrics:
    """
    Calculate opt, exp, and hmean metrics for problem instances each with a single starting point

    Args:
        na_outputs (AstarOutput): outputs from Neural A*
        va_outputs (AstarOutput): outputs from vanilla A*

    Returns:
        Metrics: opt, exp, and hmean values
    """
    pathlen_astar = va_outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
    pathlen_na = na_outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
    p_opt = (pathlen_astar == pathlen_na).mean()
    
    # pathcost_astar = (va_outputs.paths * va_outputs.g).sum((1, 2, 3)).detach().cpu().numpy()
    # pathcost_na =(na_outputs.paths * na_outputs.g).sum((1, 2, 3)).detach().cpu().numpy()
    pathcost_astar = torch.amax(va_outputs.paths * va_outputs.g, dim=(1, 2, 3)).detach().cpu().numpy()
    pathcost_na = torch.amax(na_outputs.paths * na_outputs.g, dim=(1, 2, 3)).detach().cpu().numpy()
    pcost_dif_list = pathcost_na / pathcost_astar
    pcost_dif = pcost_dif_list.mean()
    pcost_dif_list = pcost_dif_list.tolist()

    exp_astar = va_outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
    exp_na = na_outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
    # p_exp_list = np.maximum((exp_astar - exp_na) / exp_astar, 0.)
    p_exp_list = (exp_astar - exp_na) / exp_astar
    p_exp = p_exp_list.mean()
    h_mean = 2. / (1. / (p_opt + EPS) + 1. / (p_exp + EPS))

    return Metrics(p_opt, 1- p_exp, h_mean, pcost_dif, pcost_dif_list, (1 - p_exp_list).tolist(), 
                   pathcost_na.tolist(), exp_na.tolist())