from models.autoencoder import Autoencoder
from data.hmaps import GridData
from modules.planners import DifferentiableDiagAstar, get_diag_heuristic

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

import argparse


def main(mode, state_dict_path, hardness_limit=1.05):
    device = 'cuda'
    
    test_data = GridData(
        path='./TransPath_data/test',
        mode=mode
    )
    test_dataloader = DataLoader(test_data, batch_size=256,
                        shuffle=False, num_workers=0, pin_memory=True)
    model = Autoencoder(mode=mode)
    model.load_state_dict(torch.load(state_dict_path))
    model.to(device)
    model.eval()
    
    vanilla_planner = DifferentiableDiagAstar(mode='default', h_w=1)
    if mode == 'cf':
        learnable_planner = DifferentiableDiagAstar(mode='k')
    else:
        learnable_planner = DifferentiableDiagAstar(mode=mode, f_w=100)
    vanilla_planner.to(device)
    learnable_planner.to(device)    
        
    expansions_ratio = []
    cost_ratio = []
    hardness = []
    
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            map_design, start, goal, gt_heatmap = batch
            inputs = torch.cat([map_design, start + goal], dim=1) if mode == 'f' else torch.cat([map_design, goal], dim=1)
            inputs = inputs.to(device)

            predictions = (model(inputs) + 1) / 2
            learn_outputs = learnable_planner(
                predictions.to(device),
                start.to(device),
                goal.to(device),
                ((map_design == 0)*1.).to(device)
            )
            vanilla_outputs = vanilla_planner(
                ((map_design == 0)*1.).to(device),
                start.to(device),
                goal.to(device),
                ((map_design == 0)*1.).to(device)
            )
            expansions_ratio.append(((learn_outputs.histories).sum((-1, -2, -3))) / ((vanilla_outputs.histories).sum((-1, -2, -3))))
            learn_costs = (learn_outputs.g * goal.to(device)).sum((-1, -2, -3))
            vanilla_costs = (vanilla_outputs.g * goal.to(device)).sum((-1, -2, -3))
            cost_ratio.append(learn_costs / vanilla_costs)
            start_heur = (get_diag_heuristic(goal[:, 0].to(device)) * start[:, 0].to(device)).sum((-1, -2))
            hardness.append(vanilla_costs / start_heur)

    expansions_ratio = torch.cat(expansions_ratio, dim=0)
    cost_ratio = torch.cat(cost_ratio, dim=0)
    hardness = torch.cat(hardness, dim=0)
    mask = torch.where(hardness >= hardness_limit, torch.ones_like(hardness), torch.zeros_like(hardness))
    n = mask.sum()
    expansions_ratio = (expansions_ratio * mask).sum() / n
    cost_ratio = (cost_ratio * mask).sum() / n
    
    print(f'expansions_ratio: {expansions_ratio}, cost_ratio: {cost_ratio}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['f', 'cf'], default='f')
    parser.add_argument('--seed', type=int, default=39)
    parser.add_argument('--weights_path', type=str, default='./weights/focal.pth')
    
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    
    main(
        mode=args.mode,
        state_dict_path=args.weights_path,
    )
