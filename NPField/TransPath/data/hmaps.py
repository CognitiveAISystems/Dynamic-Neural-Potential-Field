import os
import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset


def proc_grid(grid):
    rows = []
    for row in grid:
        rows.append([float(i) for i in row.text.split()])
    return np.array(rows)


class PathData(Dataset):
    def __init__(self, xml_path, koef_path, h_path, grid_size=64, limit_k=1, clip_value=0.):
        self.xml_path = xml_path
        self.koef_path = koef_path
        self.h_path = h_path
        self.file_names = os.listdir(self.xml_path)
        self.size = len(self.file_names) // limit_k
        self.grid_size = grid_size
        self.clip_value = clip_value

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        tree = ET.parse(os.path.join(self.xml_path, self.file_names[idx]))

        root = tree.getroot()

        map_designs = ((torch.tensor(proc_grid(root[0][6])) == 1) * 1.).reshape(1, self.grid_size, self.grid_size)
        hm = torch.FloatTensor(proc_grid(root[1][2])).reshape(1, self.grid_size, self.grid_size)
        if self.clip_value != 0:
            hm = torch.where(hm >= self.clip_value, hm, torch.zeros_like(hm))

        start_maps = torch.zeros_like(map_designs)
        start = (int(root[0][1].text), int(root[0][0].text))
        start_maps[0][start[0], start[1]] = 1

        goal_maps = torch.zeros_like(map_designs)
        goal = (int(root[0][3].text), int(root[0][2].text))
        goal_maps[0][goal[0], goal[1]] = 1

        koef = torch.zeros_like(hm)
        if self.koef_path is not None:
            tree = ET.parse(os.path.join(self.koef_path, self.file_names[idx]))
            root = tree.getroot()
            koef = torch.FloatTensor(proc_grid(root[1][2])).reshape(1, self.grid_size, self.grid_size)

        h = torch.zeros_like(hm)
        if self.h_path is not None:
            tree = ET.parse(os.path.join(self.h_path, self.file_names[idx]))
            root = tree.getroot()
            h = torch.FloatTensor(proc_grid(root[1][2])).reshape(1, self.grid_size, self.grid_size)
        
        return map_designs, start_maps, goal_maps, hm, koef, h

    
class OODMaps(Dataset):
    def __init__(self, xml_path, grid_size=64, clip_value=0.):
        self.xml_path = xml_path
        self.file_names = os.listdir(self.xml_path)
        self.size = len(self.file_names) // 3
        self.grid_size = grid_size
        self.clip_value = clip_value

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        tree = ET.parse(os.path.join(self.xml_path, f'heatmap_{idx}_f.xml'))

        root = tree.getroot()

        map_designs = ((torch.tensor(proc_grid(root[0][6])) == 1) * 1.).reshape(1, self.grid_size, self.grid_size)
        hm = torch.FloatTensor(proc_grid(root[1][2])).reshape(1, self.grid_size, self.grid_size)
        if self.clip_value != 0:
            hm = torch.where(hm >= self.clip_value, hm, torch.zeros_like(hm))

        start_maps = torch.zeros_like(map_designs)
        start = (int(root[0][1].text), int(root[0][0].text))
        start_maps[0][start[0], start[1]] = 1

        goal_maps = torch.zeros_like(map_designs)
        goal = (int(root[0][3].text), int(root[0][2].text))
        goal_maps[0][goal[0], goal[1]] = 1

        koef = torch.zeros_like(hm)
        tree = ET.parse(os.path.join(self.xml_path, f'heatmap_{idx}_k.xml'))
        root = tree.getroot()
        koef = torch.FloatTensor(proc_grid(root[1][2])).reshape(1, self.grid_size, self.grid_size)

        h = torch.zeros_like(hm)
        tree = ET.parse(os.path.join(self.xml_path, f'heatmap_{idx}_h.xml'))
        root = tree.getroot()
        h = torch.FloatTensor(proc_grid(root[1][2])).reshape(1, self.grid_size, self.grid_size)
        
        return map_designs, start_maps, goal_maps, hm, koef, h

    
class GridData(Dataset):
    """
    'mode' argument defines type of ground truth values:
        f - focal values
        h - absolute ideal heuristic values
        cf - correction factor values
    """
    def __init__(self, path, mode='f', clip_value=0.95):
        self.clip_v = clip_value
        self.mode = mode

        self.maps   = np.load(os.path.join(path,    'maps.npy'),    mmap_mode='c')
        self.goals  = np.load(os.path.join(path,    'goals.npy'),   mmap_mode='c')
        self.starts = np.load(os.path.join(path,    'starts.npy'),  mmap_mode='c')
        
        file_gt = {'f' : 'focal.npy', 'h':'abs.npy', 'cf': 'cf.npy'}[mode]
        self.gt_values = np.load(os.path.join(path, file_gt), mmap_mode='c')


    def __len__(self):
        return len(self.gt_values)
    
    
    
    def __getitem__(self, idx):
        gt_ = torch.from_numpy(self.gt_values[idx].astype('float32'))
        if self.mode == 'f':
            gt_=  torch.where( gt_ >= self.clip_v, gt_ , torch.zeros_like( torch.from_numpy(self.gt_values[idx])))
        return (torch.from_numpy(self.maps[idx].astype('float32')), 
                torch.from_numpy(self.starts[idx].astype('float32')), 
                torch.from_numpy(self.goals[idx].astype('float32')), 
                gt_ )
