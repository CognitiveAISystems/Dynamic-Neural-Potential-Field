import pickle
import sys
from math import floor
import os
import sys

# Append the path to the TransPath folder to sys.path
script_dir = os.path.dirname(__file__)  
project_dir = os.path.dirname(script_dir)  
transpath_dir = os.path.join(project_dir, 'TransPath')  # Path to the TransPath directory
sys.path.append(transpath_dir)

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from modules.attention import SpatialTransformer
from modules.decoder import Decoder

from modules.encoder import Encoder
from modules.pos_emb import PosEmbeds
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
import random  
import imageio
from model_nn_GPT import GPTConfig, GPT 


#mkdir -p build
#cd build
#cmake -DACADOS_WITH_QPOASES=ON ..
#make install -j4


#pip install -r requirements_build.txt
#pip install l4casadi --no-build-isolation

#pip install -e /root/NPField/acados/interfaces/acados_template
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/root/NPField/acados/lib"
# export ACADOS_SOURCE_DIR="/root/NPField/acados"

# make shared_library
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/NPField/acados/lib
# make examples_c
# make run_examples_c
    

sub_maps_all = pickle.load(open("../../dataset/dataset1000/dataset_1000_maps_0_100_all.pkl", "rb"))
footprints = pickle.load(open("../../dataset/dataset1000/data_footprint.pkl", "rb"))
dyn_obst_info = pickle.load(open("../../dataset/dataset1000/dataset_initial_position_dynamic_obst.pkl", "rb"))
obst_motion_info = pickle.load(open("../../dataset/dataset1000/dataset_1000_maps_obst_motion.pkl", "rb"))



n_layer = 4
n_head = 4
n_embd = 576
dropout = 0.1 # for pretraining 0 is good, for finetuning try 0.1+
bias = True #False
block_size = 1024
device = 'cuda'

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)
model_args['vocab_size'] = 1024 # 50257 2048

gptconf = GPTConfig(**model_args)

model_gpt = GPT(gptconf)
checkpoint_name = '../../trained-models/NPField_onlyGPT_predmap9.pth'
#model_gpt.load_state_dict(torch.load(checkpoint_name))



pretrained_dict = torch.load(checkpoint_name)
model_dict = model_gpt.state_dict()
# 1. filter out unnecessary keys
rejected_keys = [k for k, v in model_dict.items() if k not in pretrained_dict]
print('REJECTED KEYS test GPT: ',rejected_keys)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# 3. load the new state dict
model_gpt.load_state_dict(model_dict)


model_gpt.to(device);
model_gpt.eval();


ANGLE = 2.0
EPISODE = 1

import sys
print(f'Episode ID: {sys.argv[1]} ID_dyn: {sys.argv[2]} Angle: {sys.argv[3]}')

EPISODE = int(sys.argv[1])
ID_DYN = int(sys.argv[2])
ANGLE = float(sys.argv[3])


for theta_angle in [ANGLE]:

    for map_id in [EPISODE]:


        id_dyn = ID_DYN
        time = 10
        resolution = 100


        res_array = np.zeros((time, resolution, resolution))
        test_map = sub_maps_all["submaps"][map_id,id_dyn]#/100.
        test_footprint = footprints["footprint_husky"]#/100.

        
        map_inp = (torch.tensor(np.hstack((test_map.flatten(), test_footprint.flatten()))).unsqueeze(0).float().to(device)/100.)
        theta = torch.tensor([theta_angle]).unsqueeze(0).float().to(device)   #np.deg2rad([90])
        d_info = obst_motion_info['motion_dynamic_obst'][map_id,id_dyn]

        with torch.no_grad():
            result = model_gpt.encode_map_footprint(torch.hstack((map_inp,torch.from_numpy(d_info).cuda().unsqueeze(0).float())))

            for ii, i in enumerate(np.arange(0.0, 5, 5 / resolution)):
                for jj, j in enumerate(np.arange(0.0, 5, 5 / resolution)):

                    x = torch.tensor(i).unsqueeze(0).unsqueeze(0).to(device).float()
                    y = torch.tensor(j).unsqueeze(0).unsqueeze(0).to(device).float()

                    input_batch = torch.hstack((result, x, y, theta))

                    npf_10 = model_gpt(input_batch)   # encode_map_pos

                    res_array[:, ii, jj] = npf_10.detach().cpu().numpy()


        frames = []
        for tme in range(time):
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(15,5), gridspec_kw={ 'wspace' : 0.3, 'hspace' : 0.1})
            ax1.imshow(sub_maps_all["submaps"][map_id,tme])
            ax2.imshow(np.rot90(res_array[tme]))
            # ax3.scatter(
            #     x=potential_2_husky["position_potential"][map_id,id_dyn,:,::4,0],
            #     y=potential_2_husky["position_potential"][map_id,id_dyn,:,::4,1],
            #     s=potential_2_husky["position_potential"][map_id,id_dyn,:,::4,3]/8,
            # )
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(data)
            plt.close(fig)

        imageio.mimsave('NPField_D3_Test_ep{}_angle_{}.gif'.format(map_id,str(d_info[2])[:3]), frames, format="GIF", loop=65535)
        print('NPField_D3_Test_ep{}_angle_{}.gif'.format(map_id,str(d_info[2])[:3]))