import numpy as np
import pickle
from math import floor 
import torch
from torch import nn
import pytorch_lightning as pl
import sys
sys.path.append('/home/vladislav/l4casadi/examples/TransPath')
#sys.path.append('/mnt/c/Users/Alsta/OneDrive/Desktop/Docker_HOME/root/l4casadi/examples/TransPath')
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.attention import SpatialTransformer
from modules.pos_emb import PosEmbeds
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.func import vmap, jacrev, hessian
from torch.fx.experimental.proxy_tensor import make_fx

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:5000"


class Autoencoder_path(nn.Module):
    def __init__(self, 
                in_channels=2, 
                out_channels=1, 
                hidden_channels=32,
                attn_blocks=4,
                attn_heads=4,
                cnn_dropout=0.15,
                attn_dropout=0.15,
                downsample_steps=3, 
                resolution=(256, 256),
                mode='f',
                *args,
                **kwargs):
        super().__init__()
        heads_dim = hidden_channels // attn_heads
        self.encoder = Encoder(1, hidden_channels, downsample_steps, cnn_dropout, num_groups=32)
        self.encoder_robot = Encoder(1, 1, 4, 0.15,num_groups=1)
        self.pos = PosEmbeds(
            hidden_channels, 
            (resolution[0] // 2**downsample_steps, resolution[1] // 2**downsample_steps)
        )
        self.transformer = SpatialTransformer(
            hidden_channels, 
            attn_heads,
            heads_dim,
            attn_blocks, 
            attn_dropout
        )
        self.decoder_pos = PosEmbeds(
            hidden_channels, 
            (resolution[0] // 2**downsample_steps, resolution[1] // 2**downsample_steps)
        )
        self.decoder = Decoder(hidden_channels, out_channels, 1, cnn_dropout)
        
        self.x_cord = nn.Sequential(nn.Linear(1, 16),nn.ReLU())
        self.y_cord = nn.Sequential(nn.Linear(1, 16),nn.ReLU())
        self.theta_sin = nn.Sequential(nn.Linear(1, 16),nn.ReLU())
        self.theta_cos = nn.Sequential(nn.Linear(1, 16),nn.ReLU())
        
        self.encoder_after = Encoder(32, 32, 1, 0.15, num_groups=32)
        self.decoder_after = Decoder(32, 32, 1, 0.15, num_groups=32)
        self.linear_after = nn.Sequential(
                    nn.Linear(4416, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                )
        self.recon_criterion = nn.MSELoss()
        self.mode = mode
        self.k = 1
        self.automatic_optimization = False
        self.example_input_array = torch.zeros((131072+3))
        self.device = torch.device('cuda')
        #self.save_hyperparameters()


    def encode_map_footprint(self,batch):
        
        mapp = batch[...,:65536].to(self.device)
        mapp = torch.reshape(mapp,(-1,1,256,256))
        
        footprint = batch[...,65536:].to(self.device)
        footprint = torch.reshape(footprint,(-1,1,256,256))

        map_encode = self.encoder(mapp)
        map_encode_robot = self.encoder_robot(footprint).flatten().view(mapp.shape[0],-1)

        encoded_input = self.encoder_after(map_encode)
        encoded_input = self.decoder_after(encoded_input)
        encoded_input = self.pos(encoded_input)
        encoded_input = self.transformer(encoded_input)
        encoded_input = self.decoder_pos(encoded_input)
        encoded_input = self.decoder(encoded_input).view(encoded_input.shape[0],-1)
        encoded_input = torch.cat((encoded_input, map_encode_robot), -1)

        print(encoded_input.shape)

        return encoded_input
    
    def encode_map_pos(self,batch):

        map_encode_robot = batch[:,:-3].to(self.device)

        x_crd = torch.reshape(batch[...,-3:-2].to(self.device),(-1,1))
        y_crd = torch.reshape(batch[...,-2:-1].to(self.device),(-1,1))
        theta = torch.reshape(batch[...,-1:].to(self.device),(-1,1))

        x_cr_encode = self.x_cord(x_crd)
        y_cr_encode = self.y_cord(y_crd)
        tsin_encode = self.theta_sin(torch.sin(theta))
        tcos_encode = self.theta_cos(torch.cos(theta))

        encoded_input = torch.cat((map_encode_robot, x_cr_encode, y_cr_encode, tsin_encode, tcos_encode), -1)
        encoded_input = self.linear_after(encoded_input)

        return encoded_input


    
    def forward(self,batch):
        
        print('BATCH SHAPE: ',batch.shape)
        
        mapp = batch[...,:65536].to(self.device)
        mapp = torch.reshape(mapp,(-1,1,256,256))
        
        footprint = batch[...,65536:-3].to(self.device)
        footprint = torch.reshape(footprint,(-1,1,256,256))
        
        x_crd = torch.reshape(batch[...,-3:-2].to(self.device),(-1,1))
        y_crd = torch.reshape(batch[...,-2:-1].to(self.device),(-1,1))
        theta = torch.reshape(batch[...,-1:].to(self.device),(-1,1))
        
        
        #print(mapp.shape)
        map_encode = self.encoder(mapp)
        #print('Map_Encode: ',map_encode.shape)
        map_encode_robot = self.encoder_robot(footprint).flatten().view(mapp.shape[0],-1)
        #print('Map_ROBOT_Encode: ',map_encode_robot.shape)
        x_cr_encode = self.x_cord(x_crd)
        #print('X_Encode: ',x_cr_encode.shape)
        y_cr_encode = self.y_cord(y_crd)
        #print('Y_Encode: ',y_cr_encode.shape)
        tsin_encode = self.theta_sin(torch.sin(theta))
        #print('ThetaS_Encode: ',tsin_encode.shape)
        tcos_encode = self.theta_cos(torch.cos(theta))
        #print('ThetaC_Encode: ',tcos_encode.shape)
        
        encoded_input = map_encode #torch.cat((map_encode, x_cr_encode, y_cr_encode, tsin_encode, tcos_encode), 1)
        #print(encoded_input.shape)
        encoded_input = self.encoder_after(encoded_input)
        #print(encoded_input.shape)
        encoded_input = self.decoder_after(encoded_input)
        #print(encoded_input.shape)
        encoded_input = self.pos(encoded_input)
        #print(encoded_input.shape)
        encoded_input = self.transformer(encoded_input)
        #print(encoded_input.shape)
        encoded_input = self.decoder_pos(encoded_input)
        #print(encoded_input.shape)
        encoded_input = self.decoder(encoded_input).view(encoded_input.shape[0],-1)
        print(encoded_input.shape,map_encode_robot.shape,x_cr_encode.shape)
        encoded_input = torch.cat((encoded_input, map_encode_robot, x_cr_encode, y_cr_encode, tsin_encode, tcos_encode), -1)
        #print(encoded_input.shape)
        encoded_input = self.linear_after(encoded_input)
        #print(encoded_input.shape)
        
   #     loss = self.recon_criterion(encoded_input,output)
     #   self.log(f'recon_loss', loss, on_step=False, on_epoch=True)
        return encoded_input
    
    def training_step(self, batch, output):
        
        optimizer = self.optimizers()
        sch = self.lr_schedulers()
        
        loss = self.step_ctrl(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sch.step()
        
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('lr', sch.get_last_lr()[0], on_step=True, on_epoch=False)
        return loss
        

    def step(self, batch, batch_idx, regime):
        map_design, start, goal, gt_hmap = batch
        inputs = torch.cat([map_design, start + goal], dim=1) if self.mode in ('f', 'nastar') else torch.cat([map_design, goal], dim=1)
        predictions = self(inputs)

        loss = self.recon_criterion((predictions + 1) / 2 * self.k, gt_hmap)
        self.log(f'{regime}_recon_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        sch = self.lr_schedulers()
        
        loss = self.step(batch, batch_idx, 'train')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sch.step()
        
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('lr', sch.get_last_lr()[0], on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'val')
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss
                 
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0004)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=4e-4, total_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [scheduler]
    
    
############### Loading learned model ############
device = torch.device('cuda')
model_loaded = Autoencoder_path(mode='k')
model_loaded.to(device)
model_loaded.eval()

model_loaded.load_state_dict(torch.load('/home/vladislav/l4casadi/examples/TransPath/learnedmodel.pth'))
#model_loaded.to_torchscript(method="trace")

print(model_loaded)


############### test learned model #################


map_inp = torch.ones((1,131072))
coords = torch.ones((1,3))

result = model_loaded.encode_map_footprint(map_inp).detach().cpu()

batch = torch.cat((result, coords), -1)

result1 = model_loaded.encode_map_pos(batch)

hess = hessian(model_loaded.encode_map_pos)(batch)


#hess_model = hessian(model_loaded)(map_inp)


#result = torch.clamp(model_loaded(map_inp), min=0.).detach().item()

print(result)    
print(result1)   
print(hess)