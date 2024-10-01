import numpy as np
import pickle
from math import floor 
import torch
#import wandb
from torch import nn
import pytorch_lightning as pl
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.attention import SpatialTransformer
from modules.pos_emb import PosEmbeds
import torch.nn as nn
import matplotlib.pyplot as plt


class Autoencoder_path(pl.LightningModule):
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
        self.save_hyperparameters()

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.pos(x)
        x = self.transformer(x)
        x = self.decoder_pos(x)
        x = self.decoder(x)
        return x
    
    def step_ctrl(self,batch):
        mapp, x_crd, y_crd, theta = batch
        #print(mapp.shape)
        map_encode = self.encoder(mapp[:,:1,:,:])
        #print('Map_Encode: ',map_encode.shape)
        map_encode_robot = self.encoder_robot(mapp[:,0:1,:,:]).flatten().view(mapp.shape[0],-1)
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
        #print(encoded_input.shape)
        encoded_input = torch.cat((encoded_input, map_encode_robot, x_cr_encode, y_cr_encode, tsin_encode, tcos_encode), 1)
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
model_loaded.load_state_dict(torch.load('learnedmodel.pth'))
print(model_loaded)


############### test learned model ###############

data = {'input_map':torch.zeros((256,256)),'footprint':torch.zeros((256,256))}
map_inp = torch.tensor(np.stack((data['input_map'],data['footprint']))).unsqueeze(0).float().to(device)
x = torch.tensor([2.5]).float().unsqueeze(0).to(device)
y = torch.tensor([3.5]).float().unsqueeze(0).to(device)
theta = torch.tensor(np.deg2rad([90])).unsqueeze(0).float().to(device)  

X_test = (map_inp, x, y, theta)

print(model_loaded.step_ctrl(X_test))    