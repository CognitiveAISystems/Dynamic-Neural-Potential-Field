import pickle
import sys
import os

# Append the path to the TransPath folder to sys.path
script_dir = os.path.dirname(__file__)  
project_dir = os.path.dirname(script_dir)  
transpath_dir = os.path.join(project_dir, 'TransPath')  # Path to the TransPath directory
sys.path.append(transpath_dir)

from math import floor
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

from modules.attention import SpatialTransformer
from modules.decoder import Decoder
from modules.encoder import Encoder
from modules.pos_emb import PosEmbeds

from casadi import vertcat , DM
import l4casadi as l4c
import imageio

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:5000"
def base_loss(criterion, na_outputs, va_outputs):
    return criterion(na_outputs.histories, va_outputs.paths)


def adv_loss(criterion, na_outputs, va_outputs):
    loss_1 = criterion(
        torch.clamp(na_outputs.histories - na_outputs.paths - va_outputs.paths, 0, 1),
        torch.zeros_like(na_outputs.histories),
    )
    na_cost = (na_outputs.paths * na_outputs.g).sum((1, 2, 3), keepdim=True)
    va_cost = (va_outputs.paths * va_outputs.g).sum((1, 2, 3), keepdim=True)
    cost_coefs = (na_cost / va_cost - 1).view(-1, 1, 1, 1)
    loss_2 = criterion(
        (na_outputs.paths - va_outputs.paths) * cost_coefs,
        torch.zeros_like(na_outputs.histories),
    )
    return loss_1 + loss_2


class Autoencoder_path(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=8,
        hidden_channels=64,
        attn_blocks=4,
        attn_heads=4,
        cnn_dropout=0.15,
        attn_dropout=0.15,
        downsample_steps=3,
        resolution=(50, 50),
        mode="f",
        *args,
        **kwargs,
    ):
        super().__init__()
        heads_dim = hidden_channels // attn_heads
        self.encoder = Encoder(
            1, hidden_channels, downsample_steps, cnn_dropout, num_groups=32
        )
        self.encoder_robot = Encoder(1, 1, 3, 0.15, num_groups=1)
        
        self.pos = PosEmbeds(
            hidden_channels,
            (
                resolution[0] // 2**downsample_steps,
                resolution[1] // 2**downsample_steps,
            ),
        )
        self.transformer = SpatialTransformer(
            hidden_channels, attn_heads, heads_dim, attn_blocks, attn_dropout
        )
        self.decoder_pos = PosEmbeds(
            hidden_channels,
            (
                resolution[0] // 2**downsample_steps,
                resolution[1] // 2**downsample_steps,
            ),
        )
        self.decoder = Decoder(hidden_channels, out_channels//2, 1, cnn_dropout)           ####### out_channels

        self.x_cord = nn.Sequential(nn.Linear(1, 16), nn.ReLU())
        self.y_cord = nn.Sequential(nn.Linear(1, 16), nn.ReLU())
        self.theta_sin = nn.Sequential(nn.Linear(1, 16), nn.ReLU())
        self.theta_cos = nn.Sequential(nn.Linear(1, 16), nn.ReLU())

        self.encoder_after = Encoder(hidden_channels, 32, 1, 0.15, num_groups=32)
        self.decoder_after = Decoder(32, hidden_channels, 1, 0.15, num_groups=32)
        
        self.decoder_MAP = Decoder(hidden_channels, 2, 3, 0.15, num_groups=32)
        
        """
        self.linear_after_mean = nn.Sequential(
            nn.Linear(1225, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            #nn.Sigmoid(),
        )
        """
        
        self.linear_after_mean = nn.Sequential(
            nn.Linear(676, 256),                       # 1225
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.GELU(),
        )

        self.recon_criterion = nn.MSELoss()
        self.mode = mode
        self.k = 1
        self.automatic_optimization = False
        self.device = torch.device("cuda")
        # self.save_hyperparameters()

    def forward(self, batch):
        batch = batch.reshape(-1, 615)

        map_encode_robot = batch[..., :-3].to(self.device)

        x_crd = torch.reshape(batch[..., -3:-2].to(self.device), (-1, 1))
        y_crd = torch.reshape(batch[..., -2:-1].to(self.device), (-1, 1))
        theta = torch.reshape(batch[..., -1:].to(self.device), (-1, 1))

        x_cr_encode = self.x_cord(x_crd)
        y_cr_encode = self.y_cord(y_crd)
        tsin_encode = self.theta_sin(torch.sin(theta))
        tcos_encode = self.theta_cos(torch.cos(theta))

        encoded_input = torch.cat(
            (map_encode_robot, x_cr_encode, y_cr_encode, tsin_encode, tcos_encode), 1
        )
        # encoded_input = self.linear_after(encoded_input)
        #encoded_input_max = self.linear_after_max(encoded_input)
        encoded_input_mean = self.linear_after_mean(encoded_input)

        return encoded_input_mean

    def encode_map_footprint(self, batch):
        mapp = batch[..., :2500].to(self.device)
        mapp = torch.reshape(mapp, (-1, 1, 50, 50))

        footprint = batch[..., 2500:].to(self.device)
        footprint = torch.reshape(footprint, (-1, 1, 50, 50))

        map_encode = self.encoder(mapp)

        map_encode_robot = (
            self.encoder_robot(footprint).flatten().view(mapp.shape[0], -1)
        )

        encoded_input = self.encoder_after(map_encode)
        encoded_input = self.decoder_after(encoded_input)
        encoded_input = self.pos(encoded_input)
        encoded_input = self.transformer(encoded_input)
        encoded_input = self.decoder_pos(encoded_input)
        encoded_input = self.decoder(encoded_input).view(encoded_input.shape[0], -1)

        encoded_input = torch.cat((encoded_input, map_encode_robot), -1)

        return encoded_input

    def encode_map_pos(self, batch):
        batch = batch.reshape(-1, 615)

        map_encode_robot = batch[..., :-3].to(self.device)

        x_crd = torch.reshape(batch[..., -3:-2].to(self.device), (-1, 1))
        y_crd = torch.reshape(batch[..., -2:-1].to(self.device), (-1, 1))
        theta = torch.reshape(batch[..., -1:].to(self.device), (-1, 1))

        x_cr_encode = self.x_cord(x_crd)
        y_cr_encode = self.y_cord(y_crd)
        tsin_encode = self.theta_sin(torch.sin(theta))
        tcos_encode = self.theta_cos(torch.cos(theta))

        encoded_input = torch.cat(
            (map_encode_robot, x_cr_encode, y_cr_encode, tsin_encode, tcos_encode), 1
        )
        # encoded_input = self.linear_after(encoded_input)
        #encoded_input_max = self.linear_after_max(encoded_input)
        encoded_input_mean = self.linear_after_mean(encoded_input)

        return encoded_input_mean
    def encode_map_footprint(self, batch):
        mapp = batch[..., :2500].to(self.device)
        mapp = torch.reshape(mapp, (-1, 1, 50, 50))

        footprint = batch[..., 2500:].to(self.device)
        footprint = torch.reshape(footprint, (-1, 1, 50, 50))

        map_encode = self.encoder(mapp)

        map_encode_robot = (
            self.encoder_robot(footprint).flatten().view(mapp.shape[0], -1)
        )

        encoded_input = self.encoder_after(map_encode)
        encoded_input = self.decoder_after(encoded_input)
        encoded_input = self.pos(encoded_input)
        encoded_input = self.transformer(encoded_input)
        encoded_input = self.decoder_pos(encoded_input)
        encoded_input = self.decoder(encoded_input).view(encoded_input.shape[0], -1)

        encoded_input = torch.cat((encoded_input, map_encode_robot), -1)

        return encoded_input

    def encode_map_pos(self, batch):
        batch = batch.reshape(-1, 615)                      # 1164

        map_encode_robot = batch[..., :-3].to(self.device)

        x_crd = torch.reshape(batch[..., -3:-2].to(self.device), (-1, 1))
        y_crd = torch.reshape(batch[..., -2:-1].to(self.device), (-1, 1))
        theta = torch.reshape(batch[..., -1:].to(self.device), (-1, 1))

        x_cr_encode = self.x_cord(x_crd)
        y_cr_encode = self.y_cord(y_crd)
        tsin_encode = self.theta_sin(torch.sin(theta))
        tcos_encode = self.theta_cos(torch.cos(theta))

        encoded_input = torch.cat(
            (map_encode_robot, x_cr_encode, y_cr_encode, tsin_encode, tcos_encode), 1
        )
        # encoded_input = self.linear_after(encoded_input)
        #encoded_input_max = self.linear_after_max(encoded_input)
        encoded_input_mean = self.linear_after_mean(encoded_input)

        return encoded_input_mean
    
    def step_ctrl(self, batch):
        mapp, x_crd, y_crd, theta = batch
        map_encode = self.encoder(mapp[:, :1, :, :])

        map_encode_robot = (
            self.encoder_robot(mapp[:, -1:, :, :]).flatten().view(mapp.shape[0], -1)
        )
        x_cr_encode = self.x_cord(x_crd)
        y_cr_encode = self.y_cord(y_crd)
        tsin_encode = self.theta_sin(torch.sin(theta))
        tcos_encode = self.theta_cos(torch.cos(theta))

        encoded_input = map_encode 
        encoded_input = self.encoder_after(encoded_input)
        encoded_input = self.decoder_after(encoded_input)
        
        decoded_map = self.decoder_MAP(encoded_input)
        
        encoded_input = self.pos(encoded_input)
        encoded_input = self.transformer(encoded_input)
        encoded_input = self.decoder_pos(encoded_input)
        encoded_input = self.decoder(encoded_input).view(encoded_input.shape[0], -1)

        encoded_input = torch.cat(
            (
                encoded_input,
                map_encode_robot,
                x_cr_encode,
                y_cr_encode,
                tsin_encode,
                tcos_encode,
            ),
            1,
        )

        #encoded_input_max = self.linear_after_max(encoded_input)
        encoded_input_mean = self.linear_after_mean(encoded_input)

        return encoded_input_mean #, decoded_map

    def training_step(self, batch, output):
        optimizer = self.optimizers()
        sch = self.lr_schedulers()

        loss = self.step_ctrl(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sch.step()

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("lr", sch.get_last_lr()[0], on_step=True, on_epoch=False)
        return loss

    def step(self, batch, batch_idx, regime):
        map_design, start, goal, gt_hmap = batch
        inputs = (
            torch.cat([map_design, start + goal], dim=1)
            if self.mode in ("f", "nastar")
            else torch.cat([map_design, goal], dim=1)
        )
        predictions = self(inputs)

        loss = self.recon_criterion((predictions + 1) / 2 * self.k, gt_hmap)
        self.log(f"{regime}_recon_loss", loss, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        sch = self.lr_schedulers()

        loss = self.step(batch, batch_idx, "train")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sch.step()

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("lr", sch.get_last_lr()[0], on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, "val")
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0004)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=4e-4, total_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [scheduler]
