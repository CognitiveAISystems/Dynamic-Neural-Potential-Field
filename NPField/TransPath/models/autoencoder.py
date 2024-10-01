import torch
import wandb
from torch import nn
import pytorch_lightning as pl

from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.attention import SpatialTransformer
from modules.pos_emb import PosEmbeds


def base_loss(criterion, na_outputs, va_outputs):
    return criterion(na_outputs.histories, va_outputs.paths)


def adv_loss(criterion, na_outputs, va_outputs):
    loss_1 = criterion(
        torch.clamp(na_outputs.histories - na_outputs.paths - va_outputs.paths, 0, 1),
        torch.zeros_like(na_outputs.histories)
    )
    na_cost = (na_outputs.paths * na_outputs.g).sum((1, 2, 3), keepdim=True)
    va_cost = (va_outputs.paths * va_outputs.g).sum((1, 2, 3), keepdim=True)
    cost_coefs = (na_cost / va_cost - 1).view(-1, 1, 1, 1)
    loss_2 = criterion(
        (na_outputs.paths - va_outputs.paths) * cost_coefs,
        torch.zeros_like(na_outputs.histories)
    )
    return loss_1 + loss_2
    

class Autoencoder(pl.LightningModule):
    def __init__(self, 
                in_channels=2, 
                out_channels=1, 
                hidden_channels=64,
                attn_blocks=4,
                attn_heads=4,
                cnn_dropout=0.15,
                attn_dropout=0.15,
                downsample_steps=3, 
                resolution=(64, 64),
                mode='f',
                *args,
                **kwargs):
        super().__init__()
        heads_dim = hidden_channels // attn_heads
        self.encoder = Encoder(in_channels, hidden_channels, downsample_steps, cnn_dropout)
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
        self.decoder = Decoder(hidden_channels, out_channels, downsample_steps, cnn_dropout)
        
        self.recon_criterion = nn.L1Loss() if mode == 'h' else nn.MSELoss()
        self.mode = mode
        self.k = 64*64 if mode == 'h' else 1
        
        self.automatic_optimization = False
        self.save_hyperparameters()

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.pos(x)
        x = self.transformer(x)
        x = self.decoder_pos(x)
        x = self.decoder(x)
        return x

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

    
class PathLogger(pl.Callback):
    def __init__(self, val_batch, num_samples=20, mode='f'):
        super().__init__()
        map_design, start, goal, gt_hmap = val_batch[:num_samples]
        inputs = torch.cat([map_design, start + goal], dim=1) if mode == 'f' else torch.cat([map_design, goal], dim=1)
        self.val_samples = inputs[:num_samples]
        if mode == 'f':
            self.hm = gt_hmap[:num_samples]
        elif mode == 'h':
            self.hm =  (gt_hmap / gt_hmap.amax(dim=(2, 3), keepdim=True))[:num_samples]
        else:
            self.hm = gt_hmap[:num_samples]
            

    def on_validation_epoch_end(self, trainer, pl_module):
        val_samples = self.val_samples.to(device=pl_module.device)
        prediction = (pl_module(val_samples) + 1) / 2
        if pl_module.mode == 'h':
            prediction = prediction * 64 * 64

        trainer.logger.experiment.log({
            'data': [wandb.Image(x) for x in torch.cat([self.val_samples, self.hm], dim=1)],
            'predictions': [wandb.Image(x) for x in torch.cat([val_samples, prediction], dim=1)]
        })
