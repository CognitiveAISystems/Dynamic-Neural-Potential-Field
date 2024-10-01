import torch 
from torch import nn

from .resblock import ResnetBlock


def nonlinearity(x):
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, upsample_steps, dropout=0.1, num_groups=32):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(upsample_steps):
            self.layers.append(
                nn.Sequential(
                    ResnetBlock(hidden_channels, hidden_channels, dropout, num_groups),
                    Upsample(hidden_channels)
                )
            )
        self.norm = Normalize(hidden_channels, num_groups)
        self.conv_out = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = nonlinearity(x)
        x = self.conv_out(x)
        return torch.tanh(x)
