from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange 
from typing import List
import math

from ..config import DDPM_Config

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps: int, embed_dim: int):
        super(SinusoidalEmbeddings, self).__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings

    def forward(self, x, t):
        embeds = self.embeddings.to(x.device)[t]
        return embeds[:, :, None, None]

class ResBlock(nn.Module):
    def __init__(self, C: int, num_groups: int, dropout_prob: float):
        super(ResBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(self, x, embeddings):
        x = x + embeddings[:, :x.shape[1], :, :]
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return r + x

class Attention(nn.Module):
    def __init__(self, C: int, num_heads:int , dropout_prob: float):
        super(Attention, self).__init__()
        self.proj1 = nn.Linear(C, C*3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        h, w = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        q,k,v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(q,k,v, is_causal=False, dropout_p=self.dropout_prob)
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, 'b h w C -> b C h w')

class UnetLayer(nn.Module):
    def __init__(self, 
            upscale: bool, 
            attention: bool, 
            num_groups: int, 
            dropout_prob: float,
            num_heads: int,
            C: int
    ):
        super(UnetLayer, self).__init__()
        self.ResBlock1 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        if upscale:
            self.conv = nn.ConvTranspose2d(C, C//2, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(C, C*2, kernel_size=3, stride=2, padding=1)
        if attention:
            self.attention_layer = Attention(C, num_heads=num_heads, dropout_prob=dropout_prob)

    def forward(self, x, embeddings):
        x = self.ResBlock1(x, embeddings)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        x = self.ResBlock2(x, embeddings)
        return self.conv(x), x
    
class Diffusion_Unet(nn.Module):
    def __init__(self,
            Config: Optional[DDPM_Config] = None,
            Channels: Optional[List[int]] = None,
            Attentions: Optional[List[bool]] = None,
            Upscales: Optional[List[bool]] = None,
            num_groups: Optional[int] = None,
            dropout_prob: Optional[float] = None,
            num_heads: Optional[int] = None,
            input_channels: Optional[int] = None,
            output_channels: Optional[int] = None,
            time_steps: Optional[int] = None
    ):
        super(Diffusion_Unet, self).__init__()

        # Use config if provided, otherwise use default
        if Config is not None:
            self.Channels = Channels if Channels is not None else Config.Channels
            self.Attentions = Attentions if Attentions is not None else Config.Attentions
            self.Upscales = Upscales if Upscales is not None else Config.Upscales
            self.num_groups = num_groups if num_groups is not None else Config.num_groups
            self.dropout_prob = dropout_prob if dropout_prob is not None else Config.dropout_prob
            self.num_heads = num_heads if num_heads is not None else Config.num_heads
            self.input_channels = input_channels if input_channels is not None else Config.input_channels
            self.output_channels = output_channels if output_channels is not None else Config.output_channels
            self.time_steps = time_steps if time_steps is not None else Config.time_steps
        else:
            self.Channels = Channels if Channels is not None else [64, 128, 256, 512, 512, 384]
            self.Attentions = Attentions if Attentions is not None else [False, True, False, False, False, True]
            self.Upscales = Upscales if Upscales is not None else [False, False, False, True, True, True]
            self.num_groups = num_groups if num_groups is not None else 32
            self.dropout_prob = dropout_prob if dropout_prob is not None else 0.1
            self.num_heads = num_heads if num_heads is not None else 8
            self.input_channels = input_channels if input_channels is not None else 1
            self.output_channels = output_channels if output_channels is not None else 1
            self.time_steps = time_steps if time_steps is not None else 1000
        
        self.num_layers = len(self.Channels)
        self.shallow_conv = nn.Conv2d(self.input_channels, self.Channels[0], kernel_size=3, padding=1)
        out_channels = (self.Channels[-1]//2)+self.Channels[0]
        self.late_conv = nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels//2, self.output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.embeddings = SinusoidalEmbeddings(time_steps=self.time_steps, embed_dim=max(self.Channels))
        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=self.Upscales[i],
                attention=self.Attentions[i],
                num_groups=self.num_groups,
                dropout_prob=self.dropout_prob,
                C=self.Channels[i],
                num_heads=self.num_heads
            )
            setattr(self, f'Layer{i+1}', layer)

    def forward(self, x, t):
        x = self.shallow_conv(x)
        residuals = []
        for i in range(self.num_layers//2):
            layer = getattr(self, f'Layer{i+1}')
            embeddings = self.embeddings(x, t)
            x, r = layer(x, embeddings)
            residuals.append(r)
        for i in range(self.num_layers//2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x = torch.concat((layer(x, embeddings)[0], residuals[self.num_layers-i-1]), dim=1)
        return self.output_conv(self.relu(self.late_conv(x)))
    


