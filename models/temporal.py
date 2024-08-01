# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformer_2d.py
# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformer_temporal.py

import torch
import math

from dataclasses import dataclass
from typing import Optional
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.models.attention import BasicTransformerBlock
from torch import nn

class SinusoidalEmbeddings(nn.Module):
    def __init__(self):
        super(SinusoidalEmbeddings, self).__init__()

    def _get_embeddings(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.cos(position * div_term)
        pe[:, 1::2] = torch.sin(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        _, seq_len, d_model = x.shape
        pe = self._get_embeddings(seq_len, d_model)
        return pe.to(x.device)
    
@dataclass
class TemporalTransformerOutput(BaseOutput):
    sample: torch.FloatTensor

class TemporalTransformer(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        norm_elementwise_affine: bool = True,
        double_self_attention: bool = True,
    ):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.positional_encoding = SinusoidalEmbeddings()
        
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    double_self_attention=double_self_attention,
                    norm_elementwise_affine=norm_elementwise_affine,
                )
                for _ in range(num_layers)
            ]
        )

        self.proj_out = nn.Linear(inner_dim, in_channels)
        

    def forward(
        self,
        hidden_states,
        return_dict: bool = True
    ):
        if hidden_states.size(2) <= 1:
            if not return_dict:
                return (hidden_states,)

            return TemporalTransformerOutput(sample=hidden_states)
        
        batch, channel, frames, height, width = hidden_states.shape

        residual = hidden_states

        hidden_states = self.norm(hidden_states)

        hidden_states = hidden_states.permute(0, 3, 4, 2, 1)
        hidden_states = hidden_states.reshape(batch * height * width, frames, channel)
        hidden_states = self.proj_in(hidden_states)
        
        pos_encodings = self.positional_encoding(hidden_states)
        hidden_states = hidden_states + pos_encodings

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states
            )

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.view(batch, height, width, frames, channel).contiguous()
        hidden_states = hidden_states.permute(0, 4, 3, 1, 2)

        hidden_states += residual

        output = hidden_states

        if not return_dict:
            return (output,)

        return TemporalTransformerOutput(sample=output)