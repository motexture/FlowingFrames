# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformer_2d.py
# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformer_temporal.py

import torch

from dataclasses import dataclass
from typing import Optional
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.models.attention import BasicTransformerBlock
from torch import nn
    
@dataclass
class SpatialTransformerOutput(BaseOutput):
    sample: torch.FloatTensor

class SpatialTransformer(ModelMixin, ConfigMixin):
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
        num_embeds_ada_norm: Optional[int] = None,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
    ):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
            deprecation_message = (
                f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
                " incorrectly set to `'layer_norm'`.Make sure to set `norm_type` to `'ada_norm'` in the config."
                " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
                " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
                " would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
            norm_type = "ada_norm"

        self.in_channels = in_channels
        
        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                )
                for _ in range(num_layers)
            ]
        )

        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ):
        batch, channel, frames, height, width = hidden_states.shape

        residual = hidden_states

        hidden_states = self.norm(hidden_states)

        hidden_states = hidden_states.permute(0, 2, 3, 4, 1)
        hidden_states = hidden_states.reshape(batch * frames, height * width, channel)
        hidden_states = self.proj_in(hidden_states)     

        encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeats=frames, dim=0)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states
            )

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.view(batch, frames, height, width, channel).contiguous()
        hidden_states = hidden_states.permute(0, 4, 1, 2, 3)

        hidden_states += residual

        output = hidden_states

        if not return_dict:
            return (output,)

        return SpatialTransformerOutput(sample=output)