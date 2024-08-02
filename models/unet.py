# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_3d_condition.py

import torch
import torch.nn as nn
import torch.utils.checkpoint

from typing import List, Optional, Tuple, Union
from models.resnet import InflatedConv
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from .unet_blocks import (
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    DownBlock3D,
    UNetMidBlock3DCrossAttn,
    UpBlock3D,
    get_down_block,
    get_up_block
)

logger = logging.get_logger(__name__)

def print_state_dict_to_file(state_dict, filename):
    with open(filename, 'w') as f:
        for name, param in state_dict.items():
            f.write(f'Layer: {name} | Size: {param.size()}\n')

class FlowingFrames(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = 128,
        in_channels: int = 4,
        out_channels: int = 4,
        down_block_types: Tuple[str] = (
            "DownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D"
        ),
        up_block_types: Tuple[str] = ("CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "UpBlock3D"),
        block_out_channels: Tuple[int] = (320, 640, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        transformer_layers_per_block: Union[int, Tuple[int]] = [1, 2, 10],
        temporal_layers_per_block: Union[int, Tuple[int]] = [2, 2, 2],
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        upcast_attention: bool = False,
        flip_sin_to_cos: bool = True,
        addition_time_embed_dim: Optional[int] = 256,
        projection_class_embeddings_input_dim: Optional[int] = 2816,
        freq_shift: int = 0,
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 2048,
        attention_head_dim: Union[int, Tuple[int]] = 64,
    ):
        super().__init__()

        self.sample_size = sample_size
        self.gradient_checkpointing = False
        
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
        )

        self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
        self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
         
        self.conv_in = InflatedConv(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = (transformer_layers_per_block,) * len(down_block_types)

        if isinstance(temporal_layers_per_block, int):
            temporal_layers_per_block = (temporal_layers_per_block,) * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers=transformer_layers_per_block[i],
                temporal_layers=temporal_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attention_head_dim=attention_head_dim[i],
                downsample_padding=downsample_padding,
                upcast_attention=upcast_attention
            )
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlock3DCrossAttn(
            in_channels=block_out_channels[-1],            
            temb_channels=time_embed_dim,
            transformer_layers=transformer_layers_per_block[-1],
            temporal_layers=temporal_layers_per_block[-1],
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim[-1],
            resnet_groups=norm_num_groups,
            upcast_attention=upcast_attention
        )
        
        self.num_upsamplers = 0

        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))
        reversed_temporal_layers_per_block = list(reversed(temporal_layers_per_block))
        reversed_attention_head_dim = list(reversed(attention_head_dim))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers=reversed_transformer_layers_per_block[i],
                temporal_layers=reversed_temporal_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attention_head_dim=reversed_attention_head_dim[i],
                upcast_attention=upcast_attention
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = InflatedConv(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def _set_gradient_checkpointing(self, value=False):
        self.gradient_checkpointing = value
        self.mid_block.gradient_checkpointing = value
        for module in self.down_blocks + self.up_blocks:
            if isinstance(module, (CrossAttnDownBlock3D, DownBlock3D, CrossAttnUpBlock3D, UpBlock3D)):
                module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        text_embeds: torch.Tensor,
        time_ids: torch.Tensor,
        previous_latents: Optional[torch.Tensor] = None,
    ) -> Tuple:
        default_overall_up_factor = 2**self.num_upsamplers
        
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        timesteps = timestep.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.dtype)
        
        emb = self.time_embedding(t_emb, None)

        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(add_embeds)
        emb = emb + aug_emb

        sample = self.conv_in(sample)
        down_block_res_samples = (sample,)

        for i, downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    conditioning_hidden_states=previous_latents
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample, 
                    temb=emb
                )
            down_block_res_samples += res_samples

        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                conditioning_hidden_states=previous_latents
            )

        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    conditioning_hidden_states=previous_latents,
                    upsample_size=upsample_size,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size
                )
        
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample