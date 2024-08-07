import torch.nn as nn

from models.spatial import SpatialTransformer
from models.temporal import TemporalTransformer
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

class Refiner(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, in_channels=4, inner_channels=320, out_channels=4):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels=in_channels, out_channels=inner_channels, kernel_size=(7, 7, 7), stride=(1, 1, 1), padding=(3, 3, 3))

        self.norm1 = nn.GroupNorm(num_channels=inner_channels, num_groups=32)
        self.silu1 = nn.SiLU()
        self.conv1 = nn.Conv3d(in_channels=inner_channels, out_channels=inner_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.norm2 = nn.GroupNorm(num_channels=inner_channels, num_groups=32)
        self.silu2 = nn.SiLU()
        self.conv2 = nn.Conv3d(in_channels=inner_channels, out_channels=inner_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.norm3 = nn.GroupNorm(num_channels=inner_channels, num_groups=32)
        self.silu3 = nn.SiLU()
        self.conv3 = nn.Conv3d(in_channels=inner_channels, out_channels=inner_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.spatial_attn = SpatialTransformer(8, 64, in_channels=inner_channels, num_layers=6, double_self_attention=True, dropout=0.1)
        self.temporal_attn = TemporalTransformer(8, 64, in_channels=inner_channels, num_layers=6, double_self_attention=True, dropout=0.1)
        
        self.conv_norm = nn.GroupNorm(num_channels=inner_channels, num_groups=8)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv3d(in_channels=inner_channels, out_channels=out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        nn.init.zeros_(self.conv_out.weight)
        if self.conv_out.bias is not None:
            nn.init.zeros_(self.conv_out.bias)

    def forward(self, hidden_states):
        identity = hidden_states

        hidden_states = self.conv_in(hidden_states)
        hidden_states = self.conv1(self.silu1(self.norm1(hidden_states)))
        hidden_states = self.conv2(self.silu2(self.norm2(hidden_states)))
        hidden_states = self.conv3(self.silu3(self.norm3(hidden_states)))
        hidden_states = self.spatial_attn(hidden_states).sample
        hidden_states = self.temporal_attn(hidden_states).sample
        hidden_states = self.conv_norm(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        hidden_states = identity + hidden_states
        
        return hidden_states