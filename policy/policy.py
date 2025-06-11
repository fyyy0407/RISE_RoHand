import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from policy.tokenizer import Sparse3DEncoder
from policy.transformer import Transformer
from policy.diffusion import DiffusionUNetPolicy
from policy.vqvae_rise.vqvae import VqVae

class RISE(nn.Module):
    def __init__(
        self, 
        num_action = 20,
        input_dim = 6,
        obs_feature_dim = 512, 
        action_dim = 15, 
        hidden_dim = 512,
        nheads = 8, 
        num_encoder_layers = 4, 
        num_decoder_layers = 1, 
        dim_feedforward = 2048, 
        dropout = 0.1
    ):
        super().__init__()
        num_obs = 1
        self.sparse_encoder = Sparse3DEncoder(input_dim, obs_feature_dim)
        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, obs_feature_dim)
        self.readout_embed = nn.Embedding(1, hidden_dim)
        self.vqvae_model=VqVae(
            obs_dim=60,
            input_dim_h=2,         # Sequence length
            input_dim_w=3,         # Action dimension
            n_latent_dims=16,
            vqvae_n_embed=8,
            vqvae_groups=2,
            eval=False,
            device='cuda',
        )
        # learnable token,汇聚所有点的上下文信息

    def forward(self, cloud, actions = None, batch_size = 24):
        src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size=batch_size)
        # padding_mask: 排除无效点的干扰
        readout = self.transformer(src, src_padding_mask, self.readout_embed.weight, pos)[-1]
        readout = readout[:, 0]
        if actions is not None: # training mode
            loss = self.action_decoder.compute_loss(readout, actions)
            return loss
        else:
            with torch.no_grad(): # prediction mode
                action_pred = self.action_decoder.predict_action(readout)
            return action_pred