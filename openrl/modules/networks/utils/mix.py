import numpy as np
import torch
import torch.nn as nn

from .util import init

from openrl.modules.networks.utils.cnn import CNNLayer
from openrl.modules.networks.utils.mlp import MLPLayer

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class MIXBase(nn.Module):
    def __init__(self, cfg, obs_shape, cnn_layers_params=None):
        super(MIXBase, self).__init__()

        self._use_orthogonal = cfg.use_orthogonal
        self._activation_id = cfg.activation_id
        self._use_maxpool2d = cfg.use_maxpool2d
        self.hidden_size = cfg.hidden_size
        self.cnn_keys = []
        self.embed_keys = []
        self.mlp_keys = []
        self.n_cnn_input = 0
        self.n_embed_input = 0
        self.n_mlp_input = 0

        for key in obs_shape:
            if obs_shape[key].__class__.__name__ == "Box":
                key_obs_shape = obs_shape[key].shape
                if len(key_obs_shape) == 3:
                    self.cnn_keys.append(key)
                else:
                    self.mlp_keys.append(key)
            else:
                raise NotImplementedError

        if len(self.cnn_keys) > 0:
            self.cnn = CNNLayer(
                obs_shape["image"].shape, self.hidden_size, self._use_orthogonal, self._activation_id
            )
        if len(self.mlp_keys) > 0:
            layer_N = 0
            self.mlp = MLPLayer(
                obs_shape["task_emb"].shape[0], self.hidden_size, layer_N, self._use_orthogonal, self._activation_id,
            )
            
        self.out_layer = nn.Linear(self.hidden_size*2, self.hidden_size)
        with torch.no_grad():
            self.out_layer.weight.data[:,self.hidden_size:] *= 0
            self.out_layer.weight.data[:,:self.hidden_size] = torch.eye(self.hidden_size)
            self.out_layer.bias.data *= 0

    def forward(self, x):
        
        out_x = None
        if len(self.cnn_keys) > 0:
            cnn_input = self._build_cnn_input(x)
            out_x = self.cnn(cnn_input)

        if len(self.mlp_keys) > 0:
            mlp_input = self._build_mlp_input(x)
            mlp_x = self.mlp(mlp_input).view(mlp_input.size(0), -1)

            if out_x is not None:
                out_x = torch.cat([out_x, mlp_x], dim=1)  # ! wrong
            else:
                out_x = mlp_x
        
        out_x = self.out_layer(out_x)
        
        return out_x

    def _build_cnn_input(self, obs):
        cnn_input = []
        for key in self.cnn_keys:
            cnn_input.append(obs[key])
        cnn_input = torch.cat(cnn_input, dim=1)
        return cnn_input

    def _build_mlp_input(self, obs):
        mlp_input = []
        for key in self.mlp_keys:
            mlp_input.append(obs[key].view(obs[key].size(0), -1))

        mlp_input = torch.cat(mlp_input, dim=1)
        return mlp_input

    @property
    def output_size(self):
        return self.hidden_size
