#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 The OpenRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""""""

import torch
import torch.nn as nn

import numpy as np

from openrl.buffers.utils.util import get_policy_obs_space
from openrl.buffers.utils.util import get_critic_obs_space
from openrl.modules.networks.base_value_network import BaseValueNetwork
from openrl.modules.networks.utils.cnn import CNNBase
from openrl.modules.networks.utils.mix import MIXBase
from openrl.modules.networks.utils.mlp import MLPBase, MLPLayer
from openrl.modules.networks.utils.popart import PopArt
from openrl.modules.networks.utils.rnn import RNNLayer
from openrl.modules.networks.utils.util import init
from openrl.utils.util import check_v2 as check


class EncoderNetwork(BaseValueNetwork):
    def __init__(
        self,
        cfg,
        input_space,
        action_space=None,
        use_half=False,
        device=torch.device("cpu"),
        extra_args=None,
    ):
        super(EncoderNetwork, self).__init__(cfg, device)

        self.latent_dim = 16

        self.hidden_size = cfg.hidden_size
        self._use_orthogonal = cfg.use_orthogonal
        self._activation_id = cfg.activation_id
        self._use_naive_recurrent_policy = cfg.use_naive_recurrent_policy
        self._use_recurrent_policy = cfg.use_recurrent_policy
        self._use_influence_policy = cfg.use_influence_policy
        self._use_popart = cfg.use_popart
        self._influence_layer_N = cfg.influence_layer_N
        self._recurrent_N = cfg.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]

        self.act_space = get_policy_obs_space(input_space)
        critic_obs_shape = get_critic_obs_space(input_space)

        if "Dict" in critic_obs_shape.__class__.__name__:
            self._mixed_obs = True
            self.base = MIXBase(
                cfg, critic_obs_shape, cnn_layers_params=cfg.cnn_layers_params
            )
        else:
            self._mixed_obs = False
            self.base = (
                CNNBase(cfg, critic_obs_shape)
                if len(critic_obs_shape) == 3
                else MLPBase(
                    cfg,
                    critic_obs_shape,
                    use_attn_internal=True,
                    use_cat_self=cfg.use_cat_self,
                )
            )

        input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                input_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
                rnn_type=cfg.rnn_type,
            )
            input_size = self.hidden_size

        if self._use_influence_policy:
            self.mlp = MLPLayer(
                critic_obs_shape[0],
                self.hidden_size,
                self._influence_layer_N,
                self._use_orthogonal,
                self._activation_id,
            )
            input_size += self.hidden_size

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.mu_out = init_(nn.Linear(input_size, self.latent_dim))
        self.logvar_out = init_(nn.Linear(input_size, self.latent_dim))

        self.to(device)

    def forward(self, global_state, actions_obs, rnn_states, rnn_masks, episode_start_idx, action_masks):
        # actions: [batch_size, 1]
        # global_state: [batch_size, sta_size]
        # rnn_masks: [batch_size]
        if self._mixed_obs:
            raise NotImplementedError
            # for key in encoder_obs.keys():
            #     encoder_obs[key] = check(encoder_obs[key]).to(**self.tpdv)
        else:
            actions_obs[episode_start_idx] = 0
            actions_obs = np.eye(self.act_space[0])[actions_obs[:,0]]
            actions_obs[episode_start_idx] = 0
            actions_obs = check(actions_obs).to(**self.tpdv)
            global_state = check(global_state).to(**self.tpdv)
            encoder_obs = torch.cat([actions_obs, global_state], -1)

        rnn_states = check(rnn_states).to(**self.tpdv)
        rnn_masks = check(rnn_masks).to(**self.tpdv)

        encoder_features = self.base(encoder_obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            encoder_features, rnn_states = self.rnn(encoder_features, rnn_states, rnn_masks)

        if self._use_influence_policy:
            mlp_critic_obs = self.mlp(encoder_obs)
            encoder_features = torch.cat([encoder_features, mlp_critic_obs], dim=1)

        mu = self.mu_out(encoder_features)
        logvar = self.logvar_out(encoder_features)

        return mu, logvar, rnn_states
