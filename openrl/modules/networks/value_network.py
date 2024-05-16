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
import numpy as np

import torch
import torch.nn as nn

from openrl.buffers.utils.util import get_critic_obs_space
from openrl.modules.networks.base_value_network import BaseValueNetwork
from openrl.modules.networks.utils.cnn import CNNBase
from openrl.modules.networks.utils.mix import MIXBase
from openrl.modules.networks.utils.mlp import MLPBase, MLPLayer
from openrl.modules.networks.utils.popart import PopArt
from openrl.modules.networks.utils.rnn import RNNLayer
from openrl.modules.networks.utils.util import init
from openrl.utils.util import check_v2 as check


available_actions = [
    # "Find cows.", 
    # "Find water.", 
    # "Find stone.", 
    # "Find tree.",
    "Collect sapling.",
    "Place sapling.",
    "Chop tree.", 
    "Kill the cow.", 
    "Mine stone.", 
    "Drink water.",
    "Mine coal.", 
    "Mine iron.", 
    "Mine diamond.", 
    "Kill the zombie.",
    "Kill the skeleton.", 
    "Craft wood_pickaxe.", 
    "Craft wood_sword.",
    "Place crafting table.", 
    "Place furnace.", 
    "Craft stone_pickaxe.",
    "Craft stone_sword.", 
    "Craft iron_pickaxe.", 
    "Craft iron_sword.",
    "Sleep."
]

class ValueNetwork(BaseValueNetwork):
    def __init__(
        self,
        cfg,
        input_space,
        action_space=None,
        use_half=False,
        device=torch.device("cpu"),
        extra_args=None,
    ):
        super(ValueNetwork, self).__init__(cfg, device)

        self.hidden_size = cfg.hidden_size
        self._use_orthogonal = cfg.use_orthogonal
        self._activation_id = cfg.activation_id
        self._use_naive_recurrent_policy = cfg.use_naive_recurrent_policy
        self._use_recurrent_policy = cfg.use_recurrent_policy
        self._use_influence_policy = cfg.use_influence_policy
        self._use_popart = cfg.use_popart
        self._use_fp16 = cfg.use_fp16 and cfg.use_deepspeed
        self._influence_layer_N = cfg.influence_layer_N
        self._recurrent_N = cfg.recurrent_N
        self._alpha = cfg.alpha_value
        self.tpdv = dict(dtype=torch.float32, device=device)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]

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

        if self._use_popart:
            self.v_out = init_(PopArt(input_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(input_size, 1))

        self.to(device)

    def forward(self, critic_obs, rnn_states, masks, actions=None, get_value=False):
        
        given_actions = actions is not None
        auto_actions = get_value
        assert given_actions ^ auto_actions
            
        if self._mixed_obs:
            for key in critic_obs.keys():
                critic_obs[key] = check(critic_obs[key]).to(**self.tpdv)
        else:
            critic_obs = check(critic_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if not isinstance(actions, torch.Tensor) and actions is not None:
            actions = check(actions).to(**self.tpdv).long()

        if self._use_fp16:
            critic_obs = critic_obs.half()
        
        # task selector
        task_feature = self.base(critic_obs, cnn_only=True)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            task_feature, task_rnn_states = self.task_actor(task_feature, rnn_states[:,1:,:], masks)
        task_logits = self.act2.action_out(task_feature, None)
        task_qvals = task_logits.logits
        
        # update task_emb in observations
        if actions is not None:
            task_idxs = actions[:,1:].flatten().cpu().numpy()
            task_name = np.array(available_actions)[task_idxs]
            task_emb = self.bert(task_name, convert_to_numpy=False)
            critic_obs["task_emb"] = torch.stack(task_emb, dim=0)
            tasks = actions[:,1:].clone()
        elif get_value:
            tasks = task_logits.sample()
            # we don't care about actions in this case, so we don't need to update task_emb

        task_qvals = torch.gather(task_qvals, 1, tasks)
        
        if get_value:
            task_logits = self.act2.action_out(task_feature, alpha = self._alpha)
            task_logp = task_logits.log_probs(tasks)
            task_qvals -= task_logp
        
        critic_features = self.base(critic_obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states[:,:1,:], masks)

        if self._use_influence_policy:
            mlp_critic_obs = self.mlp(critic_obs)
            critic_features = torch.cat([critic_features, mlp_critic_obs], dim=1)

        values = self.v_out(critic_features)
        
        values = torch.cat([values, task_qvals], dim=1)
        rnn_states = torch.cat([rnn_states, task_rnn_states], dim=1)

        return values, rnn_states
