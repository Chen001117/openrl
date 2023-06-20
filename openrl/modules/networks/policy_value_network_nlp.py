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
from typing import Any

import torch

from openrl.modules.networks.utils.nlp.gpt_causal_lm import GPTCausalLM
from openrl.modules.utils.valuenorm import ValueNorm

from openrl.utils.util import check_v2 as check

from openrl.modules.utils.util import get_optimizer_grouped_parameters

class PolicyValueNetworkNLP(GPTCausalLM):
    def __init__(
        self,
        cfg: Any,
        input_space,
        action_space,
        device=torch.device("cpu"),
        use_half=False,
    ):
        self.use_half = use_half
        self._disable_drop_out = cfg.disable_drop_out
        self._use_valuenorm = cfg.use_valuenorm
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.use_deepspeed = cfg.use_deepspeed
        
        super(PolicyValueNetworkNLP, self).__init__(
            input_space, action_space, cfg.model_path
        )

        self._use_valuenorm = cfg.use_valuenorm
        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=device)
        else:
            self.value_normalizer = None


    def forward(self, forward_type, *args, **kwargs):
        if forward_type == "original":
            return self.get_actions(*args, **kwargs)
        elif forward_type == "eval_actions":
            return self.eval_actions(*args, **kwargs)
        elif forward_type == "get_values":
            return self.get_values(*args, **kwargs)
        else:
            raise NotImplementedError
        
    def set_engine(self, actor_engine, critic_engine, critic, *args, **kwargs):
        if actor_engine:
            self.actor_engine = actor_engine
        if critic_engine:
            self.critic_engine = critic_engine
        if critic:
            self.critic = critic

    def get_actor_para(self):
        return self.policy_model.parameters()

    def get_critic_para(self):
        return self.critic.parameters()

    def get_actions(
        self, obs, rnn_states, masks, available_actions=None, deterministic=False
    ):
        for key in obs.keys():
            obs[key] = check(obs[key], self.use_half, self.tpdv)
        rnn_states = check(rnn_states, self.use_half, self.tpdv)

        policy_output = super().get_distribution(obs)
        actions = policy_output.mode() if deterministic else policy_output.sample()
        action_log_probs = policy_output.log_prob(actions)

        actions = actions.unsqueeze(-1)
        action_log_probs = action_log_probs.unsqueeze(-1)

        return actions, action_log_probs, rnn_states
        # TODO: add past_model_kwargs, i.e., past key value.

    def eval_actions(
        self, obs, rnn_states, action, masks, available_actions, active_masks=None
    ):
        for key in obs.keys():
            obs[key] = check(obs[key], self.use_half, self.tpdv)
        action = check(action, self.use_half, self.tpdv)
        rnn_states = check(rnn_states, self.use_half, self.tpdv)
        
        log_probs, dist_entropy, values = super().evaluate_actions(obs, action)
        
        action_log_probs = log_probs.unsqueeze(-1)
        dist_entropy = dist_entropy.mean()

        return action_log_probs, dist_entropy, values

    def get_values(self, obs, rnn_states, masks):
        for key in obs.keys():
            obs[key] = check(obs[key], self.use_half, self.tpdv)
        rnn_states = check(rnn_states, self.use_half, self.tpdv)

        values = super().forward_value(obs)

        return values, rnn_states
