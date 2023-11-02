#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 The OpenRL Authors.
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

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from openrl.envs.smac.smacv2_env.wrapper import StarCraftCapabilityEnvWrapper


class SMACEnv(gym.Env):
    env_name = "SMAC"

    def __init__(self, cfg, env_id, is_eval):
        

        distribution_config = {
            "n_units": 5,
            "n_enemies": 5,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": ["stalker", "zealot", "colossus"],
                "weights": [0.45, 0.45, 0.1],
                "observe": True,
                "exception_unit_types": ["colossus"],
            },
            
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "map_x": 32,
                "map_y": 32,
            },
        }

        self.env = StarCraftCapabilityEnvWrapper(
            is_eval=is_eval,
            env_id=env_id,
            capability_config=distribution_config,
            map_name=cfg.map_name,
            debug=False,
            conic_fov=False,
            use_unit_ranges=True,
            min_attack_range=2,
            obs_own_pos=True,
            fully_observable=False,
        )

        policy_obs_dim = self.env.get_obs_size()

        policy_obs = spaces.Box(
            low=-np.inf, high=+np.inf, shape=(policy_obs_dim,), dtype=np.float32
        )

        critic_obs_dim = self.env.get_state_size()

        critic_obs = spaces.Box(
            low=-np.inf, high=+np.inf, shape=(critic_obs_dim,), dtype=np.float32
        )

        self.agent_num = self.env.n_agents
        self.observation_space = gym.spaces.Dict(
            {
                "policy": policy_obs,
                "critic": critic_obs,
            }
        )

        self.action_space = spaces.Discrete(self.env.get_total_actions())

    def reset(self, seed=None, **kwargs):

        local_obs, global_state = self.env.reset()
        global_state = [global_state for _ in range(self.agent_num)]
        action_mask = self.env.get_avail_actions()

        return {"policy": local_obs, "critic": global_state}, {"action_masks": action_mask}

    def step(self, action, extra_data):
        rewards, terminated, infos = self.env.step(action, extra_data)
        rewards = [[rewards] for _ in range(self.agent_num)] 
        dones = [terminated for _ in range(self.agent_num)]
        
        local_obs = self.env.get_obs()
        global_state = self.env.get_state()
        global_state = [global_state for _ in range(self.agent_num)]
        action_mask = self.env.get_avail_actions()
        infos.update({
            "action_masks": action_mask,
        })
        
        return (
            {"policy": local_obs, "critic": global_state},
            rewards,
            dones,
            infos,
        )

    def close(self):
        self.env.close()
