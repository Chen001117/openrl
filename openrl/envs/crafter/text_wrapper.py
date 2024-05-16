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
from typing import Any, Dict, Optional, SupportsFloat, Tuple, TypeVar, Union

import gymnasium as gym
from gymnasium.core import ActType, ObsType, WrapperObsType

from openrl.envs.wrappers.base_wrapper import BaseWrapper
from openrl.envs.crafter.captioner import Captioner

import copy 
import numpy as np

ArrayType = TypeVar("ArrayType")

class TextWrapper(BaseWrapper):
    
    def __init__(self, env, cfg=None, reward_class=None) -> None:
        
        super().__init__(env, cfg, reward_class)
        self.env = env
        self.cfg = cfg
        self.reward_class = reward_class
        
        self.captioner = Captioner(env)
        
        idx_x, idx_y = np.meshgrid(np.arange(7), np.arange(9))
        self.dist_map = (idx_x - 3) ** 2 + (idx_y - 4) ** 2
        self.dist_idx = np.argsort(self.dist_map.flatten())[1:]
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        
        self.env_step = 0
        self.total_rew = 0
        
        obs, info = self.env.reset(seed, options)
        text_obs, dict_obs = self.captioner(reset=True)
        info.update({
            "text_obs": text_obs, 
            "dict_obs": dict_obs,
            "step": self.env_step,
            "episode": {"r": self.total_rew, "l": self.env_step}
        })
        
        self.last_privileged_info = self._get_privileged_info()
        
        return obs, info
        
    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, Dict[str, Any]]:
        
        obs, reward, done, truncated, _ = self.env.step(action)
        
        self.env_step += 1
        self.total_rew += reward
        
        text_obs, dict_obs = self.captioner()
        
        current_privileged_info = self._get_privileged_info()
        diff = self._get_diff(self.last_privileged_info, current_privileged_info)
        self.last_privileged_info = current_privileged_info
        
        info = {
            "text_obs": text_obs, 
            "dict_obs": dict_obs,
            "step": self.env_step,
            "obj_diff": diff,
            "episode": {"r": self.total_rew, "l": self.env_step}
        }
        
        return obs, reward, done, truncated, info

    # get difference between two frames (for rewards)
    def _get_diff(self, last_privileged_info, current_privileged_info):
        """
        return a string describing the difference between the last and current privileged information
        """
        
        diff = ""
        for k, v in last_privileged_info["inventory"].items():
            if current_privileged_info["inventory"][k] > v:
                diff += f"You have gained {current_privileged_info['inventory'][k] - v} {k}. "
            elif current_privileged_info["inventory"][k] < v:
                diff += f"You have lost {v - current_privileged_info['inventory'][k]} {k}. "
        
        entities = {"cow":14, "zombie":15, "skeleton":16} 
        for entity_name, entity_value in entities.items():
            last_map = last_privileged_info["env_map"] == entity_value
            current_map = current_privileged_info["env_map"] == entity_value
            for row in range(1, len(last_map) - 1):
                for col in range(1, len(last_map[0]) - 1):
                    if last_map[row, col] and not current_map[row, col]:
                        entity_move = False
                        if current_map[row-1,col] == True:
                            current_map[row-1,col] = False
                            entity_move = True
                        elif current_map[row+1,col] == True:
                            current_map[row+1,col] = False
                            entity_move = True
                        elif current_map[row,col-1] == True:
                            current_map[row,col-1] = False
                            entity_move = True
                        elif current_map[row,col+1] == True:
                            current_map[row,col+1] = False
                            entity_move = True
                        if not entity_move and self.dist_map[row, col] < 4:
                            diff += f"You have killed a {entity_name}. "
        
        # find something new in the environment
        
        env_item = {
            1: 'water', 3: 'stone', 6: 'tree', 8: 'coal', 
            9: 'iron', 10: 'diamond', 11: 'crafting table', 
            12: 'furnace', 14: 'cow', 18: 'plant'
        }
        for k, v in env_item.items():
            last_map_sum = np.sum(last_privileged_info["env_map"] == k)
            current_map_sum = np.sum(current_privileged_info["env_map"] == k)
            if last_map_sum == 0 and current_map_sum > 0:
                diff += f"You have found {v}. "
                
        if not last_privileged_info["is_sleeping"] and current_privileged_info["is_sleeping"]:
            diff += "You fell asleep. "
            
        if diff == "":
            diff = "Nothing has changed. "
        
        return diff
    
    def _get_privileged_info(self):
        """
        return a dictionary containing the player's privileged information
        """
        
        inventory = copy.deepcopy(self.env.player.inventory)
        env_map = copy.deepcopy(self.env._text_view.get_map(self.env._player.pos))
        
        return {"inventory": inventory, "env_map": env_map, "is_sleeping": self.env._player.sleeping}

        