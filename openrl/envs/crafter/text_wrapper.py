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

ArrayType = TypeVar("ArrayType")

class TextWrapper(BaseWrapper):
    
    def __init__(self, env, cfg=None, reward_class=None) -> None:
        
        super().__init__(env, cfg, reward_class)
        self.env = env
        self.cfg = cfg
        self.reward_class = reward_class
        
        self.period = 32
        self.period_small = 4
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.env_step = 0
        obs, info = self.env.reset(seed, options)
        text_obs = self._get_text_obs()
        self.text_obj = self.get_obj_dict()
        info.update({"text_obs": text_obs, "step": self.env_step})
        self.obj_hist = [self.get_obj_dict() for _ in range(self.period)]
        return obs, info
        
    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, Dict[str, Any]]:
        self.env_step += 1
        obs, reward, done, truncated, _ = self.env.step(action)
        self.obj_hist.append(self.get_obj_dict())
        last_obs = self.obj_hist.pop(0)
        obj_diff = ""
        for i in range(0, self.period-1, self.period_small):
            idx = min(self.period-1, i+4)
            obj_diff += self.get_obj_diff(self.obj_hist[i], self.obj_hist[idx])
        text_obs = self._get_text_obs()
        info = {"text_obs": text_obs, "step": self.env_step, "obj_diff": obj_diff}
        return obs, reward, done, truncated, info
    
    def get_obj_dict(self):
        
        obj_dict = dict()
        
        for k, v in self.env.player.inventory.items():
            k = "plant" if k == "sapling" else k
            obj_dict["inv-" + k] = v
        
        env_obj = self.env.text_view.local_obj(self.env.player)
        for k, v in env_obj.items():
            obj_dict["env-" + k] = v
        
        return obj_dict
    
    def get_obj_diff(self, last_obj, cur_obj):
        output_string = ""
        STATUS_ITEMS = ['inv-health', 'inv-food', 'inv-drink', 'inv-energy']
        for k, v in last_obj.items():
            if k in cur_obj:
                diff = cur_obj[k] - v
                if k in STATUS_ITEMS and diff < -1:
                    output_string += ("loss " + str(-diff) + " " + k[4:] + ". ")
                elif k in STATUS_ITEMS and diff > 0:
                    output_string += ("gain " + str(diff) + " " + k[4:] + ". ")
                elif "inv" in k and k not in STATUS_ITEMS and diff > 0:
                    output_string += ("get " + str(diff) + " " + k[4:] + ". ")
                elif "inv" in k and k not in STATUS_ITEMS and diff < 0:
                    output_string += ("use " + str(-diff) + " " + k[4:] + ". ")
                    
        return output_string
        
    def _get_text_obs(self):
        """
        Returns a list of strings for the inventory, and list of strings for player status.
        else returns a sentence formed from the inventory lists: "You have axe, wood..", and status lists "You feel hungry, sleepy..."
        """
        
        STATUS_ITEMS = ['health', 'food', 'drink', 'energy']
        
        inner_state = "Your "
        
        # the first 4 items in the inventory are the player's status
        for k, v in self.env.player.inventory.items():
            if k in STATUS_ITEMS:
                status = "low" if v <= 3 else "high"
                if k == "energy":
                    inner_state += f"{k} is " + status
                    inner_state += "." 
                else:
                    inner_state += f"{k} level is " + status
                    inner_state += ", "

        inventory = "You have in your inventory: "
        empty_inventory = True
        for k, v in self.env.player.inventory.items():
            if k not in STATUS_ITEMS and v > 0:
                k = "plant" if k == "sapling" else k
                inventory += f"{v} {k}, "
                empty_inventory = False
        if empty_inventory:
            inventory = "You have nothing in your inventory."
            
        surrounding_state = self.env.text_view.local_sentence_view(self.env.player)

        text_state = surrounding_state + " " + inventory + " " + inner_state
        
        if self.env.player.sleeping:
            text_state += " You are sleeping."

        return text_state  
        
        