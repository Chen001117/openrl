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

import numpy as np

ArrayType = TypeVar("ArrayType")

class TextWrapper(BaseWrapper):
    
    def __init__(self, env, cfg=None, reward_class=None) -> None:
        
        super().__init__(env, cfg, reward_class)
        self.env = env
        self.cfg = cfg
        self.reward_class = reward_class
        
        self.update_task_freq = 4
        
        self._cur_task = "Survive."
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.env_step = 0
        obs, info = self.env.reset(seed, options)
        text_obs = self._get_text_obs()
        next_task = self.get_next_task(text_obs)
        info.update({
            "text_obs": text_obs, 
            "step": self.env_step,
            "next_task": next_task
        })
        self.obj_hist = [self.get_obj_dict() for _ in range(self.update_task_freq)]
        
        self._cur_task = next_task
        
        return obs, info
        
    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, Dict[str, Any]]:
        self.env_step += 1
        obs, reward, done, truncated, _ = self.env.step(action)
        self.obj_hist.append(self.get_obj_dict())
        last_obs = self.obj_hist.pop(0)
        obj_diff = self.get_obj_diff(last_obs, self.obj_hist[-1])
        text_obs = self._get_text_obs()
        next_task = self.get_next_task(text_obs)
        info = {
            "text_obs": text_obs, 
            "step": self.env_step, 
            "obj_diff": obj_diff, 
            "next_task": next_task
        }
        
        self._cur_task = next_task
        
        return obs, reward, done, truncated, info

    def get_next_task(self, text_obs):
        """
         You see cow, grass, and tree. 
        You have nothing in your inventory. 
        Your health level is high, food level is high, drink level is high, energy is high.
        """
        
        split_text_obs = text_obs.split("You")[1:]
        
        available_actions = [
            "Find cows.", "Find water.", "Find stone.", "Find tree.",
            "Chop tree.", "Kill the cow.", "Mine stone.", "Drink water.",
            "Mine coal.", "Mine iron.", "Mine diamond.", "Kill the zombie.",
            "Kill the skeleton.", "Craft wood_pickaxe.", "Craft wood_sword.",
            "Place crafting table.", "Place furnace.", "Craft stone_pickaxe.",
            "Craft stone_sword.", "Craft iron_pickaxe.", "Craft iron_sword.",
            "Sleep."
        ]
        actions_probw = [
            0.01, 0.01, 0.01, 0.01,
            0.0001, 0.0001, 0.0001, 0.0001,
            0.0001, 0.0001, 0.0001, 0.0001,
            0.0001, 0.0001, 0.0001, 0.0001,
            0.0001, 0.0001, 0.0001, 0.0001,
            0.0001, 0.0001
        ]
        
        if "tree" in split_text_obs[0]:
            available_actions.append("Chop tree.")
            actions_probw.append(.1)
        if "cow" in split_text_obs[0]:
            available_actions.append("Kill the cow.")
            actions_probw.append(.5)
        if "stone" in split_text_obs[0]:
            if "pickaxe" in split_text_obs[1]:
                available_actions.append("Mine stone.")
                actions_probw.append(2.)
        if "water" in split_text_obs[0]:
            available_actions.append("Drink water.")
            actions_probw.append(.25)
        if "coal" in split_text_obs[0]:
            if "wood pickaxe" in split_text_obs[1]:
                available_actions.append("Mine coal.")
                actions_probw.append(1.)
        if "iron" in split_text_obs[0]:
            if "stone_pickaxe" in split_text_obs[1]:
                available_actions.append("Mine iron.")
                actions_probw.append(1.)
        if "diamond" in split_text_obs[0]:
            if "iron pickaxe" in split_text_obs[1]:
                available_actions.append("Mine diamond.")
                actions_probw.append(1.)
        if "zombie" in split_text_obs[0]:
            if "sword" in split_text_obs[1]:
                available_actions.append("Kill the zombie.")
                actions_probw.append(1.)
            else:
                available_actions.append("Kill the zombie.")
                actions_probw.append(.5)
        if "skeleton" in split_text_obs[0]:
            if "sword" in split_text_obs[1]:
                available_actions.append("Kill the skeleton.")
                actions_probw.append(.75)
            else:
                available_actions.append("Kill the skeleton.")
                actions_probw.append(.25)
        if "wood," in split_text_obs[1]:
            if "crafting table" in split_text_obs[0]:
                available_actions.append("Craft wood_pickaxe.")
                actions_probw.append(1.)
                available_actions.append("Craft wood_sword.")
                actions_probw.append(1.)
            else:
                available_actions.append("Place crafting table.")
                actions_probw.append(1.)
        if "stone," in split_text_obs[1]:
            if "crafting table" in split_text_obs[0]:
                available_actions.append("Craft stone_pickaxe.")
                actions_probw.append(3.)
                available_actions.append("Craft stone_sword.")
                actions_probw.append(3.)
            available_actions.append("Place furnace.")
            actions_probw.append(2.)
        if "iron," in split_text_obs[1]:
            if "frunace" in split_text_obs[0]:
                available_actions.append("Craft iron_pickaxe.")
                actions_probw.append(5.)
                available_actions.append("Craft iron_sword.")
                actions_probw.append(5.)
        if "food level is low" in split_text_obs[2]:
            if "cow" in split_text_obs[0]:
                available_actions.append("Kill the cow.")
                actions_probw.append(2.)
            else:
                available_actions.append("Find cows.")
                actions_probw.append(1.)
        if "drink level is low" in split_text_obs[2]:
            if "water" in split_text_obs[0]:
                available_actions.append("Drink water.")
                actions_probw.append(.5)
            else:
                available_actions.append("Find water.")
                actions_probw.append(1.)
        if "energy is low" in split_text_obs[2]:
            available_actions.append("Sleep.")
            actions_probw.append(2.)
            
        actions_probw = np.array(actions_probw)
        actions_probw = actions_probw / np.sum(actions_probw)
        
        if len(available_actions) > 0:
            return np.random.choice(available_actions, p=actions_probw)
        else:
            return "Chop tree." 
    
    def get_obj_dict(self):
        
        obj_dict = dict()
        
        for k, v in self.env.player.inventory.items():
            k = "plant" if k == "sapling" else k
            obj_dict["inv-" + k] = v
        
        env_obj = self.env.text_view.local_obj(self.env.player)
        # print("env_obj: ", env_obj)
        for k, v in env_obj.items():
            obj_dict["env-" + k] = v
        
        return obj_dict
    
    def get_obj_diff(self, last_obj, cur_obj):
        output_string = "you "
        STATUS_ITEMS = ['inv-health', 'inv-food', 'inv-drink', 'inv-energy']
        for k, v in last_obj.items():
            if k in cur_obj:
                diff = cur_obj[k] - v
                if k in STATUS_ITEMS and diff < -1:
                    output_string += ("loss " + str(-diff) + " " + k[4:] + ", ")
                elif k in STATUS_ITEMS and diff > 0:
                    output_string += ("gain " + str(diff) + " " + k[4:] + ", ")
                elif "inv-" in k and k not in STATUS_ITEMS and diff > 0:
                    output_string += ("get " + str(diff) + " " + k[4:] + ", ")
                elif "inv-" in k and k not in STATUS_ITEMS and diff < 0:
                    output_string += ("use " + str(-diff) + " " + k[4:] + ", ")
                elif "env-" in k and diff > 0:
                    if k.lower() in ["env-water", "env-cow"]:
                        output_string += ("find " + k[4:] + ", ")
                    
        for k, v in cur_obj.items():
            if k not in last_obj:
                output_string += ("find " + k[4:] + ", ")
                
        tmp = output_string
        
        output_string = output_string[:-2] + "." if output_string[-2:] == ", " else output_string
        output_string = "" if output_string == "you " else output_string
        
        for k, v in last_obj.items():
            if k in cur_obj and k not in ["env-water"]:
                diff = cur_obj[k] - v
                if "env-" in k and diff < 0:
                    output_string += (" " + str(-diff) + " " + k[4:] + " disappear. ")
            else:
                output_string += (" " + str(v) + " " + k[4:] + " disappear. ")
                
        for k, v in last_obj.items():
            if k == "inv-food":
                diff = cur_obj[k] - v
                if diff == 0 and "cow disappear" in output_string:
                    output_string += " Your food level is unchanged."
            if k == "inv-drink":
                diff = cur_obj[k] - v
                if diff == 0 and "water disappear" in output_string:
                    output_string += " Your drink level is unchanged."
        
        if self.env.player.sleeping:
            output_string += " You are sleeping."
    
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
        
        