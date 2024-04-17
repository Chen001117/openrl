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

import copy 
import numpy as np

ArrayType = TypeVar("ArrayType")

class TextWrapper(BaseWrapper):
    
    def __init__(self, env, cfg=None, reward_class=None) -> None:
        
        super().__init__(env, cfg, reward_class)
        self.env = env
        self.cfg = cfg
        self.reward_class = reward_class
        
        self._cur_task = "Survive."
        
        self.names = {value : key for (key, value) in self.env._world._mat_ids.items()}
        self.names[11] = "crafting_table"
        self.names[13] = "player"
        self.names[14] = "cow"
        self.names[15] = "zombie"
        self.names[16] = "skeleton"
        self.names[17] = "arrow"
        self.names[18] = "plant"
        
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
        obs, info = self.env.reset(seed, options)
        text_obs = self._get_text_state()
        next_task = self.get_next_task(text_obs)
        action_masks = np.ones(self.env.action_space.n)
        info.update({
            "text_obs": text_obs, 
            "step": self.env_step,
            "next_task": next_task,
            "action_masks": action_masks,
        })
        
        self.last_privileged_info = self._get_privileged_info()
        self._cur_task = next_task
        
        return obs, info
        
    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, Dict[str, Any]]:
        
        self.env_step += 1
        obs, reward, done, truncated, _ = self.env.step(action)
        
        text_obs = self._get_text_state()
        next_task = self.get_next_task(text_obs)
        
        current_privileged_info = self._get_privileged_info()
        diff = self._get_diff(self.last_privileged_info, current_privileged_info)
        self.last_privileged_info = current_privileged_info
        
        action_masks = np.ones(self.env.action_space.n)
        if "gained 1 stone" in diff:
            action_masks[7] = 0.
        # if action == 7 and "stone" in self._get_inventory():
        #     action_masks[5] = 0.
            
        info = {
            "text_obs": text_obs, 
            "step": self.env_step, 
            "next_task": next_task,
            "obj_diff": diff, 
            "action_masks": action_masks,
        }
        
        self._cur_task = next_task
        
        return obs, reward, done, truncated, info

    def get_cur_task(self, obj_diff):
        
        possible_tasks = []
        
        if "gain 1 food, " in obj_diff:
            possible_tasks.append("Kill the cow.")
        if "gain 2 food, " in obj_diff:
            possible_tasks.append("Kill the cow.")
        if "gain 3 food, " in obj_diff:
            possible_tasks.append("Kill the cow.")
        if "gain 4 food, " in obj_diff:
            possible_tasks.append("Kill the cow.")
        if "gain 1 drink, " in obj_diff:
            possible_tasks.append("Drink water.")
        if "gain 2 drink, " in obj_diff:
            possible_tasks.append("Drink water.")
        if "gain 3 drink, " in obj_diff:
            possible_tasks.append("Drink water.")
        if "gain 4 drink, " in obj_diff:
            possible_tasks.append("Drink water.")
        if "gain 1 energy, " in obj_diff:
            possible_tasks.append("Sleep.")
        if "gain 2 energy, " in obj_diff:
            possible_tasks.append("Sleep.")
        if "gain 3 energy, " in obj_diff:
            possible_tasks.append("Sleep.")
        if "gain 4 energy, " in obj_diff:
            possible_tasks.append("Sleep.")
        if "get 1 wood, " in obj_diff:
            possible_tasks.append("Chop tree.")
        if "get 2 wood, " in obj_diff:
            possible_tasks.append("Chop tree.")
        if "get 3 wood, " in obj_diff:
            possible_tasks.append("Chop tree.")
        if "get 4 wood, " in obj_diff:
            possible_tasks.append("Chop tree.")
        if "get 1 stone, " in obj_diff:
            possible_tasks.append("Mine stone.")
        if "get 2 stone, " in obj_diff:
            possible_tasks.append("Mine stone.")
        if "get 3 stone, " in obj_diff:
            possible_tasks.append("Mine stone.")
        if "get 4 stone, " in obj_diff:
            possible_tasks.append("Mine stone.")
        if "get 1 coal, " in obj_diff:
            possible_tasks.append("Mine coal.")
        if "get 2 coal, " in obj_diff:
            possible_tasks.append("Mine coal.")
        if "get 3 coal, " in obj_diff:
            possible_tasks.append("Mine coal.")
        if "get 4 coal, " in obj_diff:
            possible_tasks.append("Mine coal.")
        if "get 1 iron, " in obj_diff:
            possible_tasks.append("Mine iron.")
        if "get 2 iron, " in obj_diff:
            possible_tasks.append("Mine iron.")
        if "get 3 iron, " in obj_diff:
            possible_tasks.append("Mine iron.")
        if "get 4 iron, " in obj_diff:
            possible_tasks.append("Mine iron.")
        if "get 1 diamond, " in obj_diff:
            possible_tasks.append("Mine diamond.")
        if "get 2 diamond, " in obj_diff:
            possible_tasks.append("Mine diamond.")
        if "get 3 diamond, " in obj_diff:
            possible_tasks.append("Mine diamond.")
        if "get 4 diamond, " in obj_diff:
            possible_tasks.append("Mine diamond.")
        if "zombie disappear" in obj_diff:
            possible_tasks.append("Kill the zombie.")
        if "skeleton disappear" in obj_diff:
            possible_tasks.append("Kill the skeleton.")
        if "get 1 wood_pickaxe, " in obj_diff:
            possible_tasks.append("Craft wood_pickaxe.")
        if "get 2 wood_pickaxe, " in obj_diff:
            possible_tasks.append("Craft wood_pickaxe.")
        if "get 3 wood_pickaxe, " in obj_diff:
            possible_tasks.append("Craft wood_pickaxe.")
        if "get 4 wood_pickaxe, " in obj_diff:
            possible_tasks.append("Craft wood_pickaxe.")
        if "get 1 wood_sword, " in obj_diff:
            possible_tasks.append("Craft wood_sword.")
        if "get 2 wood_sword, " in obj_diff:
            possible_tasks.append("Craft wood_sword.")
        if "get 3 wood_sword, " in obj_diff:
            possible_tasks.append("Craft wood_sword.")
        if "get 4 wood_sword, " in obj_diff:
            possible_tasks.append("Craft wood_sword.")
        if "get 1 stone_pickaxe, " in obj_diff:
            possible_tasks.append("Craft stone_pickaxe.")
        if "get 2 stone_pickaxe, " in obj_diff:
            possible_tasks.append("Craft stone_pickaxe.")
        if "get 3 stone_pickaxe, " in obj_diff:
            possible_tasks.append("Craft stone_pickaxe.")
        if "get 4 stone_pickaxe, " in obj_diff:
            possible_tasks.append("Craft stone_pickaxe.")
        if "get 1 stone_sword, " in obj_diff:
            possible_tasks.append("Craft stone_sword.")
        if "get 2 stone_sword, " in obj_diff:
            possible_tasks.append("Craft stone_sword.")
        if "get 3 stone_sword, " in obj_diff:
            possible_tasks.append("Craft stone_sword.")
        if "get 4 stone_sword, " in obj_diff:
            possible_tasks.append("Craft stone_sword.")
        if "get 1 iron_pickaxe, " in obj_diff:
            possible_tasks.append("Craft iron_pickaxe.")
        if "get 2 iron_pickaxe, " in obj_diff:
            possible_tasks.append("Craft iron_pickaxe.")
        if "get 3 iron_pickaxe, " in obj_diff:
            possible_tasks.append("Craft iron_pickaxe.")
        if "get 4 iron_pickaxe, " in obj_diff:
            possible_tasks.append("Craft iron_pickaxe.")
        if "get 1 iron_sword, " in obj_diff:
            possible_tasks.append("Craft iron_sword.")
        if "get 2 iron_sword, " in obj_diff:
            possible_tasks.append("Craft iron_sword.")
        if "get 3 iron_sword, " in obj_diff:
            possible_tasks.append("Craft iron_sword.")
        if "get 4 iron_sword, " in obj_diff:
            possible_tasks.append("Craft iron_sword.")
        if "find crafting table, " in obj_diff:
            possible_tasks.append("Place crafting table.")
        if "find furnace, " in obj_diff:
            possible_tasks.append("Place furnace.")
        if "find cow, " in obj_diff:
            possible_tasks.append("Find cows.")
        if "find water, " in obj_diff:
            possible_tasks.append("Find water.")
        
        if len(possible_tasks) == 0:
            possible_tasks.append("Survive.")
        
        chosen_task = np.random.choice(possible_tasks)
        
        return chosen_task

    def get_next_task(self, text_obs):
        """
         You see cow, grass, and tree. 
        You have nothing in your inventory. 
        Your health level is high, food level is high, drink level is high, energy is high.
        """
        
        split_text_obs = text_obs.split("You")[2:]
        
        available_actions = [
            "Find cows.", "Find water.", "Find stone.", "Find tree.",
            "Chop tree.", "Kill the cow.", "Mine stone.", "Drink water.",
            "Mine coal.", "Mine iron.", "Mine diamond.", "Kill the zombie.",
            "Kill the skeleton.", "Craft wood_pickaxe.", "Craft wood_sword.",
            "Place crafting table.", "Place furnace.", "Craft stone_pickaxe.",
            "Craft stone_sword.", "Craft iron_pickaxe.", "Craft iron_sword.",
            "Sleep."
        ]
        # return np.random.choice(available_actions)
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
                actions_probw.append(.5)
        if "water" in split_text_obs[0]:
            available_actions.append("Drink water.")
            actions_probw.append(.25)
        if "coal" in split_text_obs[0]:
            if "pickaxe" in split_text_obs[1]:
                available_actions.append("Mine coal.")
                actions_probw.append(1.)
        if "iron" in split_text_obs[0]:
            if "stone_pickaxe" in split_text_obs[1]:
                available_actions.append("Mine iron.")
                actions_probw.append(1.)
        if "diamond" in split_text_obs[0]:
            if "iron_pickaxe" in split_text_obs[1]:
                available_actions.append("Mine diamond.")
                actions_probw.append(1.)
        if "zombie" in split_text_obs[0]:
            if "sword" in split_text_obs[1]:
                available_actions.append("Kill the zombie.")
                actions_probw.append(1.5)
            else:
                available_actions.append("Kill the zombie.")
                actions_probw.append(.5)
        if "skeleton" in split_text_obs[0]:
            if "sword" in split_text_obs[1]:
                available_actions.append("Kill the skeleton.")
                actions_probw.append(1.5)
            else:
                available_actions.append("Kill the skeleton.")
                actions_probw.append(.5)
        if "wood" in split_text_obs[1]:
            if "crafting_table" in split_text_obs[0]:
                available_actions.append("Craft wood_pickaxe.")
                actions_probw.append(.25)
                available_actions.append("Craft wood_sword.")
                actions_probw.append(.25)
            else:
                available_actions.append("Place crafting table.")
                actions_probw.append(.25)
        if "stone" in split_text_obs[1]:
            if "crafting_table" in split_text_obs[0]:
                available_actions.append("Craft stone_pickaxe.")
                actions_probw.append(3.)
                available_actions.append("Craft stone_sword.")
                actions_probw.append(3.)
            available_actions.append("Place furnace.")
            actions_probw.append(1.)
        if "iron" in split_text_obs[1]:
            if "frunace" in split_text_obs[0]:
                available_actions.append("Craft iron_pickaxe.")
                actions_probw.append(5.)
                available_actions.append("Craft iron_sword.")
                actions_probw.append(5.)
        if "food: 3/9" in split_text_obs[2] or "food: 2/9" in split_text_obs[2] or "food: 1/9" in split_text_obs[2] or "food: 0/9" in split_text_obs[2]:
            if "cow" in split_text_obs[0]:
                available_actions.append("Kill the cow.")
                actions_probw.append(2.)
            else:
                available_actions.append("Find cows.")
                actions_probw.append(1.)
        if "drink: 3/9" in split_text_obs[2] or "drink: 2/9" in split_text_obs[2] or "drink: 1/9" in split_text_obs[2] or "drink: 0/9" in split_text_obs[2]:
            if "water" in split_text_obs[0]:
                available_actions.append("Drink water.")
                actions_probw.append(.5)
            else:
                available_actions.append("Find water.")
                actions_probw.append(1.)
        if "energy: 3/9" in split_text_obs[2] or "energy: 2/9" in split_text_obs[2] or "energy: 1/9" in split_text_obs[2] or "energy: 0/9" in split_text_obs[2]:
            available_actions.append("Sleep.")
            actions_probw.append(2.)
            
        actions_probw = np.array(actions_probw)
        actions_probw = actions_probw / np.sum(actions_probw)
        chosen_task = np.random.choice(available_actions, p=actions_probw)
        
        if len(available_actions) > 0:
            return chosen_task
        else:
            return "Chop tree." 

    # get difference between two frames
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
      
    # get text state      
    def _get_text_state(self):
        """
        return a string describing the state of the environment
        """
        
        player_pos = "You are at " + str(self.env._player.pos)
        surrounding_state = self._get_surrounding_state()
        inventory = self._get_inventory()
        inner_state = self._get_inner_state()
        text_state = player_pos + "\n" + surrounding_state + "\n" + inventory + "\n" + inner_state
        if self.env._player.sleeping:
            text_state += "\nYou are sleeping. "

        return text_state  
    
    def _get_inventory(self):
        """
        return a string describing the player's inventory 
        """
        STATUS_MAX_VALUE = 9
        STATUS_ITEMS = ['health', 'food', 'drink', 'energy']
        
        inventory = "You have in your inventory: "
        empty_inventory = True
        for k, v in self.env.player.inventory.items():
            if k not in STATUS_ITEMS and v > 0:
                k = "plant" if k == "sapling" else k
                inventory += f"{v} {k}, "
                empty_inventory = False
        if empty_inventory:
            inventory = "You have nothing in your inventory."
        if inventory[-2] == ",":
            inventory = inventory[:-2] + ". "
            
        return inventory
    
    def _get_inner_state(self):
        """
        return a string describing the player's inner state
        """
        
        STATUS_MAX_VALUE = 9
        STATUS_ITEMS = ['health', 'food', 'drink', 'energy']
        inner_state = "Your inner properties: "
        
        # the first 4 items in the inventory are the player's status
        for k, v in self.env.player.inventory.items():
            if k in STATUS_ITEMS:
                inner_state += f"{k}: {v}/{STATUS_MAX_VALUE}, "
        inner_state = inner_state[:-2] + ". "
                
        return inner_state
        
    def _get_surrounding_state(self):
        
        # {0: None, 1: 'water', 2: 'grass', 3: 'stone', 4: 'path', 5: 'sand', 
        # 6: 'tree', 7: 'lava', 8: 'coal', 9: 'iron', 10: 'diamond', 11: 'table', 
        # 12: 'furnace', 13: 'player', 14: 'cow', 15: 'zombie', 16: 'skeleton', 17: 'arrow', 18: 'plant'}
        canvas = self.env._text_view.get_map(self.env._player.pos)
        
        env_items = ['water', 'stone', 'tree', 'coal', 'iron', 'diamond', 'crafting_table', 'furnace', 'plant', 'cow', 'zombie', 'skeleton']
        surrounding_items = []
        for block in canvas.flatten()[self.dist_idx]:
            if self.names[block] not in surrounding_items and self.names[block] in env_items:
                surrounding_items.append(self.names[block])
        if len(surrounding_items) > 0:
            surrounding_state = "You see around you(from near to far): " + ", ".join(surrounding_items)
            surrounding_state += ". "
        else:
            surrounding_state = "You see nothing around you. "
        
        return surrounding_state

        