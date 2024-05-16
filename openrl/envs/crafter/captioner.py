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

import copy 
import numpy as np

import json

  
class Captioner():
    
    """
    extract the text description of the environment from the observation
    though we query the environment for the text description, all the information we used is from the observation.
    """
    
    def __init__(self, env) -> None:
        
        self.env = env
        
        self.names = {value : key for (key, value) in self.env._world._mat_ids.items()}
        self.names[11] = "crafting_table"
        self.names[13] = "player"
        self.names[14] = "cow"
        self.names[15] = "zombie"
        self.names[16] = "skeleton"
        self.names[17] = "arrow"
        self.names[18] = "plant"
        
        self.env_items = [
            'water', 
            'stone', 
            'tree', 
            'coal', 
            'iron', 
            'diamond', 
            'crafting_table', 
            'furnace', 
            'plant', 
            'cow', 
            'zombie', 
            'skeleton'
        ]
        
        self.inventory_items = [
            'plant',
            'wood',
            'stone',
            'coal',
            'iron',
            'diamond',
            'wood_pickaxe',
            'stone_pickaxe',
            'iron_pickaxe',
            'wood_sword',
            'stone_sword',
            'iron_sword'
        ]
        
        self.direction = [
            "north",
            "northeast",
            "east",
            "southeast",
            "south",
            "southwest",
            "west",
            "northwest"
        ]
        
        self.idx_x, self.idx_y = np.meshgrid(np.arange(7), np.arange(9))
        self.idx_x = self.idx_x - 3
        self.idx_y = self.idx_y - 4
        self.dist_map = np.sqrt(self.idx_x ** 2 + self.idx_y ** 2)
        self.dist_idx = np.argsort(self.dist_map.flatten())[1:]
   
    def __call__(self, reset=False):
        """
        return a string describing the state of the environment
        """
        
        if reset:
            self.all_map = np.zeros((64, 64))
        
        text_sur, sur = self._get_surrounding_state()
        text_inv, inv = self._get_inventory()
        text_inn, inn = self._get_inner_state()
        text_state = text_sur + "\n" + text_inv + "\n" + text_inn
        if self.env._player.sleeping:
            text_state += "\nYou are sleeping. "  
        
        center_x, center_y = self.env._player.pos
        min_x, max_x = max(0, center_x-4), min(64, center_x+5)
        min_y, max_y = max(0, center_y-3), min(64, center_y+4)
        tmap = self.env._text_view.get_map(self.env._player.pos)
        try:
            self.all_map[min_x:max_x,min_y:max_y] = tmap
        except:
            while (tmap[0] == 0).all():
                tmap = tmap[1:]
            while (tmap[-1] == 0).all():
                tmap = tmap[:-1]
            while (tmap[:, 0] == 0).all():
                tmap = tmap[:, 1:]
            while (tmap[:, -1] == 0).all():
                tmap = tmap[:, :-1]
            self.all_map[min_x:max_x,min_y:max_y] = tmap
        
        nearest_list = []
        # for item in ["grass", "cow", "water", "coal", "iron", "diamond", "tree"]:
        for item in [2, 14, 1, 8, 9, 10, 6]:
            item_idxs = np.where(self.all_map == item)
            min_dist = 1e6
            min_idx = [-1, -1]
            for item_x, item_y in zip(item_idxs[0], item_idxs[1]):
                center_x, center_y = self.env._player.pos
                dist = np.sqrt((item_x - center_x) ** 2 + (item_y - center_y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = [item_x, item_y]
            nearest_list.append(min_idx)
            
        dict_state = {
            "surrounding": sur, 
            "inventory": inv, 
            "inner": inn, 
            "sleeping": self.env._player.sleeping,
            "pos": self.env._player.pos,
            "nearest_idx": nearest_list,
        }

        return text_state, dict_state
    
    def _get_inventory(self):
        """
        return a string describing the player's inventory 
        """
        STATUS_ITEMS = ['health', 'food', 'drink', 'energy']
        
        inventory = dict()
        for k, v in self.env.player.inventory.items():
            if k not in STATUS_ITEMS and v > 0:
                k = "plant" if k == "sapling" else k
                inventory[k] = v
            
        full_inventory = dict()
        for item in self.inventory_items:
            if item in inventory:
                full_inventory[item] = inventory[item]
            else:
                full_inventory[item] = 0

        if len(inventory) > 0:
            text_inv = "Your inventory: " + json.dumps(inventory)
        else:
            text_inv = "You have nothing in your inventory. "
            
        return text_inv, full_inventory
    
    def _get_inner_state(self):
        """
        return a string describing the player's inner state
        """
        
        STATUS_MAX_VALUE = 9
        STATUS_ITEMS = ['health', 'food', 'drink', 'energy']
        
        # the first 4 items in the inventory are the player's status
        inner_state = dict()
        for k, v in self.env.player.inventory.items():
            if k in STATUS_ITEMS:
                inner_state[k] = v
        
        text_in = "Your status: " + json.dumps(inner_state)
                
        return text_in, inner_state
        
    def _get_surrounding_state(self):
        
        # {0: None, 1: 'water', 2: 'grass', 3: 'stone', 4: 'path', 5: 'sand', 
        # 6: 'tree', 7: 'lava', 8: 'coal', 9: 'iron', 10: 'diamond', 11: 'table', 
        # 12: 'furnace', 13: 'player', 14: 'cow', 15: 'zombie', 16: 'skeleton', 17: 'arrow', 18: 'plant'}
        canvas = self.env._text_view.get_map(self.env._player.pos)
        
        surrounding_items = []
        for block_idx in self.dist_idx:
            block = canvas.flatten()[block_idx]
            if self.names[block] in self.env_items:
                x = self.idx_x.flatten()[block_idx]
                y = self.idx_y.flatten()[block_idx]
                px, py = self.env._player.pos
                item = {
                    "type": self.names[block],
                    # "pos": [int(x+px), int(y+py)],
                    "direction": self.get_direction(x, y),
                    "distance": int(self.dist_map.flatten()[block_idx]*10.)/10.
                }
                surrounding_items.append(item)
                
        
        if len(surrounding_items) > 0:
            text_sur = "You see around you: " + json.dumps(surrounding_items)
            return text_sur, surrounding_items
        else:
            text_sur = "You see nothing around you. "
            return text_sur, surrounding_items
    

    def get_direction(self, x, y):
        """
        return the direction of the vector (x, y)
        """
        if x == 0 and y == 0:
            return "center"
        elif x == 0:
            return "north" if y > 0 else "south"
        elif y == 0:
            return "east" if x > 0 else "west"
        elif x > 0:
            return "northeast" if y > 0 else "southeast"
        else:
            return "northwest" if y > 0 else "southwest"

        