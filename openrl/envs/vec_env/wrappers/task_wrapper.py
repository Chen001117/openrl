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
from typing import Any, Dict, List, Optional, Union

import copy
import numpy as np
from gymnasium.core import ActType
from gymnasium import Env, spaces
from gymnasium.spaces.dict import Dict as DictSpace

from openrl.envs.vec_env.base_venv import BaseVecEnv
from openrl.envs.vec_env.wrappers.base_wrapper import VecEnvWrapper
from openrl.rewards.base_reward import BaseReward

from openrl.envs.crafter.bert import BertEncoder


class TaskWrapper(VecEnvWrapper):
    def __init__(self, env: BaseVecEnv):
        super().__init__(env)
        
        for key in ["policy", "critic"]:
            self.observation_space[key] = DictSpace(
                {
                    "image": copy.deepcopy(self.env.observation_space[key]),
                    "task_emb": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(384,)
                    ),
                }
            )
    
        self.encoder = BertEncoder()
        
        self.available_actions = [
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

    def reset(
            self, 
            **kwargs
        ):
        
        obs, info = self.env.reset(**kwargs)
        task = ["Survive."] * len(obs["policy"])
        task_emb = self.encoder(task)
        task_emb = np.expand_dims(task_emb, axis=1)
        
        obs["policy"] = {
            "image": obs["policy"],
            "task_emb": task_emb,
        }
        obs["critic"] = {
            "image": obs["critic"],
            "task_emb": task_emb,
        }
        
        return obs, info
    
    def step(
        self, 
        actions: ActType, 
        extra_data: Optional[Dict[str, Any]] = None,
        *args, 
        **kwargs,
    ) -> Union[Any, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        
        obs, rewards, dones, infos = self.env.step(actions, extra_data, *args, **kwargs)
        
        if extra_data is None:
            obs["policy"] = {"image": obs["policy"]}
            obs["critic"] = {"image": obs["critic"]}
        else:
            task = self.get_tasks(infos, actions)
            task_emb = self.encoder(task)
            task_emb = np.expand_dims(task_emb, axis=1)
        
            obs["policy"] = {
                "image": obs["policy"],
                "task_emb": task_emb,
            }
            obs["critic"] = {
                "image": obs["critic"],
                "task_emb": task_emb,
            }

        return obs, rewards, dones, infos
    
    def set_task(self, obs, task):
        
        task_emb = self.encoder.encode(task)
        task_emb = np.expand_dims(task_emb, axis=1)
        
        obs["policy"] = {
            "image": obs["policy"]["image"],
            "task_emb": task_emb,
        }
        obs["critic"] = {
            "image": obs["critic"]["image"],
            "task_emb": task_emb,
        }
        
        return obs

    def get_tasks(self, infos, actions):
        
        tasks = []
        for info, action in zip(infos, actions):
            tasks.append(self.available_actions[action[0,1]])
        return tasks
    
    @property
    def use_monitor(self):
        return True