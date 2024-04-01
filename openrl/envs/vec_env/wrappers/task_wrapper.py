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
        
    def reset(self, **kwargs):
        
        obs, info = self.env.reset(**kwargs)
        
        obs["policy"] = {
            "image": obs["policy"],
            "task_emb": np.zeros(obs["policy"].shape[:2] + (384,), dtype=np.float32),
        }
        obs["critic"] = {
            "image": obs["critic"],
            "task_emb": np.zeros(obs["critic"].shape[:2] + (384,), dtype=np.float32),
        }
        
        return obs, info
    
    def step(
        self, 
        action: ActType, 
        extra_data: Optional[Dict[str, Any]] = None,
        *args, 
        **kwargs,
    ) -> Union[Any, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        
        obs, rewards, dones, infos = self.env.step(action, extra_data, *args, **kwargs)
        
        obs["policy"] = {
            "image": obs["policy"],
            "task_emb": np.zeros(obs["policy"].shape[:2] + (384,), dtype=np.float32),
        }
        obs["critic"] = {
            "image": obs["critic"],
            "task_emb": np.zeros(obs["critic"].shape[:2] + (384,), dtype=np.float32),
        }

        return obs, rewards, dones, infos
