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
import numpy as np

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent

def evaluation():
    
    num_episode = 10
    
    for seed in range(1,num_episode+1):
        
        env = make(
            "navigation-1", 
            render_mode="group_human", 
            env_num=1, 
            asynchronous=False
        )
        
        cfg_parser = create_config_parser()
        cfg = cfg_parser.parse_args()
        net = Net(env, cfg=cfg, device="cpu")
        agent = Agent(net, use_wandb=False)
        agent.load("./result/module.pt")
        agent.set_env(env)
    
        obs, info = env.reset(seed=seed)
        done = False
        step = 0
        total_reward = 0
        while not np.any(done):
            action, _ = agent.act(obs, deterministic=True)
            obs, r, done, info = env.step(action)
            step += 1
            total_reward += np.mean(r)
        print(f"total_reward: {total_reward}")
        env.close()


if __name__ == "__main__":
    evaluation()
