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
from openrl.envs.wrappers import GIFWrapper
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent


def render():
    # begin to test
    env = make(
        "Crafter",
        render_mode="human",
        env_num=1,
    )

    # config
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()

    # init the agent
    agent = Agent(Net(env, cfg=cfg))
    # set up the environment and initialize the RNN network.
    agent.set_env(env)
    # load the trained model
    agent.load("crafter_agent-100M-3/")

    total_step = []

    for i in range(100):
        step = 0
        obs, info = env.reset(given_task=["Chop tree."])
        while True:
            # Based on environmental observation input, predict next action.
            action, _ = agent.act(obs, deterministic=True)
            tasks = ["Chop tree.", "Mine stone.", "Kill the cow."]
            t = np.random.randint(0, 3)
            t = tasks[t]
            obs, r, done, info = env.step(action, given_task=[t])
            step += 1

            if all(done):
                break
        
        total_step.append(step)
        
        
        print(i, "Average steps: ", np.mean(total_step))
        
        


    env.close()


if __name__ == "__main__":
    render()
