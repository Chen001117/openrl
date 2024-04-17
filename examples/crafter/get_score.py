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

from tqdm import tqdm

import numpy as np

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.envs.wrappers import GIFWrapper
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent


def render(seed):
    # begin to test
    env = make(
        "Crafter",
        render_mode="human",
        env_num=1,
    )

    # config
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()
    cfg.seed = seed

    # init the agent
    agent = Agent(Net(env, cfg=cfg))
    # set up the environment and initialize the RNN network.
    agent.set_env(env)
    # load the trained model
    agent.load("models/crafter_agent-100M-2/")

    # begin to test
    obs, info = env.reset()
    total_reward = 0
    for step in tqdm(range(1001000)):
        
        tasks = ["Chop tree.", "Mine stone.", "Kill the cow."]
        current_task = tasks[np.random.randint(0, 3)]
        obs = env.set_task(obs, [current_task])
        
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, info=info, deterministic=True)
        obs, r, done, info = env.step(action)

    env.close()


if __name__ == "__main__":
    seed = 1
    render(seed)
