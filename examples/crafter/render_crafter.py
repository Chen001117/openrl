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

from PIL import Image, ImageDraw, ImageFont

def save_img(obs, task):
    img = obs["policy"]["image"][0, 0]
    img = img.transpose((1, 2, 0))
    img = Image.fromarray(img)
    img = img.resize((256, 256))
    draw = ImageDraw.Draw(img)
    draw.text((10,10), task, fill=(255,0,0))
    # img.save("run_results/image.png")
    return img

available_actions = [
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
    agent.load("models/crafter_agent-10M-00/")

    # begin to test
    history_imgs = []
    obs, info = env.reset()
    step = 0
    total_reward = 0
    while True:
        
        # obs = env.set_task(obs, [current_task])
        
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, info=info, deterministic=True)
        obs, r, done, info = env.step(action)
        current_task = available_actions[action[0,0,1]]
        step += 1
        total_reward += r

        if all(done):
            break
        
        history_imgs.append(save_img(obs, current_task))
        
    print("step", step, "total_reward", total_reward)

    # save the trajectory to gif
    import imageio

    name = "run_results/crafter.gif"
    history_imgs[0].save(
        name, 
        save_all=True, 
        append_images=history_imgs[1:], 
        duration=100, 
        loop=0
    )
        
    env.close()
    
    return total_reward


if __name__ == "__main__":
    rewards = []
    for seed in range(1):
        reward = render(seed)
        rewards.append(reward)
    print("rewards", np.mean(rewards))
