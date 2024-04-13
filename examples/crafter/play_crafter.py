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

import imageio
import numpy as np

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.envs.wrappers import GIFWrapper
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent


from PIL import Image, ImageDraw, ImageFont


def save_img(obs, task, idx=0):
    img = obs["policy"]["image"][0, 0]
    img = img.transpose((1, 2, 0))
    img = Image.fromarray(img)
    img = img.resize((256, 256))
    draw = ImageDraw.Draw(img)
    draw.text((10,10), task, fill=(255,0,0))
    img.save("run_results/image.png")
    return img


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
    agent.load("crafter_agent-10M-3/")

    # begin to test
    trajectory = []
    obs, info = env.reset()
    step = 0
    
    print("Enter the task: ", end="")
    current_task = input()
    last_obs = info[0]["text_obs"]
    print("Enter the repeat time:", end="")
    cnt_down = int(input())
    print("")
    cnt_down -= 1

    img = save_img(obs, current_task)
    trajectory.append(img)
    
    while True:
        
        # Based on environmental observation input, predict next action.
        obs = env.set_task(obs, [current_task])
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action, given_task=[current_task])
        step += 1
           
        img = save_img(obs, current_task, step)
        trajectory.append(img)
        
        if cnt_down == 0:
            print("Enter the task:", end="")
            input_text = input()
            current_task = input_text if input_text != "" else current_task
            if current_task == "exit":
                break
            print("Enter the repeat time:", end="")
            cnt_down = int(input())
            last_state = info[0]["text_obs"]
        cnt_down -= 1
                

        if all(done):
            break
        
    print("total_step", step)

    # save the trajectory to gif
    trajectory[0].save(
        "run_results/crafter.gif", 
        save_all=True, 
        append_images=trajectory[1:], 
        duration=100, 
        loop=0
    )

    env.close()


if __name__ == "__main__":
    render()


# chop trees.
# 30
# craft wood_pickaxe.
# 5
# mine stones.
# 50
# drink water.
# 30
# exit
