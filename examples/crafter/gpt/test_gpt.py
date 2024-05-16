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

import time

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
    # img.save("run_results/image.png")
    return img

def render(seed, model_name, random):

    # config
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()
    cfg.seed = seed * 17 + 110171
    
    # begin to test
    env = make(
        "Crafter",
        env_num=1,
        cfg=cfg,
    )

    # init the agent
    agent = Agent(Net(env, cfg=cfg))
    agent.set_env(env)
    agent.load(model_name)
    
    # GPT client
    from openrl.envs.crafter.gpt_client import GPTClient
    import asyncio
    api_key = "EMPTY" 
    api_base = "http://0.0.0.0:11016" 
    model = "meta-llama/Meta-Llama-3-8B" 

    # begin to test
    obs, info = env.reset()
    
    # statistics 
    history_imgs = []
    traj = []
    env_step = 0
    total_reward = 0
    original_rewards = 0
    
    while True:
        
        current_task = "Survive."
        
        # history_imgs.append(save_img(obs, current_task))
        
        # env step
        obs = env.set_task(obs, [current_task]) # get correct observation
        action, _ = agent.act(obs, info=info, deterministic=True)
        extra_data = {"task": [current_task]} # get correct rewards
        obs, r, done, info = env.step(action, extra_data=extra_data)
        traj.append(info[0]["text_obs"])
        if env_step == 100:
            print(traj)
            exit()
        env_step += 1
        original_rewards += info[0]["original_rewards"][0][0]
        total_reward += r[0][0][0]

        if all(done):
            # name = "run_results/crafter.gif" if not random else "run_results/crafter-random.gif"
            # history_imgs[0].save(
            #     name, 
            #     save_all=True, 
            #     append_images=history_imgs[1:], 
            #     duration=100, 
            #     loop=0
            # )
            break
    
    env.close()
    
    del agent
    
    return env_step, total_reward, original_rewards


if __name__ == "__main__":

    model = "../models/crafter_agent-10M-75/"
    print("model", model)
    
    traj_diff = []
    episode_len = []
    total_reward = []
    original_rewards = []
    
    begin = time.time()
    for seed in range(100):
        env_step, rew, orew = render(seed, model, random=False)
        
        episode_len.append(env_step)
        total_reward.append(rew)
        original_rewards.append(orew)
    
        # print mean
        print("survival time-step", np.mean(episode_len))
        print("original_rewards", np.mean(original_rewards))
        print("instruction following rewards", np.mean(total_reward))

    print("model", model)