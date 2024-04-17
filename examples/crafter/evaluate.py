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

def render(seed, tasks, cnts, model_name):

    # config
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()
    cfg.seed = seed
    
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

    # begin to test
    obs, info = env.reset()
    
    # set tasks
    if tasks is None:
        current_task = "Survive."
        cnt_down = 1
    else:
        current_task = tasks[0]
        cnt_down = cnts[0]
    query_cnt = 1
    cnt_down -= 1
    
    # statistics 
    traj = []
    env_step = 0
    total_reward = 0
    
    all_task = ["Chop tree.", "Kill the cow.", "Mine stone.", "Drink water.", "Mine coal.", "Mine iron.", "Mine diamond.", "Kill the zombie.", "Kill the skeleton.", "Craft wood_pickaxe.", "Craft wood_sword.", "Place crafting table.", "Craft stone_pickaxe.", "Craft stone_sword.", "Craft iron_pickaxe.", "Craft iron_sword.", "Find cows.", "Find water.", "Sleep.", "Place furnace.", "Find tree.", "Find stone.", "Survive."]
    
    while True:
        
        obs = env.set_task(obs, [current_task]) # get correct observation
        action, _ = agent.act(obs, info=info, deterministic=True)
        extra_data = {"task": [current_task]} # get correct rewards
        obs, r, done, info = env.step(action, extra_data=extra_data)
        env_step += 1
        if env_step < 150:
            total_reward += r[0]
        if env_step > 25 and env_step < 150:
            traj.append(obs["policy"]["image"][0,0])
        
        if cnt_down == 0:
            if tasks is None:
                current_task = all_task[np.random.randint(len(all_task))]
                cnt_down = 1 
            else:
                if query_cnt >= len(tasks):
                    break
                current_task = tasks[query_cnt] # all_task[np.random.randint(len(all_task))]
                cnt_down = cnts[query_cnt]
                query_cnt += 1
        
        cnt_down -= 1

        if all(done) or env_step >= 1279:
            break
    
    env.close()
    
    return traj, env_step, total_reward


if __name__ == "__main__":
    
    model = "models/crafter_agent-2M-27/"
    print("model", model)
    
    traj_diff = []
    episode_len1 = []
    episode_len2 = []
    total_reward1 = []
    total_reward2 = []
    
    begin = time.time()
    for seed in range(100):
        tasks = [
            "Chop tree.", 
            "Place crafting table.",
            "Craft wood_pickaxe.", 
            "Craft wood_sword.", 
            "Find stone.", 
            "Mine stone.", 
            "Place crafting table.",
            "Craft stone_pickaxe.", 
            "Craft stone_sword.", 
            "Find water.", 
            "Drink water.", 
            "Find cows.", 
            "Kill the cow.", 
            "Sleep.",
            "Mine stone.", 
            "Find water.", 
            "Drink water.", 
            "Find cows.", 
            "Kill the cow.", 
            "Sleep.",
            "Mine stone.", 
        ]
        cnts = [
            30, 2, 2, 1, 10, 30, 2, 2, 1, 10, 10, 10, 10, 10, 100, 10, 10, 10, 10, 10, 1000
        ]
        traj1, env_step1, r1 = render(seed, tasks, cnts, model)
        traj2, env_step2, r2 = render(seed, None, None, model)
        min_len = min(len(traj1), len(traj2))
        diff = np.array(traj1)[:min_len] - np.array(traj2)[:min_len]
        diff = np.mean((diff**2).sum(-1).sum(-1).sum(-1))
        
        episode_len1.append(env_step1)
        episode_len2.append(env_step2)
        total_reward1.append(r1)
        total_reward2.append(r2)
        traj_diff.append(diff)
    
        # print mean
        print("survival time-step (good instruction):", np.mean(episode_len1))
        print("instruction following result (good instruction):", np.mean(total_reward1))
        print("survival time-step (random instruction):", np.mean(episode_len2))
        print("instruction following result (random instruction):", np.mean(total_reward2))
        print("traj_diff", np.mean(traj_diff), "time", time.time()-begin)

    print("model", model)
    
    
# result

# model models/crafter_agent-100M-2/ PPO
# total_step 341.69 total_reward 7.66 traj_diff 362176.9051933416

# model models/crafter_agent-100M-4/  PPO+high_entropy
# total_step 334.71 total_reward 7.57 traj_diff 379925.13196826645

# model models/crafter_agent-2M-22/ fine-tuned-no_DKL
# total_step 291.36 total_reward 11.46 traj_diff 543734.2699659602

# model models/crafter_agent-2M-23/ fine-tuned
# total_step 366.98 total_reward 10.07 traj_diff 527047.964233871


# total_step 292.64 total_reward 9.12 traj_diff 472657.5765405597 time 1280.8099598884583                        
# model models/crafter_agent-10M-30/
# total_step 342.99 total_reward 8.5 traj_diff 473077.856060225 time 1728.4287180900574                          
# model models/crafter_agent-10M-31/
# total_step 356.39 total_reward 10.43 traj_diff 490570.8574656822 time 1743.899439573288                        
# model models/crafter_agent-10M-32/
# total_step 325.7 total_reward 10.92 traj_diff 545414.0732909743 time 1545.4534697532654                        
# model models/crafter_agent-10M-33/

# total_step 358.55 total_reward 6.83 traj_diff 489648.8531827958 time 1869.1202957630157                        
# model models/crafter_agent-10M-32/
# total_step 360.76 total_reward 6.51 traj_diff 459902.8622623656 time 1801.3613457679749                        
# model models/crafter_agent-10M-28/
# total_step 364.73 total_reward 6.88 traj_diff 461869.5444441244 time 1838.2085411548615                        
# model models/crafter_agent-2M-23/
# total_step 353.58 total_reward 7.1 traj_diff 384791.31483870966 time 1835.861520767212                         
# model models/crafter_agent-100M-2/

# random tasks
# total_step 335.73 total_reward 14.73 traj_diff 549567.6901632039 time 1716.5807662010193
# model models/crafter_agent-10M-34/
# total_step 303.53 total_reward 12.95 traj_diff 561920.0116185134 time 1492.3062028884888
# model models/crafter_agent-10M-35/
# total_step 275.99 total_reward 15.92 traj_diff 440635.70799602295 time 1418.0942595005035
# model models/crafter_agent-10M-36/
# total_step 337.13 total_reward 10.39 traj_diff 479959.5726845534 time 1725.2639038562775
# model models/crafter_agent-10M-37/

# given tasks
# total_step 340.74 total_reward 9.77 traj_diff 570636.0829032259 time 1730.7477412223816                                                        
# model models/crafter_agent-10M-34/
# total_step 283.51 total_reward 10.13 traj_diff 575452.5557657283 time 1480.4870274066925                       ─────────────────────────────────
# model models/crafter_agent-10M-35/
# total_step 275.26 total_reward 10.5 traj_diff 453112.5234699513 time 1446.4643087387085                        
# model models/crafter_agent-10M-36/
# total_step 330.93 total_reward 10.02 traj_diff 511449.4917741935 time 1616.5335485935211                       
# model models/crafter_agent-10M-37/