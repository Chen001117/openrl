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

import json
import time

import imageio
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

def render(seed, model_name, random):

    # config
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()
    model_name = cfg.model_dir
    print("model", model_name)
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
    np.random.seed(seed)

    # begin to test
    obs, info = env.reset()
    
    # statistics 
    history_imgs = []
    env_step = 0
    total_reward = 0
    original_rewards = 0
    traj = {
        "image":[],
        "task_emb":[],
        "actions":[],
        "step":[],
    }
    
    all_task = [
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
    
    state = 0
    cnt = 0
    
    while True:
        
        if random:
            if env_step == 0:
                current_task = str(np.random.choice(all_task))
            elif np.random.rand() < 0.1:
                current_task = str(np.random.choice(all_task))

        else:
            
            # get current text_obs
            text_obs = info[0]["dict_obs"]
            
            near_zombie = False
            for item in text_obs["surrounding"]:
                if item["type"] == "zombie" and item["distance"] <= 3.5:
                    near_zombie = True
                    break
            near_skeleton = False
            for item in text_obs["surrounding"]:
                if item["type"] == "skeleton" and item["distance"] <= 3.5:
                    near_skeleton = True
                    break
            
            food = text_obs["inner"]["food"]
            drink = text_obs["inner"]["drink"]
            energy = text_obs["inner"]["energy"]
            
            if near_zombie:
                current_task = "Kill the zombie."
            elif near_skeleton:
                current_task = "Kill the skeleton."
            
            elif food <= drink and food <= energy and food <= 3:
                current_task = "Kill the cow."
            elif drink < food and drink <= energy and drink <= 3:
                current_task = "Drink water."
            elif energy < food and energy < drink and energy <= 3:
                current_task = "Sleep."
            
            elif state == 0:
                if text_obs["inventory"]["plant"] > 0:
                    current_task = "Place sapling."
                    state = 0.1
                current_task = "Collect sapling."
            elif state == 0.1:
                if "plant" in str(text_obs["surrounding"]):
                    current_task = "Chop tree."
                    state = 0.5
                current_task = "Place sapling."
            elif state == 0.5: # begining of the game
                if text_obs["inventory"]["wood"] >= 4:
                    current_task = "Place crafting table."
                    state = 1
                elif text_obs["inventory"]["wood_pickaxe"] >= 1:
                    current_task = "Mine stone."
                    state = 3
                else:
                    current_task = "Chop tree."
            elif state == 1: # place crafting table
                if "crafting_table" in str(text_obs["surrounding"]):
                    current_task = "Craft wood_pickaxe."
                    state = 2
                else:
                    current_task = "Place crafting table."
            elif state == 2: # craft wood_pickaxe
                if text_obs["inventory"]["wood_pickaxe"] >= 1 and text_obs["inventory"]["wood_sword"] >= 1:
                    current_task = "Mine stone."
                    state = 3
                elif text_obs["inventory"]["wood_pickaxe"] >= 1 and text_obs["inventory"]["wood_sword"] < 1:
                    current_task = "Craft wood_sword."
                    state = 2.5
                else:
                    current_task = "Craft wood_pickaxe."
            elif state == 2.5: # craft wood_sword
                if text_obs["inventory"]["wood_sword"] >= 1:
                    current_task = "Mine stone."
                    state = 3
                else:
                    current_task = "Craft wood_sword."
            elif state == 3: # mine stone
                if "crafting_table" in str(text_obs["surrounding"]) and text_obs["inventory"]["stone"] >= 2 and text_obs["inventory"]["wood"] >= 2:
                    current_task = "Craft stone_pickaxe."
                    state = 4
                elif text_obs["inventory"]["wood"] < 4:
                    current_task = "Chop tree."
                elif text_obs["inventory"]["stone"] >= 2 and text_obs["inventory"]["wood"] >= 4:
                    current_task = "Place crafting table."
                    state = 3.5
                elif text_obs["inventory"]["stone_pickaxe"] >= 1:
                    current_task = "Mine iron."
                    state = 5
                else:
                    current_task = "Mine stone."
            elif state == 3.5: # place crafting table
                if "crafting_table" in str(text_obs["surrounding"]):
                    current_task = "Craft stone_pickaxe."
                    state = 4
                else:
                    current_task = "Place crafting table."
            elif state == 4: # craft stone_pickaxe
                if "crafting_table" not in str(text_obs["surrounding"]):
                    current_task = "Place crafting table."
                    state = 3.5
                elif text_obs["inventory"]["stone_pickaxe"] >= 1 and text_obs["inventory"]["stone_sword"] >= 1:
                    current_task = "Mine iron."
                    state = 5
                elif text_obs["inventory"]["stone_pickaxe"] >= 1 and text_obs["inventory"]["stone_sword"] < 1:
                    current_task = "Craft stone_sword."
                    state = 4.5
                else:
                    current_task = "Craft stone_pickaxe."
            elif state == 4.5: # craft stone_sword
                if text_obs["inventory"]["stone_sword"] >= 1:
                    current_task = "Mine iron."
                    state = 5
                else:
                    current_task = "Craft stone_sword."
            elif state == 5: # mine iron
                
                if text_obs["inventory"]["stone"] > 6 and text_obs["inventory"]["iron"] >= 1 and text_obs["inventory"]["coal"] >= 1 and text_obs["inventory"]["wood"] >= 2:
                    current_task = "Place furnace."
                    state = 6
                else:
                    if cnt < 5:
                        current_task = "Sleep."
                        cnt += 1
                    elif "iron" in str(text_obs["surrounding"]):
                        current_task = "Mine iron."
                    elif "coal" in str(text_obs["surrounding"]):
                        current_task = "Mine coal."
                    elif "tree" in str(text_obs["surrounding"]):
                        current_task = "Chop tree."
                    else:
                        current_task = "Mine stone."
            
            elif state == 6: # place furnace
                if "furnace" in str(text_obs["surrounding"]):
                    current_task = "Place crafting table."
                    state = 6.5
                else:
                    current_task = "Place furnace."
        
            elif state == 6.5: # place crafting table
                if "crafting_table" in str(text_obs["surrounding"]):
                    current_task = "Craft iron_pickaxe."
                    state = 7
                else:
                    current_task = "Place crafting table."
            
            elif state == 7: # Craft iron_pickaxe
                if text_obs["inventory"]["iron_pickaxe"] >= 1:
                    current_task = "Craft iron_sword."
                    state = 7.5
                else:
                    current_task = "Craft iron_pickaxe."
            
            elif state == 7.5: # Craft iron_sword
                if text_obs["inventory"]["iron_sword"] >= 1:
                    current_task = "Mine diamond."
                    state = 8
                else:
                    current_task = "Craft iron_sword."
            
            elif state == 8: # mine diamond
                
                food = text_obs["inner"]["food"]
                drink = text_obs["inner"]["drink"]
                energy = text_obs["inner"]["energy"]
                if text_obs["inventory"]["iron_pickaxe"] >= 1:
                    current_task = "Mine diamond."
                    state = 6
                elif food <= drink and food <= energy and food <= 9:
                    current_task = "Kill the cow."
                elif drink < food and drink <= energy and drink <= 9:
                    current_task = "Drink water."
                elif energy < food and energy < drink and energy <= 9:
                    current_task = "Sleep."
                else:
                    current_task = "Mine diamond."
        
        current_task = str(current_task)
        history_imgs.append(save_img(obs, current_task))
        
        # env step
        obs = env.set_task(obs, [current_task])
        action, _ = agent.act(obs, info=info, deterministic=True, render=True)
        
        # traj["image"].append(obs["policy"]["image"].tolist())
        # traj["task_emb"].append(obs["policy"]["task_emb"].tolist())
        # try:
        #     traj["actions"].append(all_task.index(current_task))
        # except:
        #     traj["actions"].append(4)
        
        obs, r, done, info = env.step(action)
        env_step += 1
        total_reward += 0 #r[0]
        original_rewards += r[0,0,0] #info[0]["original_rewards"][0][0]
    

        if all(done):
            name = "run_results/crafter.gif" if not random else "run_results/crafter-random.gif"
            history_imgs[0].save(
                name, 
                save_all=True, 
                append_images=history_imgs[1:], 
                duration=100, 
                loop=0
            )
            # name = "run_results/buffer/result_{:02d}.json".format(seed)
            # json.dump(traj, open(name, "w"))
            break
    
    env.close()
    
    del agent
    
    return env_step, total_reward, original_rewards


if __name__ == "__main__":

    model = None #"models/crafter_agent-10M-01/"
    # print("model", model)
    
    episode_len1 = []
    episode_len2 = []
    total_reward1 = []
    total_reward2 = []
    original_rewards1 = []
    original_rewards2 = []
    
    begin = time.time()
    for seed in range(1):
        
        env_step1, r1, or1 = render(seed, model, random=False)
        env_step2, r2, or2 = render(seed, model, random=True)
        
        episode_len1.append(env_step1)
        episode_len2.append(env_step2)
        total_reward1.append(r1)
        total_reward2.append(r2)
        original_rewards1.append(or1)
        original_rewards2.append(or2)
    
        # print mean
        print("survival time-step", np.mean(episode_len1), " ", np.mean(episode_len2))
        print("original_rewards", np.mean(original_rewards1), " ", np.mean(original_rewards2))
        print("instruction following rewards", np.mean(total_reward1), " ", np.mean(total_reward2))

    print("model", model)