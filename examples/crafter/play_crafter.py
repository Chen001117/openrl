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

from openrl.envs.crafter.gpt_client import GPTClient
import asyncio

from PIL import Image, ImageDraw, ImageFont


COMPLETION_SYSTEM_PROMPT = "\
You are a helpful assistant that tells me whether the given task in Crafter game has been completed. \
Desired format: Completion Criteria: <reasoning>. Answer: yes or no.\n\n\
Here is an example:\n\
Completion Criteria: The task's completion would be indicated by an increase in the drink property, as the objective involves consuming water to address thirst; Answer: no.\n\n"

COMPLETE_FEW_SHOT = "\
The task at hand is to drink water from the nearby stream. \
Initially, you see grass, and tree. You have nothing in your inventory. \
Your health is high, food level is high, drink level is low, energy level is high. \
Currently, You see grass, tree, and water. \
You have nothing in your inventory. \
Your health is high, food level is high, drink level is high, energy level is high. \
Has the task been completed?"

def save_img(obs, task, idx=0):
    img = obs["policy"]["image"][0, 0]
    img = img.transpose((1, 2, 0))
    img = Image.fromarray(img)
    img = img.resize((256, 256))
    draw = ImageDraw.Draw(img)
    draw.text((10,10), task, fill=(255,0,0))
    img.save("run_results/image{:03d}.png".format(idx))
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
    agent.load("crafter_agent-2M/")

    # begin to test
    trajectory = []
    obs, info = env.reset()
    step = 0
    
    print("Enter the task: ", end="")
    current_task = input()
    last_obs = info[0]["text_obs"]
    print("")

    img = save_img(obs, current_task)
    trajectory.append(img)
    
    api_key = "EMPTY" #"isQQQqPJUUSWXvz4NqG36Q6v5pxdPTkG",
    api_base = "http://localhost:11017/v1" # "https://azure-openai-api.shenmishajing.workers.dev/v1", 
    model = "meta-llama/Llama-2-70b-chat-hf" #"berkeley-nest/Starling-LM-7B-alpha" #"gpt-4-32k",
    client = GPTClient(api_key, api_base, model)
    
    query = 0
    ex = 0
    
    while True:
        
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action, given_task=[current_task])
        step += 1
        
        query = (query + 1) % 32
        if query == 0:
        
            task2do = "The task at hand is to " + current_task + ". "
            last_state = "Initially, " + last_obs + " "
            text_state = "Currently, " + info[0]["text_obs"] + " "
            mid = "During the period, you " + info[0]["obj_diff"]
            question = task2do + last_state + mid + text_state + "Has the task been completed?"
                    
            prompt1 = {"role": "system", "content": COMPLETION_SYSTEM_PROMPT}
            prompt2 = {"role": "user", "content": COMPLETE_FEW_SHOT}
            
            prompt3 = {"role": "assistant", "content": "Completion Criteria: The task's completion would be indicated by an increase in the drink property, as the objective involves consuming water to address thirst; Answer: yes.\n\n"}     
            prompt4 = {"role": "user", "content": question}
            prompts = [[prompt1, prompt2, prompt3, prompt4]]
            
            responses = asyncio.run(client.async_query(prompts))
            responses = responses[0].choices[0].message.content
            
            print("Response: ", question + responses)
            
            print("Enter the task:", end="")
            current_task = input()
            last_state = info[0]["text_obs"]
                
            img = save_img(obs, current_task, step)
            trajectory.append(img)

        if all(done):
            break
        
    print("step", step)

    # save the trajectory to gif
    trajectory[0].save(
        "run_results/crafter.gif", 
        save_all=True, 
        append_images=trajectory[1:], 
        duration=100, 
        loop=0
    )
    # import imageio
    # imageio.mimsave("run_results/crafter.gif", trajectory, duration=0.01)

    env.close()


if __name__ == "__main__":
    render()
