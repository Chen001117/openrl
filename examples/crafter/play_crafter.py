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
    agent.load("crafter_agent-100M-2/")

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
    
    from openrl.envs.crafter.gpt_client import GPTClient
    import asyncio
    api_key = "EMPTY" #"isQQQqPJUUSWXvz4NqG36Q6v5pxdPTkG",
    api_base = "http://localhost:11016/v1" # "https://azure-openai-api.shenmishajing.workers.dev/v1", 
    model = "mistralai/Mistral-7B-Instruct-v0.2" #"berkeley-nest/Starling-LM-7B-alpha" #"gpt-4-32k",
    client = GPTClient(api_key, api_base, model)
    
    query = 0
    ex = 0
    
    while True:
        
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action, given_task=[current_task])
        step += 1
        
        # if end with .
        if current_task[-1] == ".":
            
            transi = "During the period, " + info[0]["obj_diff"]
            transi = "During the period, nothing has changed." if transi == "During the period, " else transi
            transi = "You are sleeping." if transi == "During the period,  You are sleeping." else transi
            
            prefix = "You are a helpful assistant that tells me whether the given task in Crafter game has been completed. Be concise and deterministic."
            prefix += "Desired format: Completion Criteria: <reasoning>. Answer: <yes/no>.\n\n"
            prefix += " The task at hand is to chop tree. During the period, you get 1 wood. 1 tree disappear."
            prefix += " Has the task been completed?"
            few_shot_a = "Completion Criteria: The task's completion depends on the successful chopping of a tree and acquiring the wood; Answer: yes.\n\n"
            question = "The task at hand is to " + current_task + ". " 
            question += transi + " Has the task been completed?"
            
            prompt1 = {"role": "user", "content": prefix}
            prompt2 = {"role": "assistant", "content": few_shot_a}
            prompt3 = {"role": "user", "content": question}
            prompts = [[prompt1, prompt2, prompt3]]
            
            responses = asyncio.run(client.async_query(prompts))
            responses = responses[0].choices[0].message.content
            
            print("prompts: ", prompts)
            print("Response: ", responses)
            
        img = save_img(obs, current_task, step)
        trajectory.append(img)
        
        print("Enter the task:", end="")
        input_text = input()
        current_task = input_text if input_text != "" else current_task
        if current_task == "exit":
            break
        last_state = info[0]["text_obs"]
                

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
