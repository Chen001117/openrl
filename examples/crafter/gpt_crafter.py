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

from langchain_openai import ChatOpenAI
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import importlib.util

def load_text(fpaths, by_lines=False):
    with open(fpaths, "r") as fp:
        if by_lines:
            return fp.readlines()
        else:
            return fp.read()

def get_code_from_response(response):
    response = response.split("```python\n")[1]
    response = response.split("```")[0]
    return response

def func_name_from_code(code):
    code = code.split("def ")[1]
    code = code.split(":")[0]
    return code

def get_obs():
    return info[0]["text_obs"]

        
prefix = """
from PIL import Image, ImageDraw, ImageFont

def save_img(obs, task):
    img = obs["policy"]["image"][0, 0]
    img = img.transpose((1, 2, 0))
    img = Image.fromarray(img)
    img = img.resize((256, 256))
    draw = ImageDraw.Draw(img)
    draw.text((10,10), task, fill=(255,0,0))
    img.save("run_results/image.png")
    return img

def do(language, agent, env, info, trajectory):
    current_task = language
    action, _ = agent.act(info[0], deterministic=True)
    obs, r, done, info = env.step(action, given_task=[current_task])
    img = save_img(obs, current_task)
    trajectory.append(img)
    if all(done):
        trajectory[0].save(
            "run_results/crafter.gif", 
            save_all=True, 
            append_images=trajectory[1:], 
            duration=100, 
            loop=0
        )
        exit()
    return (obs, r, done, info)

def get_obs(info):
    return info[-1][0]

"""

def postprocess(code):
    
    name = func_name_from_code(code)
    all_code = prefix + code
    
    with open("function/func.py", "w") as text_file:
        text_file.write(all_code)

def import_class_from_file(file_path, function_name):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, function_name)
    return function

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
    agent.set_env(env)
    agent.load("crafter_agent-10M-3/")

    # GPT client
    from openrl.envs.crafter.gpt_client import GPTClient
    import asyncio
    api_key = "EMPTY" 
    api_base = "http://localhost:11016/v1"  
    model = "mistralai/Mistral-7B-Instruct-v0.2" 

    # gpt client
    llm = ChatOpenAI(
        openai_api_key = api_key,
        openai_api_base = api_base,
        model_name=model,
    )

    # begin to test
    trajectory = []
    obs, infos = env.reset()

    step = 0

    env_info = (obs, None, None, infos)
    while True:
        
        system_template = load_text("prompt/action.txt")
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )
        response_template = load_text("prompt/response.txt")
        programs = load_text("prompt/do.txt")
        system_message = system_message_prompt.format(
            programs=programs, response_format=response_template
        )
        human_message = HumanMessage(content=env_info[-1][0]["text_obs"])
        
        response = llm([system_message, human_message]).content
        
        print(response)
        
        print("="*60)
        
        code = get_code_from_response(response)
        
        postprocess(code)
        
        CrafterAgent = import_class_from_file("function/func.py", "CrafterAgent")
        env_info = CrafterAgent(agent, env, env_info, trajectory)
        

if __name__ == "__main__":
    render()