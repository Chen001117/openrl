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
from openrl.envs.crafter.utils import load_text, postprocess, import_class_from_file

from langchain_openai import ChatOpenAI
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


test = """
Explain: Since the health, food, drink, and energy levels are currently high, the immediate focus should be on preparing for later survival by obtaining resources and tools. The environment has grass and trees, but no immediately useful resources or threats are mentioned. The agent should begin by collecting wood from trees to enable crafting of basic tools.

Plan:
1) The agent needs to collect wood by chopping trees. The wood will be crucial for crafting a wood pickaxe and a wood sword, and later on for creating a crafting table.
2) After collecting enough wood, the agent should craft a wood pickaxe to enable the mining of stone. This is essential for creating stone tools and a furnace.
3) The agent should also craft a wood sword to defend against potential threats such as zombies and skeletons.

Code:
```python
def survival_prep(agent, env, env_info, trajectory):
    loop_counter = 0
    inventory_obs = get_obs(env_info)
    
    # Chop trees until wood is obtained or the loop counter reaches 30.
    while "wood" not in inventory_obs and loop_counter < 30:
        env_info = do("Chop trees.", agent, env, env_info, trajectory)
        inventory_obs = get_obs(env_info)
        loop_counter += 1
        if loop_counter >= 30:
            return env_info
    
    # Craft a wood pickaxe if there is wood in the inventory.
    if "wood" in inventory_obs:
        env_info = do("Craft wood_pickaxe.", agent, env, env_info, trajectory)
        loop_counter += 1

    # Craft a wood sword if there is wood in the inventory.
    if "wood" in inventory_obs and loop_counter < 30:
        env_info = do("Craft wood_sword.", agent, env, env_info, trajectory)
        loop_counter += 1

    return env_info
```

"""

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
    cfg.seed = 42

    # init the agent
    agent = Agent(Net(env, cfg=cfg))
    agent.set_env(env)
    agent.load("models/crafter_agent-10M-28/")

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
    trajectory = {"img" : [], "text" : []}
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
        
        # response = test
        print(response)
        print("="*60)
        
        func_name, code = postprocess(response)
        
        trajectory["text"].append(env_info[-1][0]["text_obs"])
        trajectory["text"].append("Then the agent run the following code:\n```python\n" + code + "\n```\n")
        
        CrafterAgent = import_class_from_file("function/func.py", func_name)
        env_info = CrafterAgent(agent, env, env_info, trajectory)

if __name__ == "__main__":
    render()
