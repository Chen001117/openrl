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


import time

import imageio
import numpy as np

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.envs.wrappers import GIFWrapper
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent
from openrl.envs.crafter.utils import load_text, postprocess

from gpt_env import GPTEnv
from gpt_agent import GPTAgent

from langchain_openai import ChatOpenAI
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

def render(seed):
    
    # begin to test
    env = make(
        "Crafter",
        render_mode="human",
        asynchronous=False,
        env_num=1,
    )

    # config
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()
    cfg.seed = seed * 11 + 1117

    # init the agent
    agent = Agent(Net(env, cfg=cfg))
    agent.set_env(env)
    agent.load("../models/crafter_agent-10M-50/")

    # GPT client
    from openrl.envs.crafter.gpt_client import GPTClient
    import asyncio
    api_key = "EMPTY" 
    api_base = "http://localhost:11016/v1" 
    model = "meta-llama/Meta-Llama-3-8B-Instruct" 

    # gpt client
    llm = ChatOpenAI(
        openai_api_key = api_key,
        openai_api_base = api_base,
        model_name = model,
        temperature = 0.,
        stop= ["<|eot_id|>", "end of the code."]
    )
    
    # env 
    env = GPTEnv(env)
    agent = GPTAgent(env, agent)

    # begin to test
    text_obs, dict_obs = env.reset()
    
    # retry variables
    code, error_msg = None, None
    retry_time = 0
    
    # logger
    total_cost = 0.
    trajectory = [text_obs]
    response_history = []
            
    # system_message_prompt
    system_template = load_text("prompt/action.txt")
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_template
    )
    beginner_tutorial = load_text("prompt/beginner_tutorial.txt")
    programs = load_text("prompt/programs.txt")
    text_obs_format = load_text("prompt/text_obs_format.txt")
    response_requirement = load_text("prompt/response_requirement.txt")
    response_format = load_text("prompt/response_format.txt")
    system_message = system_message_prompt.format(
        beginner_tutorial=beginner_tutorial,
        programs=programs, 
        text_obs_format=text_obs_format,
        response_requirement=response_requirement,
        response_format=response_format
    )
    
    while True:

        
        # human_message_prompt
        human_message = env.infos[0]["text_obs"]
        if retry_time > 0:
            human_message += f"\nCode from the last round:\n{code}"
            human_message += f"\nExecution error:\n{error_msg}"
        human_message = HumanMessage(content=human_message)
        
        try:
            
            # query gpt4
            # response = AIMessage(
            #     content=load_text("prompt/test.txt"),
            #     response_metadata = {'token_usage': {'prompt_tokens': 0, 'completion_tokens': 0}}
            # )
            response = llm([system_message, human_message])
            response_history.append(response.content)
            
            # cost
            total_cost += response.response_metadata['token_usage']['prompt_tokens'] * 3e-5
            total_cost += response.response_metadata['token_usage']['completion_tokens'] * 6e-5
            
            # implement the function
            func_name = "function/func_{:02d}.py".format(seed)
            gpt_function, code = postprocess(response.content, func_name)
            env.reset_query_counter()
            agent.code = code # for trajectory saving
            env, agent = gpt_function(env, agent, env.infos[0]["dict_obs"])
            assert env.gpt_counter > 0, "environment is not updated." # for llama
            retry_time = 0 
            
        except Exception as error:
            
            print("Error:", error)
            error_msg = error
            
            retry_time += 1
            if retry_time > 3:
                break
        
        # exit if done
        if env.is_done:
            break

    base_dir = "../run_results/gpt_result"
    # save trajectory
    trajectory_dir = base_dir + "/trajectory/{:02d}.json".format(seed)
    agent.save_trajectory(trajectory_dir)
    # # save obs
    # obs_dir = base_dir + "/obs/{:02d}.json".format(seed)
    # env.save_obs(obs_dir)
    # save gif
    gif_dir = base_dir + "/visualization/{:02d}.gif".format(seed)
    env.save_gif(gif_dir)
    # save response history
    response_dir = base_dir + "/response/{:02d}_{:.2f}.txt".format(seed, env.total_reward)
    with open(response_dir, "w") as f:
        for idx, response in enumerate(response_history):
            f.write(response + "\n")
            f.write("=" * 60 + "\n")
    # print cost
    print("Total cost:", total_cost)

if __name__ == "__main__":
    render(1)