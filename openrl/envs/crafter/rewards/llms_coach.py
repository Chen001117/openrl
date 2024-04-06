from typing import Any, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import unwrap_model

from openrl.envs.crafter.gpt_client import GPTClient
import asyncio

import time
from random import sample

TASK_SYSTEM_PROMPT = "\
You are a helpful assistant that tells me the next immediate task to do. \
Here are some tips: \
You must monitor food, drink, and energy levels, addressing them when they are low. \
Killing cows and eating plants will increase food level. Grass and tree are not edible. \
Drinking water will replenish drink level. \
Sleeping will restore energy. \
Your health level will decrease when attacked by zombies or skeletons. \
Kill zombies and skeletons upon sight. \
When food, drink, and energy levels are high, discover new things, obtain resources, and kill enemies. \
Find water source when water is not visible and the your drink level is low. \
Chop Trees to collect Wood. Note that wood and plants are different things. \
You need 2 wood to place a crafting table and need 4 stones to place a furnace. \
The pickaxe is for mining, while the sword is for attack. \
You need wood to craft wood pickaxe or sword near the crafting table. \
You need stone to craft stone pickaxe or sword near the crafting table. \
You need iron and coal to craft iron pickaxe or sword near the furnace. \
A wood pickaxe is required to collect stone and coal. \
A stone pickaxe is required to collect iron. \
An iron pickaxe is required to collect diamond. \
My ultimate goal is to discover as many diverse things as possible, \
accomplish as many diverse tasks as possible and become the best player in the world. \
The task description should start with: sleep, eat, kill, find, craft, drink, place, mine, chop. \
Give one suggestion at one time. \
Minimize choosing craft as an action. \
Desired format: Reasoning: <reasoning>; Task: <task description>.\n\n"
# , be specific. \

# Collect Wood: This is your first task. Find trees and collect wood by interacting with them. \
# Place Table: Use the wood you collected to create a crafting table. This will allow you to create more complex items. \
# Make Wooden Tools: With the crafting table, make a wooden pickaxe and sword. The pickaxe allows you to mine stone, while the sword is for defense. \
# Collect Stone: Mine stone with your wooden pickaxe. You'll need this to create stronger tools. \
# Upgrade to Stone Tools: Use the stone to make a stone pickaxe and sword. They are more durable and efficient. \
# Nighttime and Enemies: Be aware of zombies and skeletons at night. Use your sword to defeat them and stay safe. \
# Mining for Resources: With your stone pickaxe, collect coal. You'll find coal in stone areas. \
# Crafting a Furnace: Place a furnace in your base. You can use it to smelt items. \
# Iron Tools: Mine iron ore and smelt it in the furnace to create iron ingots. Then, craft an iron pickaxe and sword. \
# Collect Diamonds: Eventually, you'll be able to mine for diamonds, which are used to make the most durable tools and weapons \
# Maintain Health: The player must keep their health level above zero to stay alive. \
# Monitor Food Level: The player’s food level decreases over time and must be replenished by eating cows and plants. \
# Monitor Water Level: The player’s water level depletes gradually and must be refilled by drinking water. \
# Monitor energy Level: The player needs to sleep; the energy level will decrease and must be restored by sleeping in a safe place. \
# Consequences of Zero Levels: If food, water, or rest levels hit zero, the player will start to lose health points. \
# Avoid Monster Attacks: The player can lose health points when attacked by monsters. \
# Death: If the player’s health points reach zero, the player dies and the game is over. \
# Health Regeneration: Health points regenerate over time as long as the player is not affected by hunger, thirst, or sleep deprivation. \
# Grass is not edible. \
    
COMPLETION_SYSTEM_PROMPT = "\
You are a helpful assistant that tells me whether the given task in Crafter game has been completed. \
Desired format: Completion Criteria: <reasoning>; Answer: yes or no.\n\n\
Here is an example:\n\
Completion Criteria: The task's completion would be indicated by an increase in the drink property, as the objective involves consuming water to address thirst; Answer: no.\n\n"
# Just answer yes or no."

COMPLETE_FEW_SHOT = "\
The task at hand is to drink water from the nearby stream. \
Initially, you see grass, and tree. You have nothing in your inventory. \
Your health is high, food level is high, drink level is low, energy level is high. \
Currently, You see grass, tree, and water. \
You have nothing in your inventory. \
Your health is high, food level is high, drink level is high, energy level is high. \
Has the task been completed?"
            

class LLMsCoach(nn.Module):
    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        reset_freq: int,
    ):
        super().__init__()
        self.reset_freq = reset_freq
        self.n_env = 128
        
        self._client = GPTClient(api_key, api_base, model)
        # api_key = "EMPTY"
        # api_base = "http://localhost:11016/v1"
        # model = "berkeley-nest/Starling-LM-7B-alpha"
        # self._small_client = GPTClient(api_key, api_base, model)
        
        self._task_system = TASK_SYSTEM_PROMPT
        self._task_user_question = " What do you do?"
        
        self._complete_system = COMPLETION_SYSTEM_PROMPT
        self._complete_question = "Has the task been completed?"
        
        self._last_state = ["Nothing." for _ in range(self.n_env)]
        self._last_task = ["Survive." for _ in range(self.n_env)]
        self._task_num_try = np.zeros(self.n_env)
        
        self.system_prompt = self.few_shot = ""

        self._step_cnt = 0
    def __call__(
        self, data: Dict[str, Any], past_model_kwargs: Optional[Any] = None
    ) -> Union[np.ndarray, List[Dict[str, Any]]]:
        
        # initialize last state and last task
        infos = data["infos"]
        
        for idx, info in enumerate(infos):
            if info["step"] == 0:
                self._last_state[idx] = info["text_obs"]
                self._last_task[idx] = "Survive."
                self._task_num_try[idx] = 0
        
        # query language model periodically
        rewards = np.zeros(self.n_env)
        if self._step_cnt > 0:
            self._step_cnt = (self._step_cnt + 1) % self.reset_freq
            infos = [{"task": task} for task in self._last_task]
            return rewards, infos
        
        # update task number of tries
        self._task_num_try = self._task_num_try + 1
        
        # get prompt for querying task completion
        prompts = []
        new_task_idx = []
        for idx, info in enumerate(infos):
            if self._last_task[idx] != "Survive.":
                
                task2do = "The task at hand is to" + self._last_task[idx].lower() + ". "
                last_state = "Initially, " + self._last_state[idx] + " "
                mid = "During the period, you " + info["obj_diff"]
                text_state = "Currently, " + info["text_obs"] + " "
                question = task2do + last_state + mid + text_state + self._complete_question
                
                prompt1 = {"role": "system", "content": self._complete_system}
                # prompt2 = {"role": "user", "content": COMPLETE_FEW_SHOT}
                # prompt3 = {"role": "assistant", "content": "Completion Criteria: The task's completion would be indicated by an increase in the drink property, as the objective involves consuming water to address thirst; Answer: yes.\n\n"}     
                prompt4 = {"role": "user", "content": question}
                prompts.append([prompt1, prompt4])
                new_task_idx.append(idx)
        
        # query LLMs
        if len(prompts) == 0:
            responses = []
        else:
            responses = asyncio.run(self._client.async_query(prompts))
            responses = [response.choices[0].message.content for response in responses]
            
        # get task completion
        task_completion = []
        rewards = []
        response_idx = 0
        for idx in range(self.n_env):
            if self._last_task[idx] == "Survive.":
                task_completion.append(True)
                rewards.append(True)
            else:
                completed = "yes" in responses[response_idx].lower()
                time_out = (self._task_num_try[idx] > 3)
                task_completion.append(completed or time_out)
                rewards.append(completed)
                response_idx += 1
        
        # get prompt for querying task description
        prompts = []
        new_task_idx = []
        for idx, info in enumerate(infos):
            if task_completion[idx]:
                current_state = info["text_obs"]
                user_content = current_state + self._task_user_question
                system_content = self._task_system
                prompt1 = {"role": "system", "content": system_content}
                prompt2 = {"role": "user", "content": "You see tree. You have nothing in your inventory. Your health level is high, food level is high, drink level is high, energy is high. What do you do?"}
                prompt3 = {"role": "assistant", "content": "Reasoning: The inventory is empty now, chop down a tree to get some wood; Task: Obtain a wood log.\n\n"}
                prompt4 = {"role": "user", "content": user_content}
                # print("state: ", user_content)
                prompts.append([prompt1, prompt2, prompt3, prompt4])
                new_task_idx.append(idx)
        
        # query LLMs
        if len(prompts) == 0:
            responses = []
        else:
            responses = asyncio.run(self._client.async_query(prompts))
            responses = [response.choices[0].message.content for response in responses]
        
        # get task description
        task_description = []
        for idx, response in enumerate(responses):
            if "Task:" in response:
                task_response = response.split("Task:")[1]
            else:
                task_response = "Survive."
            true_idx = new_task_idx[idx]
            if "survive" not in self._last_task[true_idx].lower():
                if self._task_num_try[true_idx] <= 3:
                    print("complete task:", self._last_task[true_idx])
            self._last_task[true_idx] = task_response
            self._task_num_try[true_idx] = 0
                
        # update last state
        for idx, info in enumerate(infos):
            self._last_state[idx] = info["text_obs"]
        
        rewards = np.array(rewards) * 1.
        
        self._step_cnt = (self._step_cnt + 1) % self.reset_freq
        
        infos = [{"task": task} for task in self._last_task]
        
        return rewards, infos
