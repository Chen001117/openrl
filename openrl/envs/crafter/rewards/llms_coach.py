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
You are a helpful assistant that tells me the next immediate task to do in Crafter game. \
Here are some tips: \
You have to worry about food, drink, and energy levels when they are low. \
Killing cows and eating plants will increase food level. Tree is not edible. \
Drinking water will increase drink level. \
Sleeping in a safe place (surrounded by blocks) will increase energy. \
Health level will decrease when attacked by monsters. \
Discovering new things, obtaining resources or crafting new tools when food, drink, and energy levels are high. \
Chop Trees to collect Wood. \
Use the wood to create a crafting table. \
Crafting pickaxe and sword near the crafting table. \
The pickaxe allows you to mine stone, while the sword is for attack. \
My ultimate goal is to discover as many diverse things as possible, accomplish as many diverse tasks as possible and become the best Crafter player in the world. \
The task description should start with: sleep, eat, kill, find, drink, place, craft, mine, chop.\n\
Desired format: Reasoning: <reasoning>. Task: <task description>.\n\n"
# Here is an example:\n\
# Reasoning: You need to eat to restore your food level, and the cow is the only food source available. Task: Kill the cow.\n\n"


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

# TASK_SYSTEM_PROMPT = "\
# You are a helpful assistant that tells me the next immediate task to do in Crafter game. \
# My ultimate goal is to discover as many diverse things as possible, \
# accomplish as many diverse tasks as possible and become the best Crafter player in the world. \
# The challenge in Crafter is to survive and unlock achievements by making strategic decisions based on the game's mechanics, which include managing health, food, water, and rest levels. Here are some foundational strategies:\n\
# 1 Forage for food and water immediately. Lakes and cows are primary sources.\n\
# 2 Build shelter by collecting wood and other materials to defend against monsters, especially at night.\n\
# 3 Explore to find different biomes (forests, mountains, caves) that contain unique resources.\n\
# 4 Collect resources like wood, stone, and eventually iron and diamonds, which are critical for crafting tools.\n\
# 5 Start crafting with basic tools (e.g., wood pickaxe) to enable the collection of more advanced materials.\n\
# 6 Progress through the technology tree by crafting advanced tools and objects, like furnaces, for accessing higher-tier resources.\n\
# 7 Build weapons and learn to defend against creatures. Zombies appear at night, and skeletons reside in caves.\n\
# 8 Keep an eye on your health, food, water, and rest levels. Neglecting these can lead to death.\n\
# 9 Plan for the long term by planting saplings for wood and building structures for protection and resource processing.\n\
# Desired format: Reasoning: <reasoning>. Task: <task description>.\n\n"

# TASK_FEW_SHOT = "\
# Here are some example responses:\n\
# Example 1:\n\
# Reasoning: The inventory is empty now, chop down a tree to get some wood. Task: Obtain a wood log.\n\n \n\
# Example 2:\n\
# Reasoning: With an low food level and a nearby cow, it's important to collect food first to avoid hunger depletion. Task: Kill a cow and get some meat.\n\n"
    
COMPLETION_SYSTEM_PROMPT = "\
You are a helpful assistant that tells me whether the given task in Crafter game has been completed. \
Desired format: Completion Criteria: <reasoning>. Answer: yes or no.\n\n\
Here is an example:\n\
Completion Criteria: The task's completion would be indicated by an increase in the drink property, as the objective involves consuming water to address thirst. Answer: no.\n\n"
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
        self.n_env = 2
        
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
        
        self._completed_tasks = []
        self._num_few_shot = 5
        self._max_learned_tasks = 100

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
                text_state = "Currently, " + info["text_obs"] + " "
                question = task2do + last_state + text_state + self._complete_question
                
                prompt1 = {"role": "system", "content": self._complete_system}
                prompt2 = {"role": "user", "content": COMPLETE_FEW_SHOT}
                prompt3 = {"role": "assistant", "content": "Completion Criteria: The task's completion would be indicated by an increase in the drink property, as the objective involves consuming water to address thirst. Answer: yes.\n\n"}     
                prompt4 = {"role": "user", "content": question}
                prompts.append([prompt1, prompt2, prompt3, prompt4])
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
                if len(self._completed_tasks) > 0:
                    # learned_task = sample(self._completed_tasks, min(len(self._completed_tasks), self._num_few_shot))
                    # learned_task = " ".join(t for t in learned_task)
                    # learned_task_prefix = "Here are some tasks I have learned so far:"
                    system_content = self._task_system
                else:
                    system_content = self._task_system
                prompt1 = {"role": "system", "content": system_content}
                prompt2 = {"role": "user", "content": "You see tree. You have nothing in your inventory. Your health level is high, food level is high, drink level is high, energy is high. What do you do?"}
                prompt3 = {"role": "assistant", "content": "Reasoning: The inventory is empty now, chop down a tree to get some wood. Task: Obtain a wood log.\n\n"}
                prompt4 = {"role": "user", "content": user_content}
                # print("task", user_content)
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
                    self._completed_tasks.append(self._last_task[true_idx])
                    if len(self._completed_tasks) > self._max_learned_tasks:
                        self._completed_tasks.pop(0)
            self._last_task[true_idx] = task_response
            self._task_num_try[true_idx] = 0
                
        # update last state
        for idx, info in enumerate(infos):
            self._last_state[idx] = info["text_obs"]
        
        rewards = np.array(rewards) * 1.
        
        self._step_cnt = (self._step_cnt + 1) % self.reset_freq
        
        infos = [{"task": task} for task in self._last_task]
        
        return rewards, infos
