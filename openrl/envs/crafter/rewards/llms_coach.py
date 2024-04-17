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
The task description should start with: sleep, eat, kill, find, craft, drink, place, mine, chop, get. \
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
Here is an example: \
Completion Criteria: The task's completion would be indicated by an increase in the drink property, as the objective involves consuming water to address thirst; Answer: no.\n\n"
# Just answer yes or no."
# Desired format: yes or no."

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
        update_task_freq: int,
    ):
        super().__init__()
        self.n_env = 128
        
        # self._client = GPTClient(api_key, api_base, model)
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

        self.update_task_cnt = 0
        self.update_task_freq = update_task_freq
        self.num_try_query_task = 4
        
        self.task_cnt = dict()
           
    def get_rewards(self, obj_diff, current_task):
        
        if current_task == "Survive.":
            return True
        elif current_task == "Chop tree.":
            suc = "gained 1 wood" in obj_diff
            suc |= "gained 2 wood" in obj_diff
            suc |= "gained 3 wood" in obj_diff 
            suc |= "gained 4 wood" in obj_diff
            return suc 
        elif current_task == "Kill the cow.": 
            suc = "gained 1 food" in obj_diff
            suc |= "gained 2 food" in obj_diff
            suc |= "gained 3 food" in obj_diff
            suc |= "gained 4 food" in obj_diff
            suc &= "killed a cow" in obj_diff
            return suc 
        elif current_task == "Mine stone.":
            gain_stone_num = obj_diff.count("gained 1 stone")
            place_stone_num = obj_diff.count("lost 1 stone")
            suc = gain_stone_num > place_stone_num
            return suc 
        elif current_task == "Drink water.":
            suc = "gained 1 drink" in obj_diff
            suc |= "gained 2 drink" in obj_diff
            suc |= "gained 3 drink" in obj_diff
            suc |= "gained 4 drink" in obj_diff
            return suc 
        elif current_task == "Mine coal.":
            suc = "gained 1 coal" in obj_diff
            suc |= "gained 2 coal" in obj_diff
            suc |= "gained 3 coal" in obj_diff
            suc |= "gained 4 coal" in obj_diff
            return suc 
        elif current_task == "Mine iron.":
            suc = "gained 1 iron" in obj_diff
            suc |= "gained 2 iron" in obj_diff
            suc |= "gained 3 iron" in obj_diff
            suc |= "gained 4 iron" in obj_diff
            return suc 
        elif current_task == "Mine diamond.":
            suc = "gained 1 diamond" in obj_diff
            suc |= "gained 2 diamond" in obj_diff
            suc |= "gained 3 diamond" in obj_diff
            suc |= "gained 4 diamond" in obj_diff
            return suc 
        elif current_task == "Kill the zombie.":
            suc = "killed a zombie" in obj_diff
            return suc 
        elif current_task == "Kill the skeleton.":
            suc = "killed a skeleton" in obj_diff
            return suc 
        elif current_task == "Craft wood_pickaxe.":
            suc = "gained 1 wood_pickaxe" in obj_diff
            suc |= "gained 2 wood_pickaxe" in obj_diff
            suc |= "gained 3 wood_pickaxe" in obj_diff
            suc |= "gained 4 wood_pickaxe" in obj_diff
            return suc 
        elif current_task == "Craft wood_sword.":
            suc = "gained 1 wood_sword" in obj_diff
            suc |= "gained 2 wood_sword" in obj_diff
            suc |= "gained 3 wood_sword" in obj_diff
            suc |= "gained 4 wood_sword" in obj_diff
            return suc 
        elif current_task == "Place crafting table.":
            suc = "found crafting table" in obj_diff
            return suc 
        elif current_task == "Craft stone_pickaxe.":
            suc = "gained 1 stone_pickaxe" in obj_diff
            suc |= "gained 2 stone_pickaxe" in obj_diff
            suc |= "gained 3 stone_pickaxe" in obj_diff
            suc |= "gained 4 stone_pickaxe" in obj_diff
            return suc 
        elif current_task == "Craft stone_sword.":
            suc = "gained 1 stone_sword" in obj_diff
            suc |= "gained 2 stone_sword" in obj_diff
            suc |= "gained 3 stone_sword" in obj_diff
            suc |= "gained 4 stone_sword" in obj_diff
            return suc 
        elif current_task == "Craft iron_pickaxe.":
            suc = "gained 1 iron_pickaxe" in obj_diff
            suc |= "gained 2 iron_pickaxe" in obj_diff
            suc |= "gained 3 iron_pickaxe" in obj_diff
            suc |= "gained 4 iron_pickaxe" in obj_diff
            return suc 
        elif current_task == "Craft iron_sword.":
            suc = "gained 1 iron_sword" in obj_diff
            suc |= "gained 2 iron_sword" in obj_diff
            suc |= "gained 3 iron_sword" in obj_diff
            suc |= "gained 4 iron_sword" in obj_diff
            return suc 
        elif current_task == "Find cows.":
            suc = "found cow" in obj_diff
            return suc 
        elif current_task == "Find water.":
            suc = "found water" in obj_diff
            return suc 
        elif current_task == "Sleep.":
            suc = "fell asleep" in obj_diff
            return suc
        elif current_task == "Place furnace.": 
            suc = "found furnace" in obj_diff
            return suc
        elif current_task == "Find tree.":
            suc = "found tree" in obj_diff
            return suc
        elif current_task == "Find stone.":
            suc = "found stone" in obj_diff
            return suc
        
        raise ValueError("Task {} not recognized.".format(current_task))
        
     
    def __call__(
        self, data: Dict[str, Any], past_model_kwargs: Optional[Any] = None
    ) -> Union[np.ndarray, List[Dict[str, Any]]]:
        
        infos = data["infos"]
        
        if "task" in data:
            # raise ValueError("Task should not be in data.")
            self._last_task = data["task"]
        
        # set task to survive for new episodes
        for idx, info in enumerate(infos):
            if info["step"] == 0:
                self._last_state[idx] = info["text_obs"]
                self._last_task[idx] = "Survive."
                self._task_num_try[idx] = 1
        
        # query language model periodically
        rewards = np.zeros(self.n_env)
        if self.update_task_cnt > 0:
            self.update_task_cnt = (self.update_task_cnt + 1) % self.update_task_freq
            new_infos = [{"task": task} for task in self._last_task]
            return rewards, new_infos
        self.update_task_cnt = (self.update_task_freq + 1) % self.update_task_freq
        
        # update task number of tries
        self._task_num_try = self._task_num_try - 1
        
        if False:
            # get prompt for querying task completion
            prompts = []
            new_task_idx = []
            for idx, info in enumerate(infos):
                if self._last_task[idx] != "Survive.":
                    
                    transi = "During the period, " + info["obj_diff"]
                    transi = "During the period, nothing has changed." if transi == "During the period, " else transi
                    transi = "You are sleeping." if transi == "During the period,  You are sleeping." else transi
                    
                    prefix = "You are a helpful assistant that tells me whether the given task in Crafter game has been completed. "
                    prefix += "Drinking water will replenish drink level. "
                    prefix += "Killing cows will increase food level. "
                    prefix += "Choping trees will gained wood. "
                    prefix += "Desired format: Completion Criteria: <reasoning>. Answer: <yes/no>.\n\n"
                    prefix += " The task at hand is to chop tree. During the period, you gained 1 wood. 1 tree disappear."
                    prefix += " Has the task been completed?"
                    few_shot = "Completion Criteria: The task's completion depends on the successful chopping of a tree and acquiring the wood; Answer: yes.\n\n"
                    
                    question = "The task at hand is to " + self._last_task[idx].lower() + " " 
                    question += transi + " Has the task been completed?"

                    prompt1 = {"role": "user", "content": prefix}
                    prompt2 = {"role": "assistant", "content": few_shot}
                    prompt3 = {"role": "user", "content": question}
                    prompts.append([prompt1, prompt2, prompt3])
        
            # query LLMs
            if len(prompts) == 0:
                responses = []
            else:
                responses = asyncio.run(self._client.async_query(prompts))
                responses = [response.choices[0].message.content for response in responses]
            
            # get task completion
            need_new_task = []
            rewards = []
            response_idx = 0
            for idx in range(self.n_env):
                if self._last_task[idx] == "Survive.":
                    need_new_task.append(True)
                    rewards.append(True)
                else:
                    completed = "answer: yes" in responses[response_idx].lower()
                    need_new_task.append(completed or self._task_num_try[idx] == 0)
                    rewards.append(completed)
                    response_idx += 1
                    if completed:
                        print("completed task name: ", self._last_task[idx])
        
        else:
            
            rewards = []
            need_new_task = []
            for idx, info in enumerate(infos):
                if self._last_task[idx] == "Survive.":
                    need_new_task.append(True)
                    rewards.append(True)
                else:
                    obj_diff = info["obj_diff"]
                    cur_task = self._last_task[idx]
                    completed = self.get_rewards(obj_diff, cur_task)
                    need_new_task.append(completed or self._task_num_try[idx] == 0)
                    rewards.append(completed)
                    if completed:
                        if cur_task not in self.task_cnt:
                            self.task_cnt[cur_task] = 1
                        else:
                            self.task_cnt[cur_task] += 1
                            
            if np.random.rand() < 0.001:
                print("completed task name: ", self.task_cnt)
        
        if False: #all(need_new_task):
        
            # get prompt for querying task description
            prompts = []
            new_task_idx = []
            for idx, info in enumerate(infos):
                if need_new_task[idx]:
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
            for idx, response in enumerate(responses):
                if "Task:" in response:
                    task_response = response.split("Task:")[1]
                else:
                    task_response = "Survive."
                true_idx = new_task_idx[idx]
                self._last_task[true_idx] = task_response
                self._task_num_try[true_idx] = self.num_try_query_task
                
        else:
            
            for idx, info in enumerate(infos):
                if need_new_task[idx]:
                    task_response = info["next_task"]
                    self._last_task[idx] = task_response
                    self._task_num_try[idx] = 1
        
        # update last state
        for idx, info in enumerate(infos):
            self._last_state[idx] = info["text_obs"]
        
        rewards = np.array(rewards) * 4.
        
        new_infos = []
        for i in range(self.n_env):
            env_dict = {
                "task": self._last_task[i],
                "instruction_following_rewards": rewards[i],
            }
            new_infos.append(env_dict)
        
        return rewards, new_infos
