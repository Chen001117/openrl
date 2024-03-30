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

TASK_SYSTEM_PROMPT = "\
You are a helpful assistant that tells me the next immediate task to do in Crafter game. \
My ultimate goal is to discover as many diverse things as possible, \
accomplish as many diverse tasks as possible and become the best Crafter player in the world. \
The challenge in Crafter is to survive and unlock achievements by making strategic decisions based on the game's mechanics, which include managing health, food, water, and rest levels. Here are some foundational strategies:\n\
1 Forage for food and water immediately. Lakes and cows are primary sources.\n\
2 Build shelter by collecting wood and other materials to defend against monsters, especially at night.\n\
3 Explore to find different biomes (forests, mountains, caves) that contain unique resources.\n\
4 Collect resources like wood, stone, and eventually iron and diamonds, which are critical for crafting tools.\n\
5 Start crafting with basic tools (e.g., wood pickaxe) to enable the collection of more advanced materials.\n\
6 Progress through the technology tree by crafting advanced tools and objects, like furnaces, for accessing higher-tier resources.\n\
7 Build weapons and learn to defend against creatures. Zombies appear at night, and skeletons reside in caves.\n\
8 Keep an eye on your health, food, water, and rest levels. Neglecting these can lead to death.\n\
9 Plan for the long term by planting saplings for wood and building structures for protection and resource processing.\n\
Desired format: Reasoning: <reasoning>. Task: <task description>.\n\n"

# TASK_FEW_SHOT = "\
# Here are some example responses:\n\
# Example 1:\n\
# Reasoning: The inventory is empty now, chop down a tree to get some wood. Task: Obtain a wood log.\n\n \n\
# Example 2:\n\
# Reasoning: With an low food level and a nearby cow, it's important to collect food first to avoid hunger depletion. Task: Kill a cow and get some meat.\n\n"
    
COMPLETION_SYSTEM_PROMPT = "You are a helpful assistant that tells me whether the given task has been completed. \
Just answer yes or no."

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
        api_key = "EMPTY"
        api_base = "http://localhost:11016/v1"
        model = "berkeley-nest/Starling-LM-7B-alpha"
        self._small_client = GPTClient(api_key, api_base, model)
        
        self._task_system = TASK_SYSTEM_PROMPT
        self._task_user_prefix = "The current state is "
        self._task_user_question = "\nWhat do you do?\n"
        
        self._complete_system = COMPLETION_SYSTEM_PROMPT
        self._complete_question = "Has the task been completed?"
        
        self._last_state = ["Nothing" for _ in range(self.n_env)]
        self._last_task = ["Survive" for _ in range(self.n_env)]
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
                self._last_task[idx] = "Survive"
                self._task_num_try[idx] = 0
        
        # query language model periodically
        rewards = np.zeros(self.n_env)
        if self._step_cnt > 0:
            self._step_cnt = (self._step_cnt + 1) % self.reset_freq
            return rewards, []
        
        # update task number of tries
        self._task_num_try = self._task_num_try + 1
        
        # get prompt for querying task completion
        prompts = []
        for idx, info in enumerate(infos):
            task2do = "The task to complete is " + self._last_task[idx] + ".\n"
            last_state = "The begining state is " + self._last_state[idx] + ".\n"
            text_state = "The current state is " + info["text_obs"] + ".\n"
            question = task2do + last_state + text_state + self._complete_question
            # print("completion", question)
            prompt1 = {"role": "system", "content": self._complete_system}
            prompt2 = {"role": "user", "content": question}
            prompts.append([prompt1, prompt2])
        
        # query LLMs
        responses = asyncio.run(self._small_client.async_query(prompts))
        responses = [response.choices[0].message.content for response in responses]
        
        # get task completion
        task_completion = []
        rewards = []
        for idx, response in enumerate(responses):
            completed = "yes" in response.lower()
            isSurvive = "survive" in self._last_task[idx].lower()
            time_out = (self._task_num_try[idx] > 3)
            task_completion.append(completed or isSurvive or time_out)
            rewards.append(completed)
        
        # get prompt for querying task description
        prompts = []
        new_task_idx = []
        for idx, info in enumerate(infos):
            if task_completion[idx]:
                current_state = info["text_obs"]
                user_content = self._task_user_prefix + current_state + self._task_user_question
                prompt1 = {"role": "system", "content": self._task_system}
                prompt2 = {"role": "user", "content": "You see tree.\n You have nothing in your inventory.\n Your inner properties: health: 9/9, food: 9/9, drink: 9/9, energy: 9/9.\n What do you do?"}
                prompt3 = {"role": "assistant", "content": "Reasoning: The inventory is empty now, chop down a tree to get some wood. Task: Obtain a wood log.\n\n"}
                prompt4 = {"role": "user", "content": user_content}
                # print("task", user_content)
                prompts.append([prompt1, prompt2, prompt3, prompt4])
                new_task_idx.append(idx)
        
        # query LLMs
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
        
        return rewards, []
