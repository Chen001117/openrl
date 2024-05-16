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
            

class LLMsCoach(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_env = 128
        
        self.available_actions = [
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
        
        self._task_num_try = np.ones(self.n_env)
        
        self.task_cnt = dict()
           
    def get_rewards(self, obj_diff, current_task):
        
        if current_task == "Survive.":
            return 1.
        
        suc_task = []
        suc = "gained 1 wood" in obj_diff
        suc |= "gained 2 wood" in obj_diff
        suc |= "gained 3 wood" in obj_diff 
        suc |= "gained 4 wood" in obj_diff
        if suc:
            suc_task.append("Chop tree.")
        suc = "gained 1 food" in obj_diff
        suc |= "gained 2 food" in obj_diff
        suc |= "gained 3 food" in obj_diff
        suc |= "gained 4 food" in obj_diff
        suc &= "killed a cow" in obj_diff
        if suc:
            suc_task.append("Kill the cow.")
        gain_stone_num = obj_diff.count("gained 1 stone")
        place_stone_num = obj_diff.count("lost 1 stone")
        suc = gain_stone_num > place_stone_num
        if suc:
            suc_task.append("Mine stone.")
        suc = "gained 1 drink" in obj_diff
        suc |= "gained 2 drink" in obj_diff
        suc |= "gained 3 drink" in obj_diff
        suc |= "gained 4 drink" in obj_diff
        if suc:
            suc_task.append("Drink water.")
        suc = "gained 1 coal" in obj_diff
        suc |= "gained 2 coal" in obj_diff
        suc |= "gained 3 coal" in obj_diff
        suc |= "gained 4 coal" in obj_diff
        if suc:
            suc_task.append("Mine coal.")
        suc = "gained 1 iron" in obj_diff
        suc |= "gained 2 iron" in obj_diff
        suc |= "gained 3 iron" in obj_diff
        suc |= "gained 4 iron" in obj_diff
        if suc:
            suc_task.append("Mine iron.")
        suc = "gained 1 diamond" in obj_diff
        suc |= "gained 2 diamond" in obj_diff
        suc |= "gained 3 diamond" in obj_diff
        suc |= "gained 4 diamond" in obj_diff
        if suc:
            suc_task.append("Mine diamond.")
        suc = "killed a zombie" in obj_diff
        if suc:
            suc_task.append("Kill the zombie.")
        suc = "killed a skeleton" in obj_diff
        if suc:
            suc_task.append("Kill the skeleton.")
        suc = "gained 1 wood_pickaxe" in obj_diff
        suc |= "gained 2 wood_pickaxe" in obj_diff
        suc |= "gained 3 wood_pickaxe" in obj_diff
        suc |= "gained 4 wood_pickaxe" in obj_diff
        if suc:
            suc_task.append("Craft wood_pickaxe.")
        suc = "gained 1 wood_sword" in obj_diff
        suc |= "gained 2 wood_sword" in obj_diff
        suc |= "gained 3 wood_sword" in obj_diff
        suc |= "gained 4 wood_sword" in obj_diff
        if suc:
            suc_task.append("Craft wood_sword.")
        suc = "found crafting table" in obj_diff
        if suc:
            suc_task.append("Place crafting table.")
        suc = "gained 1 stone_pickaxe" in obj_diff
        suc |= "gained 2 stone_pickaxe" in obj_diff
        suc |= "gained 3 stone_pickaxe" in obj_diff
        suc |= "gained 4 stone_pickaxe" in obj_diff
        if suc:
            suc_task.append("Craft stone_pickaxe.")
        suc = "gained 1 stone_sword" in obj_diff
        suc |= "gained 2 stone_sword" in obj_diff
        suc |= "gained 3 stone_sword" in obj_diff
        suc |= "gained 4 stone_sword" in obj_diff
        if suc:
            suc_task.append("Craft stone_sword.")
        suc = "gained 1 iron_pickaxe" in obj_diff
        suc |= "gained 2 iron_pickaxe" in obj_diff
        suc |= "gained 3 iron_pickaxe" in obj_diff
        suc |= "gained 4 iron_pickaxe" in obj_diff
        if suc:
            suc_task.append("Craft iron_pickaxe.")
        suc = "gained 1 iron_sword" in obj_diff
        suc |= "gained 2 iron_sword" in obj_diff
        suc |= "gained 3 iron_sword" in obj_diff
        suc |= "gained 4 iron_sword" in obj_diff
        if suc:
            suc_task.append("Craft iron_sword.")
        suc = "fell asleep" in obj_diff
        if suc:
            suc_task.append("Sleep.")
        suc = "found furnace" in obj_diff
        if suc:
            suc_task.append("Place furnace.")
        suc = "gained 1 sapling" in obj_diff
        if suc:
            suc_task.append("Collect sapling.")
        suc = "lost 1 sapling" in obj_diff
        if suc:
            suc_task.append("Place sapling.")
        
        # suc = "found cow" in obj_diff
        # if suc:
        #     suc_task.append("Find cows.")
        # suc = "found water" in obj_diff
        # if suc:
        #     suc_task.append("Find water.")
        # suc = "found tree" in obj_diff
        # if suc:
        #     suc_task.append("Find tree.")
        # suc = "found stone" in obj_diff
        # if suc:
        #     suc_task.append("Find stone.")
        
        if current_task in suc_task:
            return 1.
        else:
            return 0.
            if len(suc_task) > 0:
                return -.1
            else:
                return 0.
        
        raise ValueError("Task {} not recognized.".format(current_task))
     
    def __call__(
        self, data: Dict[str, Any], past_model_kwargs: Optional[Any] = None
    ) -> Union[np.ndarray, List[Dict[str, Any]]]:
        
        infos = data["infos"]
        actions = data["actions"]
        
        # for inference
        if "task" in data:
            raise ValueError("Task should not be in data during training.")
            self._last_task = data["task"]
        
        # get rewards
        rewards = []
        for info, action in zip(infos, actions):
            if info["step"] == 0:
                # set task to survive for new episodes
                cur_task = "Survive."
                completed = False
                rewards.append(1.)
            else:
                obj_diff = info["obj_diff"]
                cur_task = self.available_actions[action[0,1]]
                reward = self.get_rewards(obj_diff, cur_task)
                completed = reward > 0
                rewards.append(reward)
                
            # statistics
            if completed:
                if cur_task not in self.task_cnt:
                    self.task_cnt[cur_task] = 1
                else:
                    self.task_cnt[cur_task] += 1    
        if np.random.rand() < 0.001:
            print("completed task name: ", self.task_cnt)
        
        # scale rewards
        rewards = np.array(rewards) * 4.
        
        # update infos
        new_infos = []
        for i in range(self.n_env):
            env_dict = {"instruction_following_rewards": rewards[i]}
            new_infos.append(env_dict)
        
        return rewards, new_infos
