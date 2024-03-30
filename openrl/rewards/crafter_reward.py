from typing import Any, Dict, List, Union

import numpy as np
from gymnasium import Env

from openrl.envs.crafter.rewards.llms_coach import LLMsCoach
from openrl.rewards.base_reward import BaseReward


class CrafterReward(BaseReward):
    def __init__(
        self,
        env: Env,
        api_key: str,
        api_base: str,
        model: str,
    ):
        self.inner_rew_funcs = dict()
        self.batch_rew_funcs = dict()
        self.step_rew_funcs = {
            "lm_rewards": LLMsCoach(api_key, api_base, model, reset_freq=32),
        }

    def step_reward(
        self, data: Dict[str, Any]
    ) -> Union[np.ndarray, List[Dict[str, Any]]]:
        
        rewards = data["rewards"].copy()
         
        for rew_func in self.step_rew_funcs.values():
            new_rew, _ = rew_func(data)
            rewards = new_rew.reshape(rewards.shape)
        
        return rewards, []

    def batch_rewards(self, buffer) -> Dict[str, Any]:
        """
        calculate batch rewards and update KL_penalty's alpha coeff here.

        Args:
            buffer (): buffer.data.rewards is updated here
        """
        return dict()
