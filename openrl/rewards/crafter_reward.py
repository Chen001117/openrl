from typing import Any, Dict, List, Union

import numpy as np
from gymnasium import Env

from openrl.envs.crafter.rewards.llms_coach import LLMsCoach
from openrl.envs.crafter.rewards.kl_penalty import KLPenalty
from openrl.rewards.base_reward import BaseReward


class CrafterReward(BaseReward):
    def __init__(
        self,
        env: Env,
        cfg: Any,
        base_model: str,
        target_kl: float,
        init_alpha: float,
    ):
        self.inner_rew_funcs = dict()
        self.batch_rew_funcs = dict()
        self.step_rew_funcs = {
            "lm_rewards": LLMsCoach(),
            "kl_pen": KLPenalty(env, cfg, base_model, target_kl, init_alpha)
        }

    def step_reward(
        self, data: Dict[str, Any]
    ) -> Union[np.ndarray, List[Dict[str, Any]]]:
        
        rewards = data["rewards"].copy() * 0.
        infos = [{"original_rewards": rew} for rew in data["rewards"]]
        
        for rew_func in self.step_rew_funcs.values():
            new_rew, new_info = rew_func(data)
            rewards += new_rew.reshape(rewards.shape) 
            if len(infos) == 0:
                infos = new_info
            else:
                for i in range(len(infos)):
                    infos[i].update(new_info[i])
        
        # [instruction following rewards, extrinsic rewards]
        rewards = np.concatenate([rewards, data["rewards"]], -1)
        
        return rewards, infos

    def batch_rewards(self, buffer) -> Dict[str, Any]:
        """
        calculate batch rewards and update KL_penalty's alpha coeff here.

        Args:
            buffer (): buffer.data.rewards is updated here
        """
        return dict()
