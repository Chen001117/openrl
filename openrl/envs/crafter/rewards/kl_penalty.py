from typing import Any, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import Env
import torch
from torch import nn

import copy
from gymnasium import Env, spaces
from gymnasium.spaces.dict import Dict as DictSpace

from openrl.modules.ppo_module import PPOModule
from openrl.envs.nlp.utils.distribution import CategoricalDistribution


class KLPenalty(nn.Module):
    def __init__(
        self,
        env: Env,
        cfg: Any,
        base_model: Optional[str] = None,
    ):
        super().__init__()

        self.device = "cuda"
        
        obs_space = copy.deepcopy(env.observation_space)
        for key in ["policy", "critic"]:
            obs_space[key] = DictSpace(
                {
                    "image": copy.deepcopy(obs_space[key]),
                    "task_emb": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(384,)
                    ),
                }
            )
        
        self._ref_net = PPOModule(
            cfg=cfg,
            policy_input_space=obs_space,
            critic_input_space=obs_space,
            act_space=env.action_space,
            share_model=cfg.use_share_model,
            device=self.device,
            rank=0,
            world_size=1,
        )
        
        for key in self._ref_net.models:
            self._ref_net.models[key] = self._ref_net.models[key].eval()
        
        self._alpha = 0.2
        
        self._action_dist = CategoricalDistribution(env.action_space.n)

    def __call__(
        self, data: Dict[str, Any]
    ) -> Union[np.ndarray, List[Dict[str, Any]]]:
        
        step = data["step"]
        obs = data["buffer"].data.get_batch_data("policy_obs", step)
        rnn_states_actor = data["buffer"].data.get_batch_data("rnn_states", step)
        masks = data["dones"]
        actions = data["actions"]
        action_log_probs = data["action_log_probs"]

        actions = torch.tensor(actions).flatten()
        for key in self._ref_net.models:
            self._ref_net.models[key] = self._ref_net.models[key].eval()
        
        with torch.no_grad():
            ref_log_prob, _, _ = self._ref_net.models["policy"](
                "eval_actions",
                obs,
                rnn_states_actor,
                actions,
                masks,
            )

        ref_log_prob = ref_log_prob.reshape(action_log_probs.shape)

        kl_div = action_log_probs.copy() - ref_log_prob.detach().cpu().numpy()
        rew = -self._alpha * kl_div
        infos = []
        for kl in kl_div:
            infos.append(
                {
                    "alpha": self._alpha,
                    "kl_div": kl.mean(),
                }
            )
        return rew, infos

