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

import io
import pathlib

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
        )
        
        path = "crafter_agent-100M-3/"
        print("KL penalty load model from ", path)
        if isinstance(path, str):
            path = pathlib.Path(path)
        assert path.exists(), f"{path} does not exist"
        if path.is_dir():
            path = path / "module.pt"
        assert path.exists(), f"{path} does not exist"
        
        module = torch.load(path)
        self._ref_net.models["policy"].base = module.models["policy"].base
        self._ref_net.models["policy"].rnn = module.models["policy"].rnn
        self._ref_net.models["policy"].act = module.models["policy"].act
        # self._ref_net.models["policy"].out_layer = module.models["policy"].base.out_layer
        
        for key in self._ref_net.models:
            self._ref_net.models[key] = self._ref_net.models[key].eval()
        
        self._alpha = 0.1
        self._target_kl = 0.1
        self._update_rate = 0.1
        self._clip_coef = 0.2
        self._kl_length = 64
        self._kl_data = []
        
        self._action_dist = CategoricalDistribution(env.action_space.n)

    def update_alpha(self, kl_div: float) -> None:
        diff_to_target = (kl_div - self._target_kl) / self._target_kl
        e_t = np.clip(diff_to_target, -self._clip_coef, self._clip_coef)
        self._alpha = self._alpha * (1 + self._update_rate * e_t)
        # print("update alpha: ", self._alpha, "KL", kl_div)
        
    def __call__(
        self, data: Dict[str, Any]
    ) -> Union[np.ndarray, List[Dict[str, Any]]]:
        step = data["step"]
        obs = data["buffer"].data.get_batch_data("policy_obs", step)
        rnn_states = data["buffer"].data.get_batch_data("rnn_states", step)
        masks = data["buffer"].data.get_batch_data("masks", step)
        actions = data["actions"]
        action_log_probs = data["action_log_probs"]

        actions = torch.tensor(actions).flatten()
        for key in self._ref_net.models:
            self._ref_net.models[key] = self._ref_net.models[key].eval()
        
        with torch.no_grad():
            ref_log_prob, _, _ = self._ref_net.models["policy"](
                "eval_actions",
                obs,
                rnn_states,
                actions,
                masks,
            )

        ref_log_prob = ref_log_prob.reshape(action_log_probs.shape)

        kl_div = action_log_probs.copy() - ref_log_prob.detach().cpu().numpy()
        # print("action log probs", action_log_probs.flatten()[:5])
        # print("ref log probs", ref_log_prob.flatten()[:5])
        rew = -self._alpha * kl_div 
        infos = []
        for kl in kl_div:
            infos.append(
                {
                    "alpha": self._alpha,
                    "kl_div": kl.mean(),
                }
            )
            
        self._kl_data.append(kl_div.mean())
        if len(self._kl_data) > self._kl_length:
            self.update_alpha(np.mean(self._kl_data))
            self._kl_data = []
            
        # print("KL", kl_div.mean(), "Alpha", self._alpha)
            
        return rew, infos

