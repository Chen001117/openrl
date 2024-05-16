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
        target_kl: Optional[float] = 0.5,
        init_alpha: Optional[float] = 0.,
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
        
        path = pathlib.Path(base_model)
        assert path.exists(), f"{path} does not exist"
        if path.is_dir():
            path = path / "module.pt"
        assert path.exists(), f"{path} does not exist"
        print("KL penalty load model from ", path)
        
        if not torch.cuda.is_available():
            raise NotImplementedError("KL penalty only support cuda")
        else:
            self._ref_net = torch.load(path)
            self._ref_net.models["policy"].base.set_isbase()
        
        for key in self._ref_net.models:
            self._ref_net.models[key] = self._ref_net.models[key].eval()
        
        self._alpha = init_alpha
        self._init_alpha = init_alpha
        self._target_kl = target_kl
        self._update_rate = 0.3
        self._clip_coef = 0.2
        self._kl_length = 16
        self._kl_data = []
        self._action_dist = CategoricalDistribution(env.action_space[0].n)

    def update_alpha(self, kl_div: float) -> None:
        diff_to_target = (kl_div - self._target_kl) / self._target_kl
        e_t = np.clip(diff_to_target, -self._clip_coef, self._clip_coef)
        self._alpha = self._alpha * (1 + self._update_rate * e_t)
        self._alpha = max(self._init_alpha, self._alpha)
        # print("update alpha: ", self._alpha, "KL", kl_div)
        
    def __call__(
        self, data: Dict[str, Any]
    ) -> Union[np.ndarray, List[Dict[str, Any]]]:
        step = data["step"]
        obs = data["buffer"].data.get_batch_data("policy_obs", step)
        rnn_states = data["buffer"].data.get_batch_data("rnn_states", step)
        masks = data["buffer"].data.get_batch_data("masks", step)
        actions = data["actions"][:,:,:1]
        action_log_probs = data["action_log_probs"][:,:,:1]
        action_masks = data["buffer"].data.get_batch_data("action_masks", step)

        actions = torch.tensor(actions).squeeze(-1)
        for key in self._ref_net.models:
            self._ref_net.models[key] = self._ref_net.models[key].eval()
        
        with torch.no_grad():
            ref_log_prob, _, _ = self._ref_net.models["policy"](
                "eval_actions",
                obs,
                rnn_states,
                actions,
                masks,
                action_masks,
                ref=True
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
                    "kl_rewards": rew.mean(),
                }
            )
            
        self._kl_data.append(kl_div.mean())
        if len(self._kl_data) > self._kl_length:
            self.update_alpha(np.mean(self._kl_data))
            self._kl_data = []
            
        # print("KL", kl_div.mean(), "Alpha", self._alpha)
            
        return rew, infos

