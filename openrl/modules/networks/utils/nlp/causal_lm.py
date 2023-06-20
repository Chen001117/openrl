from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple, Optional, Union

import torch
from gym.spaces import Discrete
from gym.spaces.dict import Dict as DictSpace
from torch import nn
from torch.distributions import Categorical
from transformers import AutoTokenizer, PreTrainedModel
from transformers.modeling_utils import unwrap_model

from openrl.envs.nlp.utils.distribution import CategoricalDistribution

class CausalLM(nn.Module):
    def __init__(
        self,
        observation_space: DictSpace,
        action_space: Discrete,
        model_name: str,
        generation_kwargs: Dict[str, Any] = {},
        prompt_truncation_side: str = "left",
        state_dict: Dict[str, Any] = None,
    ):
        """

        Args:
            observation_space (DictSpace): Observation space
            action_space (Discrete): Action space
            model_name (str): name of the causal or seq2seq model from transformers library
            optimizer_kwargs (Dict[str, Any], optional): optimizer kwargs. Defaults to {}.
            weight_decay (float, optional): weight decay. Defaults to 1e-6.
            use_sde (bool, optional): Use state-dependent exploration. Defaults to None. (Unused parameter from stable-baselines3)
            apply_model_parallel (bool, optional): whether to apply model parallel. Defaults to True.
            optimizer_class (torch.optim.Optimizer, optional): Optimizer class. Defaults to torch.optim.AdamW.
            generation_kwargs (Dict[str, Any], optional): generation parameters for rollout. Defaults to {}.
            prompt_truncation_side (str, optional): truncation side for prompt text. Defaults to "left".
        """
        super().__init__()
        self._action_space = action_space
        self._action_dist = CategoricalDistribution(self._action_space.n)
        self._generation_kwargs = generation_kwargs
        self._prompt_truncation_side = prompt_truncation_side

    def _prepare_inputs_for_model(
        self,
        model: nn.Module,
        input_ids: torch.tensor,
        model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ):
        model_inputs = unwrap_model(model).prepare_inputs_for_generation(
            input_ids, **model_kwargs
        )
        return model_inputs

    def load_from_dict(self, state_dict: dict = None):
        if state_dict is not None:
            self.policy_model.load_state_dict(state_dict["policy_model"])
            self.value_model.load_state_dict(state_dict["value_model"])
            self.value_head.load_state_dict(state_dict["value_head"])
            self.optimizer.load_state_dict(state_dict["optimizer"])

    def forward_policy(
        self,
        obs,
        actions: torch.Tensor,
        past_model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]

        past_model_kwargs = {"attention_mask": attention_mask}

        model_inputs = self._prepare_inputs_for_model(
            self.actor_engine, input_ids, past_model_kwargs
        )

        # forward pass to transformers
        output = self.actor_engine(output_hidden_states=False, **model_inputs)
        
        # compute action probs - policy head
        next_token_logits = output.logits[:, -1, :]
        dist = self._action_dist.proba_distribution(action_logits=next_token_logits)
        entropy = dist.entropy()

        # sample act
        # actions_input = actions.to(next_token_logits.device)
        log_prob = dist.log_prob(actions)

        return log_prob, entropy
    
    def forward_value(
        self,
        obs,
        past_model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ) -> torch.Tensor:
        
        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]

        past_model_kwargs = {"attention_mask": attention_mask}

        model_inputs = self._prepare_inputs_for_model(
            self.critic_engine, input_ids, past_model_kwargs
        )

        # forward pass to transformers
        values = self.critic_engine(output_hidden_states=True, **model_inputs)
        
        # # pool the hidden states 
        # last_tokens_hidden = output.hidden_states[-1][:, -1, :]
        # values = self.value_head.forward(last_tokens_hidden)

        return values

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        log_probs, entropy = self.forward_policy(obs=obs, actions=actions)
        values = self.forward_value(obs)

        return log_probs, entropy, values

    def get_distribution(self, obs, past_model_kwargs=None, detach=False):

        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]

        past_model_kwargs = {"attention_mask": attention_mask}

        if detach:
            with torch.no_grad():
                model_inputs = self._prepare_inputs_for_model(
                    self.actor_engine, input_ids, past_model_kwargs
                )
                # forward pass to transformers
                output = self.actor_engine(output_hidden_states=False, **model_inputs)
        else:
            model_inputs = self._prepare_inputs_for_model(
                self.actor_engine, input_ids, past_model_kwargs
            )
            # forward pass to transformers
            output = self.actor_engine(output_hidden_states=False, **model_inputs)

        # compute action probs - policy head
        next_token_logits = output.logits[:, -1, :]
        distribution = self._action_dist.proba_distribution(action_logits=next_token_logits)

        return distribution
