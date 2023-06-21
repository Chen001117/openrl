from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch
from gym.spaces import Discrete
from gym.spaces.dict import Dict as DictSpace
from torch import nn
from torch.distributions import Categorical
from transformers import AutoTokenizer, PreTrainedModel
from transformers.modeling_utils import unwrap_model

from openrl.envs.nlp.utils.distribution import CategoricalDistribution


class PolicyType(Enum):
    CAUSAL = 0
    SEQ2SEQ = 1


def get_device(device: Union[torch.device, str] = "auto") -> torch.device:
    """
    Retrieve PyTorch device.
    It checks torchat the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.
    :param device: One for 'auto', 'cuda', 'cpu'
    :return:
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to torch.device
    device = torch.device(device)

    # Cuda not available
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device


@dataclass
class EvaluateActionsOutput:
    """
    Dataclass for the output of the method policy.evaluate_actions().
    This is invoked during training phase for each mini-batch in the rollout buffer
    """

    # values of the given state
    values: torch.tensor
    # log prob of chosen actions
    log_prob: torch.tensor
    # entropy of action dist
    entropy: torch.tensor


@dataclass
class PolicyOutput:
    """
    Dataclass for the output of the method policy.foward_policy()
    """

    # chosen actions by policy
    actions: torch.tensor
    # raw log probs corresponding to chosen actions
    raw_log_probs: torch.tensor
    # processed log probs (eg: after action masking) for chosen actions
    log_probs: torch.tensor
    # entropy of action dist
    entropy: torch.tensor
    # cached policy activations for sequential forward passes
    past_model_kwargs: torch.tensor


@dataclass
class RefPolicyOutput:
    """
    Dataclass for the output of the method policy.get_ref_log_probs()
    """

    # ref log_probs for corresponding observation and chosen action
    log_probs: torch.tensor
    # cached policy activations for sequential forward passes
    past_model_kwargs: torch.tensor


@dataclass
class ValueOutput:
    """
    Dataclass for the output of the method policy.forward_value()
    """

    # values corresponding to given state
    values: torch.tensor
    # cached value activations for sequential forward passes
    past_model_kwargs: Dict[str, torch.tensor]


@dataclass
class GenerationInputs:
    # prompt inputs
    inputs: torch.tensor
    # prompt attention masks
    attention_masks: torch.tensor


@dataclass
class GenerationOutputs:
    # log probs at each time step
    step_wise_logprobs: List[List[torch.tensor]]
    # actions at each time step
    step_wise_actions: List[torch.tensor]
    # generated tokens
    gen_tokens: List[List[int]]
    # generated texts
    gen_texts: List[str]
    # action masks
    action_masks: List[torch.tensor] = None


class LMActorCriticPolicy(nn.Module):
    def __init__(
        self,
        observation_space: DictSpace,
        action_space: Discrete,
        model_name: str,
        optimizer_kwargs: Dict[str, Any] = {},
        weight_decay: float = 1e-6,
        use_sde: bool = None,
        apply_model_parallel: bool = True,
        optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
        generation_kwargs: Dict[str, Any] = {},
        prompt_truncation_side: str = "left",
        config: Optional[str] = None,
        device: str = "cpu",
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
        self._apply_model_parallel = apply_model_parallel
        self._build_model_heads(model_name, config, device)
        self._setup_optimizer(optimizer_kwargs, weight_decay, optimizer_class)
        self._action_dist = CategoricalDistribution(self._action_space.n)
        self._generation_kwargs = generation_kwargs
        self._prompt_truncation_side = prompt_truncation_side

    @property
    def device(self):
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.
        :return:"""
        for param in self.parameters():
            return param.device
        return get_device("cpu")

    def _setup_optimizer(
        self,
        optimizer_kwargs: Dict[str, Any],
        weight_decay: float,
        optimizer_class: torch.optim,
    ):
        params = list(self.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = optimizer_class(
            optimizer_grouped_parameters, **optimizer_kwargs
        )

    def forward(self, *args, **kwargs):
        # dummy just to comply with base policy
        pass

    # Following methods need to be implemented by sub-classing
    @abstractmethod
    def _build_model_heads(self, model_name: str):
        """
        Builds policy and value models
        and sets self._policy_model and self._value_model
        """
        raise NotImplementedError

    @abstractmethod
    def forward_policy(
        self,
        obs,
        actions: torch.tensor,
        past_model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ) -> PolicyOutput:
        raise NotImplementedError

    @abstractmethod
    def forward_value(
        self,
        obs,
        past_model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ) -> ValueOutput:
        raise NotImplementedError

    @abstractmethod
    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> EvaluateActionsOutput:
        """
        Evaluates specified <observation, action>
        and returns log_probs, values, entropy

        This is invoked for each mini-batch in rollout buffer during training iteration
        """
        raise NotImplementedError

