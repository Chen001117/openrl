from typing import Any, Dict, Optional

import torch
from torch import nn

from gymnasium.spaces import Discrete
from gymnasium.spaces.dict import Dict as DictSpace
from transformers import AutoConfig, AutoModelForCausalLM

from openrl.modules.networks.utils.nlp.causal_lm import CausalLM

class GPTCausalLM(CausalLM):
    def __init__(
        self,
        observation_space: DictSpace,
        action_space: Discrete,
        model_name: str,
        generation_kwargs: Dict[str, Any] = {},
        prompt_truncation_side: str = "left",
        state_dict: Dict[str, Any] = None,
    ):

        super(GPTCausalLM, self).__init__(
            observation_space,
            action_space,
            model_name,
            generation_kwargs,
            prompt_truncation_side,
            state_dict,
        )
        
        # disable drop-out layer
        config = AutoConfig.from_pretrained(model_name)
        
        # create critic model (with an additional linear value head)
        self.value_model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config
        )
        self.value_head = nn.Linear(
            self.value_model.config.hidden_size, 1, bias=False
        )
        
        if self._disable_drop_out:
            config_dict = config.to_dict()
            for key in config_dict:
                if "drop" in key:
                    config_dict[key] = 0.0
            config = config.from_dict(config_dict)
        
        # create actor model
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config
        )

        self.load_from_dict(state_dict)

