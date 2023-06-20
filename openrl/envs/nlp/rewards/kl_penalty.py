from typing import Any, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import unwrap_model

from openrl.envs.nlp.utils.distribution import CategoricalDistribution

def get_train_ds_config(offload, stage=0):

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        },
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": -1,
        "train_micro_batch_size_per_gpu": -1,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }

def get_optimizer_grouped_parameters(model,
                                     weight_decay,
                                     no_decay_name_list=[
                                         "bias", "LayerNorm.weight"
                                     ]):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]
    return optimizer_grouped_parameters

class KLPenalty(nn.Module):
    def __init__(
        self,
        action_space: gym.Space,
        ref_model: str,
        apply_model_parallel: bool = False,
    ):
        super().__init__()

        # reference model
        self._apply_model_parallel = apply_model_parallel
        self._ref_net = AutoModelForCausalLM.from_pretrained(ref_model)
        self._ref_net = self._ref_net.to("cuda:0")
        self._ref_net = self._ref_net.eval()

        import deepspeed

        ds_config = get_train_ds_config(offload=True)
        ds_config['train_micro_batch_size_per_gpu'] = 16
        ds_config['train_batch_size'] = 32

        # DeepSpeed Engine
        self.ref_engine, *_ = deepspeed.initialize(
            model=self._ref_net,
            config=ds_config
        )

        # alpha adjustment
        self._alpha = 0.2
        self._target_kl = 0.05
        self._update_rate = 0.1
        self._clip_coef = 0.2
        self._action_dist = CategoricalDistribution(action_space.n)

    def update_alpha(self, kl_div: float) -> None:
        diff_to_target = (kl_div - self._target_kl) / self._target_kl
        e_t = np.clip(diff_to_target, -self._clip_coef, self._clip_coef)
        self._alpha = self._alpha * (1 + self._update_rate * e_t)

    def __call__(
        self, data: Dict[str, Any], past_model_kwargs: Optional[Any] = None
    ) -> Union[np.ndarray, List[Dict[str, Any]]]:
        step = data["step"]
        obs = data["buffer"].data.get_batch_data("policy_obs", step)
        actions = data["actions"]
        action_log_probs = data["action_log_probs"]
        input_ids = torch.tensor(obs["input_encoded_pt"]).int()
        attention_mask = torch.tensor(obs["input_attention_mask_pt"])

        actions = torch.tensor(actions).flatten()
        input_ids = torch.squeeze(input_ids, dim=1)
        attention_mask = torch.squeeze(attention_mask, dim=1)

        past_model_kwargs = { "attention_mask": attention_mask}

        self._ref_net = self._ref_net.eval()

        model_inputs = self._prepare_inputs_for_model(
            self._ref_net, input_ids, past_model_kwargs
        )

        with torch.no_grad():
            output = self.ref_engine(output_hidden_states=True, **model_inputs)
            next_token_logits = output.logits[:, -1, :]
            dist = self._action_dist.proba_distribution(action_logits=next_token_logits)
            action_input = actions.to(next_token_logits.device)
            ref_log_prob = dist.log_prob(action_input)

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

    def _prepare_inputs_for_model(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.tensor,
        model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ):
        
        model_inputs = unwrap_model(model).prepare_inputs_for_generation(
            input_ids, **model_kwargs
        )

        for key in model_inputs:
            if model_inputs[key] is not None:
                model_inputs[key] = model_inputs[key].to(self.ref_engine.device)

        # if self._apply_model_parallel and unwrap_model(model).is_parallelizable:
        #     # if model is in parallel mode, move the tensors to the first device
        #     model_inputs = {
        #         key: (
        #             value.to(model.transformer.first_device)
        #             if isinstance(value, torch.Tensor)
        #             and hasattr(model.transformer, "first_device")
        #             else value
        #         )
        #         for key, value in model_inputs.items()
        #     }
        return model_inputs
