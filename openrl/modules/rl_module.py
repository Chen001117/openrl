#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 The OpenRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""""""
from abc import abstractmethod
from pathlib import Path
from typing import Dict, Union

import torch
from torch import nn

from gym import spaces

from openrl.modules.base_module import BaseModule
from openrl.modules.model_config import ModelTrainConfig

from transformers.modeling_utils import unwrap_model

class Critic(nn.Module):
    def __init__(self, value_model, value_head):
        super(Critic, self).__init__()
        self.value_model = value_model
        self.value_head = value_head
    
    def forward(self, output_hidden_states=True, **kwargs):
        x = self.value_model(output_hidden_states=output_hidden_states, **kwargs)
        x = x.hidden_states[-1][:, -1, :]
        x = self.value_head.forward(x)
        return x
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return unwrap_model(self.value_model).prepare_inputs_for_generation(input_ids, **kwargs)

class RLModule(BaseModule):
    def __init__(
        self,
        cfg,
        model_configs: Dict[str, ModelTrainConfig],
        act_space: spaces.Box,
        rank: int = 0,
        world_size: int = 1,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super(RLModule, self).__init__(cfg)

        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.lr = cfg.lr
        self.critic_lr = cfg.critic_lr
        self.opti_eps = cfg.opti_eps
        self.weight_decay = cfg.weight_decay
        self.load_optimizer = cfg.load_optimizer

        self.act_space = act_space

        self.program_type = cfg.program_type
        self.rank = rank
        self.world_size = world_size

        use_half_actor = self.program_type == "actor" and cfg.use_half_actor

        for model_key in model_configs:
            model_cg = model_configs[model_key]
            model = model_cg["model"](
                cfg=cfg,
                input_space=model_cg["input_space"],
                action_space=act_space,
                device=device,
                use_half=use_half_actor,
            )
            self.models.update({model_key: model})

            if self.program_type == "actor":
                continue

            if cfg.use_deepspeed:
                import deepspeed
                from deepspeed.ops.adam import FusedAdam
                from deepspeed.ops.adam import DeepSpeedCPUAdam
                from transformers import get_scheduler
                from openrl.modules.utils.util import get_ds_config, get_optimizer_grouped_parameters
                
                # DS Config
                actor_ds_config = get_ds_config(offload=True)
                actor_ds_config['train_micro_batch_size_per_gpu'] = 32
                actor_ds_config['train_batch_size'] = 64

                # Optimizer
                actor_optim_params = get_optimizer_grouped_parameters(model.policy_model, 1e-6)
                actor_optim = DeepSpeedCPUAdam(
                    actor_optim_params,
                    lr=cfg.lr,
                    betas=(0.9, 0.95)
                )

                # LR Scheduler
                num_training_steps = cfg.num_env_steps / \
                    cfg.episode_length * cfg.num_mini_batch * cfg.ppo_epoch
                actor_lr_scheduler = get_scheduler(
                    name="constant",
                    optimizer=actor_optim,
                    num_warmup_steps=0,
                    num_training_steps=num_training_steps,
                )

                # DeepSpeed Engine
                self.actor_engine, *_ = deepspeed.initialize(
                    model=model.policy_model,
                    optimizer=actor_optim,
                    lr_scheduler=actor_lr_scheduler,
                    config=actor_ds_config
                )

                critic_ds_config = get_ds_config(offload=True)
                critic_ds_config['train_micro_batch_size_per_gpu'] = 32
                critic_ds_config['train_batch_size'] = 64

                # Optimizer
                critic = Critic(model.value_model, model.value_head)
                critic_optim_params = get_optimizer_grouped_parameters(critic, 1e-6)
                critic_optim = DeepSpeedCPUAdam(
                    critic_optim_params,
                    lr=cfg.critic_lr,
                    betas=(0.9, 0.95)
                )

                # LR Scheduler
                critic_lr_scheduler = get_scheduler(
                    name="constant",
                    optimizer=critic_optim,
                    num_warmup_steps=0,
                    num_training_steps=num_training_steps,
                )

                # DeepSpeed Engine
                self.critic_engine, *_ = deepspeed.initialize(
                    model=critic,
                    optimizer=critic_optim,
                    lr_scheduler=critic_lr_scheduler,
                    config=critic_ds_config
                )

                model.set_engine(self.actor_engine, self.critic_engine, critic)
                        
            else:
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=model_cg["lr"],
                    eps=cfg.opti_eps,
                    weight_decay=cfg.weight_decay,
                )
                self.optimizers.update({model_key: optimizer})

                if cfg.use_amp:
                    self.scaler = torch.cuda.amp.GradScaler()
                else:
                    self.scaler = None

    @abstractmethod
    def get_actions(self):
        raise NotImplementedError

    @abstractmethod
    def get_values(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate_actions(self):
        raise NotImplementedError

    @abstractmethod
    def act(self):
        raise NotImplementedError

    @abstractmethod
    def get_critic_value_normalizer(self):
        raise NotImplementedError

    def load_policy(self, model_path: str) -> None:
        model_path = Path(model_path)
        assert (
            model_path.exists()
        ), "can not find policy weight file to load: {}".format(model_path)
        state_dict = torch.load(str(model_path), map_location=self.device)
        if "policy" in self.models:
            self.models["policy"].load_state_dict(state_dict)
        else:
            self.models["model"].load_state_dict(state_dict)
        del state_dict

    def restore(self, model_dir: str) -> None:
        model_dir = Path(model_dir)
        assert model_dir.exists(), "can not find model directory to restore: {}".format(
            model_dir
        )

        for model_name in self.models:
            state_dict = torch.load(
                str(model_dir) + "/{}.pt".format(model_name), map_location=self.device
            )
            self.models[model_name].load_state_dict(state_dict)
            del state_dict

        if self.load_optimizer:
            if Path(str(model_dir) + "/actor_optimizer.pt").exists():
                for optimizer_name in self.optimizers:
                    state_dict = torch.load(
                        str(model_dir) + "/{}_optimizer.pt".format(optimizer_name),
                        map_location=self.device,
                    )
                    self.optimizers[optimizer_name].load_state_dict(state_dict)
                    del state_dict
            else:
                print("can't find optimizer to restore")
        # TODO
        # optimizer.load_state_dict(resume_state['optimizer'])

    def save(self, save_dir: str) -> None:
        print("\n\n\nenter here")
        pass
