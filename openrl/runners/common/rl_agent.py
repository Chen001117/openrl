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
import io
import pathlib
import time
from abc import abstractmethod
from typing import Optional, Tuple, Union

import gym
import torch

from openrl.modules.common import BaseNet
from openrl.runners.common.base_agent import BaseAgent, SelfAgent
from openrl.utils.callbacks import CallbackFactory
from openrl.utils.callbacks.callbacks import BaseCallback, CallbackList, ConvertCallback
from openrl.utils.callbacks.processbar_callback import ProgressBarCallback
from openrl.utils.type_aliases import MaybeCallback

available_actions = [
    # "Find cows.", 
    # "Find water.", 
    # "Find stone.", 
    # "Find tree.",
    "Collect sapling.",
    "Place sapling.",
    "Chop tree.", 
    "Kill the cow.", 
    "Mine stone.", 
    "Drink water.",
    "Mine coal.", 
    "Mine iron.", 
    "Mine diamond.", 
    "Kill the zombie.",
    "Kill the skeleton.", 
    "Craft wood_pickaxe.", 
    "Craft wood_sword.",
    "Place crafting table.", 
    "Place furnace.", 
    "Craft stone_pickaxe.",
    "Craft stone_sword.", 
    "Craft iron_pickaxe.", 
    "Craft iron_sword.",
    "Sleep."
]

def clone_module(module, memo=None):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

    **Description**

    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().

    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.

    **Arguments**

    * **module** (Module) - Module to be cloned.

    **Return**

    * (Module) - The cloned module.

    **Example**

    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate all the way to net.
    ~~~
    """
    # NOTE: This function might break in future versions of PyTorch.

    # TODO: This function might require that module.forward()
    #       was called in order to work properly, if forward() instanciates
    #       new variables.
    # TODO: We can probably get away with a shallowcopy.
    #       However, since shallow copy does not recurse, we need to write a
    #       recursive version of shallow copy.
    # NOTE: This can probably be implemented more cleanly with
    #       clone = recursive_shallow_copy(model)
    #       clone._apply(lambda t: t.clone())

    if memo is None:
        # Maps original data_ptr to the cloned tensor.
        # Useful when a Module uses parameters from another Module; see:
        # https://github.com/learnables/learn2learn/issues/174
        memo = {}

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[param_ptr] = cloned

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
    return clone


class RLAgent(BaseAgent):
    def __init__(
        self,
        net: Optional[Union[torch.nn.Module, BaseNet]] = None,
        env: Union[gym.Env, str] = None,
        run_dir: Optional[str] = None,
        env_num: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        project_name: str = "RLAgent",
    ) -> None:
        self.net = net
        if self.net is not None:
            self.net.reset()
        self._cfg = net.cfg
        self._use_wandb = use_wandb
        self._use_tensorboard = not use_wandb and use_tensorboard
        self.project_name = project_name

        if env is not None:
            self._env = env
        elif hasattr(net, "env") and net.env is not None:
            self._env = net.env
        else:
            raise ValueError("env is None")

        if env_num is not None:
            self.env_num = env_num
        else:
            self.env_num = self._env.parallel_env_num

        # current number of timesteps
        self.num_time_steps = 0
        self._episode_num = 0
        self._total_time_steps = 0

        self._cfg.n_rollout_threads = self.env_num
        self._cfg.learner_n_rollout_threads = self._cfg.n_rollout_threads

        self.rank = rank
        self.world_size = world_size

        self.client = None
        self.agent_num = self._env.agent_num
        if run_dir is None:
            self.run_dir = self._cfg.run_dir
        else:
            self.run_dir = run_dir

        if self.run_dir is None:
            assert (not self._use_wandb) and (not self._use_tensorboard), (
                "run_dir must be set when using wandb or tensorboard. Please set"
                " run_dir in the config file or pass run_dir in the"
                " command line."
            )

        if self._cfg.experiment_name == "":
            self.exp_name = "rl"
        else:
            self.exp_name = self._cfg.experiment_name

        self._alpha = self._cfg.alpha_value

    @abstractmethod
    def train(
        self: SelfAgent,
        total_time_steps: int,
        callback: MaybeCallback = None,
    ) -> None:
        raise NotImplementedError

    def _setup_train(
        self,
        total_time_steps: int,
        callback: MaybeCallback = None,
        reset_num_time_steps: bool = True,
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_time_steps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_time_steps: Whether to reset or not the ``num_time_steps`` attribute

        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Total time_steps and callback(s)
        """
        self.start_time = time.time_ns()

        if reset_num_time_steps:
            self.num_time_steps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_time_steps += self.num_time_steps
        self._total_time_steps = total_time_steps

        # Create eval callback if needed
        callback = self._init_callback(callback, progress_bar)

        return total_time_steps, callback

    def _init_callback(
        self,
        callback: MaybeCallback,
        progress_bar: bool = False,
    ) -> BaseCallback:
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Add progress bar callback
        if progress_bar:
            callback = CallbackList([callback, ProgressBarCallback()])

        if self._cfg.callbacks:
            cfg_callback = CallbackFactory.get_callbacks(self._cfg.callbacks)
            callback = CallbackList([callback, cfg_callback])
        callback.init_callback(self)
        return callback

    @abstractmethod
    def act(self, **kwargs) -> None:
        raise NotImplementedError

    def reset(self):
        self.net.reset()

    def set_env(
        self,
        env: Union[gym.Env, str],
    ):
        self.net.reset()

        if env is not None:
            self._env = env
            self.env_num = env.parallel_env_num
            self.agent_num = env.agent_num
        env.reset(seed=self._cfg.seed)

        self.net.reset(env)

    def save(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        if isinstance(path, str):
            path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.module, path / "module.pt")

    def load(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        if isinstance(path, str):
            path = pathlib.Path(path)

        assert path.exists(), f"{path} does not exist"

        if path.is_dir():
            path = path / "module.pt"

        assert path.exists(), f"{path} does not exist"

        if not torch.cuda.is_available():
            self.net.module = torch.load(path, map_location=torch.device("cpu"))
            self.net.module.device = torch.device("cpu")
            for key in self.net.module.models:
                self.net.module.models[key].tpdv = dict(
                    dtype=torch.float32, device=torch.device("cpu")
                )
        else:
            self.net.module = torch.load(path)
            
            fine_tune = False
            from_scratch = False # do not support at this moment
            if fine_tune:
                # fine tuned
                from torch import nn
                from openrl.envs.crafter.bert import BertEncoder
                from openrl.modules.networks.utils.act import ACTLayer
                from openrl.modules.networks.utils.mlp import MLPLayer
                from openrl.modules.networks.utils.rnn import RNNLayer
                
                # base network
                net = self.net.module.models['policy'].base
                net.out_layer = nn.Linear(net.hidden_size*2, net.hidden_size)
                with torch.no_grad():
                    net.out_layer.weight.data[:,net.hidden_size:] *= 0
                    net.out_layer.weight.data[:,:net.hidden_size] = torch.eye(net.hidden_size)
                    net.out_layer.bias.data *= 0
                net.out_layer.to(self.net.module.models['policy'].device)
                opt = self.net.module.optimizers["policy"]
                opt.add_param_group({"params": net.out_layer.parameters(), "name": "out_layer"})
                # critic network
                net = self.net.module.models['critic'].base
                net.out_layer = torch.nn.Linear(net.hidden_size*2, net.hidden_size)
                with torch.no_grad():
                    net.out_layer.weight.data[:,net.hidden_size:] *= 0
                    net.out_layer.weight.data[:,:net.hidden_size] = torch.eye(net.hidden_size)
                    net.out_layer.bias.data *= 0
                net.out_layer.to(self.net.module.models['critic'].device)
                opt = self.net.module.optimizers["critic"]
                opt.add_param_group({"params": net.out_layer.parameters(), "name": "out_layer"})
                # task actor
                net = self.net.module.models['policy']
                net.bert = BertEncoder(device=net.device)
                net.task_actor = RNNLayer(
                    net.hidden_size,
                    net.hidden_size,
                    net._recurrent_N,
                    net._use_orthogonal,
                )
                net2 = self.net.module.models['critic']
                net.task_actor.load_state_dict(net2.rnn.state_dict())
                act_space = gym.spaces.Discrete(20) # task_dim
                net.act2 = ACTLayer(act_space, net.hidden_size, net._use_orthogonal, net._gain)
                net.task_actor.to(self.net.module.models['policy'].device)
                net.act2.to(self.net.module.models['policy'].device)
                opt = self.net.module.optimizers["policy"]
                opt.add_param_group({"params": net.task_actor.parameters(), "name": "task_layer"})
                opt.add_param_group({"params": net.act2.parameters(), "name": "task_layer"})
                # assign critic's CNN to actor
                net = self.net.module.models['policy']
                net.critic_base = self.net.module.models['critic'].base
                # assign actor's task_actor to critic
                net = self.net.module.models['critic']
                net.task_actor = self.net.module.models['policy'].task_actor
                net.bert = self.net.module.models['policy'].bert
                net.act2 = self.net.module.models['policy'].act2
                # new value norm
                net = self.net.module.models["critic"]
                net.ex_value_normalizer = clone_module(net.value_normalizer)
                # set alpha
                net = self.net.module.models["critic"]
                net._alpha = self._alpha
                net = self.net.module.models["policy"]
                net._alpha = self._alpha
            elif from_scratch:
                net = self.net.module.models['policy']
                net.out_layer = nn.Linear(net.hidden_size*2, net.hidden_size)
                net.out_layer.to(self.net.module.models['policy'].device)
                opt = self.net.module.optimizers["policy"]
                opt.add_param_group({"params": net.out_layer.parameters()})
                net = self.net.module.models['critic'].base
                net.out_layer = torch.nn.Linear(net.hidden_size*2, net.hidden_size)
                net.out_layer.to(self.net.module.models['critic'].device)
                opt = self.net.module.optimizers["critic"]
                opt.add_param_group({"params": net.out_layer.parameters()})
            # #########################
            # # mlp
            # net = self.net.module.models['policy'].base
            # net.mlp = nn.Sequential(
            #     nn.Linear(22, net.hidden_size),
            #     nn.ReLU(),
            #     nn.Linear(net.hidden_size, net.hidden_size),
            #     nn.ReLU(),
            #     nn.Linear(net.hidden_size, net.hidden_size),
            # )
            # net.mlp.to(self.net.module.models['policy'].device)
            # opt = self.net.module.optimizers["policy"]
            # opt.add_param_group({"params": net.mlp.parameters(), "name": "out_layer"})
            # net = self.net.module.models['critic'].base
            # net.mlp = nn.Sequential(
            #     nn.Linear(22, net.hidden_size),
            #     nn.ReLU(),
            #     nn.Linear(net.hidden_size, net.hidden_size),
            #     nn.ReLU(),
            #     nn.Linear(net.hidden_size, net.hidden_size),
            # )
            # net.mlp.to(self.net.module.models['critic'].device)
            # opt = self.net.module.optimizers["critic"]
            # opt.add_param_group({"params": net.mlp.parameters(), "name": "out_layer"})
            
        self.net.reset()

    def load_policy(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        self.net.load_policy(path)
