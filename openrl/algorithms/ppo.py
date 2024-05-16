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

from typing import Union

import numpy as np
import torch
import json
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from openrl.algorithms.base_algorithm import BaseAlgorithm
from openrl.modules.networks.utils.distributed_utils import reduce_tensor
from openrl.modules.utils.util import get_grad_norm, huber_loss, mse_loss
from openrl.utils.util import check


class PPOAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        cfg,
        init_module,
        agent_num: int = 1,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self._use_share_model = cfg.use_share_model
        self.use_joint_action_loss = cfg.use_joint_action_loss
        super(PPOAlgorithm, self).__init__(cfg, init_module, agent_num, device)
        self.train_list = [self.train_ppo]
        self.use_deepspeed = cfg.use_deepspeed

        self.bc_term = False
        if self.bc_term:
            # load data
            self.buffer_size = 100
            self.expert_obs = [{
                "image": [],
                "task_emb": [],
            } for _ in range(self.buffer_size)]
            self.expert_act = [[] for _ in range(self.buffer_size)]
            for i in range(self.buffer_size):
                name = "run_results/buffer/result_{:02d}.json".format(i)
                data = json.load(open(name))
                self.expert_obs[i]["image"] = np.array(data["image"])[:,0,0]
                self.expert_obs[i]["task_emb"] = np.array(data["task_emb"])[:,0,0]
                self.expert_act[i] = np.array(data["actions"])
                self.expert_act[i] = check(self.expert_act[i]).to(**self.tpdv)
                if len(self.expert_act[i]) % 2 == 1:
                    self.expert_act[i] = self.expert_act[i][:-1]
                    self.expert_obs[i]["image"] = self.expert_obs[i]["image"][:-1]
                    self.expert_obs[i]["task_emb"] = self.expert_obs[i]["task_emb"][:-1]

    def ppo_update(self, sample, turn_on=True):
        for optimizer in self.algo_module.optimizers.values():
            if not self.use_deepspeed:
                optimizer.zero_grad()

        (
            critic_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            action_masks_batch,
        ) = sample
        
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        if self.use_amp:
            with torch.cuda.amp.autocast():
                (
                    loss_list,
                    value_loss,
                    policy_loss,
                    dist_entropy,
                    ratio,
                    task_entropy
                ) = self.prepare_loss(
                    critic_obs_batch,
                    obs_batch,
                    rnn_states_batch,
                    rnn_states_critic_batch,
                    actions_batch,
                    masks_batch,
                    action_masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    value_preds_batch,
                    return_batch,
                    active_masks_batch,
                    turn_on,
                )
            for loss in loss_list:
                self.algo_module.scaler.scale(loss).backward()
        else:
            loss_list, in_value_loss, ex_value_loss, policy_loss, dist_entropy, ratio, task_entropy = self.prepare_loss(
                critic_obs_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                masks_batch,
                action_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                value_preds_batch,
                return_batch,
                active_masks_batch,
                turn_on,
            )
            if self.use_deepspeed:
                if self._use_share_model:
                    for loss in loss_list:
                        self.algo_module.models["model"].backward(loss)
                else:
                    actor_loss = loss_list[0]
                    critic_loss = loss_list[1]
                    self.algo_module.models["policy"].backward(actor_loss)
                    self.algo_module.models["critic"].backward(critic_loss)
            else:
                for loss in loss_list:
                    loss.backward()

        # else:
        if self._use_share_model:
            actor_para = self.algo_module.models["model"].get_actor_para()
        else:
            actor_para = self.algo_module.models["policy"].parameters()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(actor_para, self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(actor_para)

        if self._use_share_model:
            critic_para = self.algo_module.models["model"].get_critic_para()
        else:
            critic_para = self.algo_module.models["critic"].parameters()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(critic_para, self.max_grad_norm)
        else:
            critic_grad_norm = get_grad_norm(critic_para)

        if self.use_amp:
            for optimizer in self.algo_module.optimizers.values():
                self.algo_module.scaler.unscale_(optimizer)

            for optimizer in self.algo_module.optimizers.values():
                self.algo_module.scaler.step(optimizer)

            self.algo_module.scaler.update()
        else:
            if self.use_deepspeed:
                if self._use_share_model:
                    self.algo_module.optimizers["model"].step()
                else:
                    self.algo_module.optimizers["policy"].step()
                    self.algo_module.optimizers["critic"].step()
            else:
                for optimizer in self.algo_module.optimizers.values():
                    optimizer.step()

        if self.world_size > 1:
            torch.cuda.synchronize()

        return (
            in_value_loss,
            ex_value_loss,
            critic_grad_norm,
            policy_loss,
            dist_entropy,
            actor_grad_norm,
            ratio,
            task_entropy
        )

    def cal_value_loss(
        self,
        in_value_normalizer,
        ex_value_normalizer,
        values,
        value_preds_batch,
        return_batch,
        active_masks_batch,
    ):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )

        if (self._use_popart or self._use_valuenorm) and in_value_normalizer is not None:
            in_value_normalizer.update(return_batch[:,:1])
            ex_value_normalizer.update(return_batch[:,1:]) #
            norm_return_batch = torch.cat([
                in_value_normalizer.normalize(return_batch[:,:1]), 
                ex_value_normalizer.normalize(return_batch[:,1:])
            ], dim=-1)
            error_clipped = (
                norm_return_batch - value_pred_clipped
            )
            error_original = norm_return_batch - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            in_val_loss = (value_loss[:,:1] * active_masks_batch).sum() / active_masks_batch.sum()
            ex_val_loss = (value_loss[:,1:] * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()
        # print(value_loss)
        # import pdb;pdb.set_trace()
        return in_val_loss, ex_val_loss

    def to_single_np(self, input):
        reshape_input = input.reshape(-1, self.agent_num, *input.shape[1:])
        return reshape_input[:, 0, ...]

    def construct_loss_list(self, policy_loss, dist_entropy, value_loss, turn_on):
        loss_list = []
        if turn_on:
            final_p_loss = policy_loss - dist_entropy * self.entropy_coef

            loss_list.append(final_p_loss)

        final_v_loss = value_loss * self.value_loss_coef
        loss_list.append(final_v_loss)

        return loss_list

    def prepare_loss(
        self,
        critic_obs_batch,
        obs_batch,
        rnn_states_batch,
        rnn_states_critic_batch,
        actions_batch,
        masks_batch,
        action_masks_batch,
        old_action_log_probs_batch,
        adv_targ,
        value_preds_batch,
        return_batch,
        active_masks_batch,
        turn_on,
    ):
        if self.use_joint_action_loss:
            critic_obs_batch = self.to_single_np(critic_obs_batch)
            rnn_states_critic_batch = self.to_single_np(rnn_states_critic_batch)
            critic_masks_batch = self.to_single_np(masks_batch)
            value_preds_batch = self.to_single_np(value_preds_batch)
            return_batch = self.to_single_np(return_batch)
            adv_targ = adv_targ.reshape(-1, self.agent_num, 1)
            adv_targ = adv_targ[:, 0, :]

        else:
            critic_masks_batch = masks_batch

        (
            values,
            action_log_probs,
            dist_entropy,
            policy_values,
            task_entropy,
        ) = self.algo_module.evaluate_actions(
            critic_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            action_masks_batch,
            active_masks_batch,
            critic_masks_batch=critic_masks_batch,
            return_task_entropy=True
        )
        
        if self.bc_term:
            expert_idx = np.random.randint(self.buffer_size)
            expert_rnn_states_actor = np.zeros([1,2,64])
            expert_rnn_states_actor = check(expert_rnn_states_actor).to(**self.tpdv)
            expert_masks = np.ones([1, 1])
            expert_action_masks = np.ones([1, 39])
            expert_rnn = []
            for i in range(len(self.expert_act[expert_idx])):
                if i%2 == 0:
                    expert_rnn.append(expert_rnn_states_actor)
                _, _, expert_rnn_states_actor = self.algo_module.models["policy"](
                    "original",
                    {
                        "image" : self.expert_obs[expert_idx]["image"][i:i+1],
                        "task_emb" : self.expert_obs[expert_idx]["task_emb"][i:i+1],
                    }, # 1, 3, 64, 64; 1, 1, 384
                    expert_rnn_states_actor, # 1, 2, 64
                    expert_masks, # 1, 1
                    expert_action_masks, # 1, 39
                    deterministic=False, # False
                )

            expert_rnn_states_actor = torch.stack(expert_rnn, 0) 
            expert_masks = np.ones([len(self.expert_act[expert_idx]), 1])
            expert_actions = torch.stack([self.expert_act[expert_idx]*0, self.expert_act[expert_idx]], -1)
            bc_action_log_probs, _, _ = self.algo_module.models["policy"](
                "eval_actions",
                self.expert_obs[expert_idx],
                expert_rnn_states_actor[:,0],
                expert_actions,
                expert_masks,
            )

        if self.use_joint_action_loss:
            action_log_probs_copy = (
                action_log_probs.reshape(-1, self.agent_num, action_log_probs.shape[-1])
                .sum(dim=(1, -1), keepdim=True)
                .reshape(-1, 1)
            )
            old_action_log_probs_batch_copy = (
                old_action_log_probs_batch.reshape(
                    -1, self.agent_num, old_action_log_probs_batch.shape[-1]
                )
                .sum(dim=(1, -1), keepdim=True)
                .reshape(-1, 1)
            )

            active_masks_batch = active_masks_batch.reshape(-1, self.agent_num, 1)
            active_masks_batch = active_masks_batch[:, 0, :]

            ratio = torch.exp(action_log_probs_copy - old_action_log_probs_batch_copy)
        else:
            ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
            ratio = ratio[:,:1] # update actor only!
            adv_targ = adv_targ[:,:1]

        if self.dual_clip_ppo:
            ratio = torch.min(ratio, self.dual_clip_coeff)

        surr1 = ratio * adv_targ
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        )

        surr_final = torch.min(surr1, surr2)

        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(surr_final, dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(surr_final, dim=-1, keepdim=True).mean()

        if self._use_policy_vhead:
            if isinstance(self.algo_module.models["actor"], DistributedDataParallel):
                policy_value_normalizer = self.algo_module.models[
                    "actor"
                ].module.value_normalizer
            else:
                policy_value_normalizer = self.algo_module.models[
                    "actor"
                ].value_normalizer
            policy_value_loss = self.cal_value_loss(
                policy_value_normalizer,
                policy_values,
                value_preds_batch,
                return_batch,
                active_masks_batch,
            )
            policy_loss = (
                policy_action_loss + policy_value_loss * self.policy_value_loss_coef
            )
        else:
            policy_loss = policy_action_loss

        if self.bc_term:
            policy_loss = policy_loss - bc_action_log_probs[:,1:].mean() * .001

        # critic update
        if self._use_share_model:
            value_normalizer = self.algo_module.models["model"].value_normalizer
        elif isinstance(self.algo_module.models["critic"], DistributedDataParallel):
            value_normalizer = self.algo_module.models["critic"].module.value_normalizer
        else:
            in_value_normalizer = self.algo_module.get_critic_value_normalizer()
            ex_value_normalizer = self.algo_module.get_ex_critic_value_normalizer()
        in_value_loss, ex_value_loss = self.cal_value_loss(
            in_value_normalizer,
            ex_value_normalizer,
            values,
            value_preds_batch,
            return_batch,
            active_masks_batch,
        )

        loss_list = self.construct_loss_list(
            policy_loss, dist_entropy, in_value_loss+ex_value_loss*0., turn_on #
        )
        return loss_list, in_value_loss, ex_value_loss, policy_loss, dist_entropy, ratio, task_entropy

    def get_data_generator(self, buffer, advantages):
        if self._use_recurrent_policy:
            if self.use_joint_action_loss:
                data_generator = buffer.recurrent_generator_v3(
                    advantages, self.num_mini_batch, self.data_chunk_length
                )
            else:
                data_generator = buffer.recurrent_generator(
                    advantages, self.num_mini_batch, self.data_chunk_length
                )
        elif self._use_naive_recurrent:
            data_generator = buffer.naive_recurrent_generator(
                advantages, self.num_mini_batch
            )
        else:
            data_generator = buffer.feed_forward_generator(
                advantages, self.num_mini_batch
            )
        return data_generator

    def train_ppo(self, buffer, turn_on):
        if self._use_popart or self._use_valuenorm:
            if self._use_share_model:
                value_normalizer = self.algo_module.models["model"].value_normalizer
            elif isinstance(self.algo_module.models["critic"], DistributedDataParallel):
                value_normalizer = self.algo_module.models[
                    "critic"
                ].module.value_normalizer
            else:
                in_value_normalizer = self.algo_module.get_critic_value_normalizer()
                ex_value_normalizer = self.algo_module.get_ex_critic_value_normalizer()
            if in_value_normalizer is not None:
                in_val = in_value_normalizer.denormalize(buffer.value_preds[:-1,:,:,:1])
                in_adv = buffer.returns[:-1,:,:,:1] - in_val
                ex_val = ex_value_normalizer.denormalize(buffer.value_preds[:-1,:,:,1:])
                ex_adv = buffer.returns[:-1,:,:,1:] - ex_val
                advantages = np.concatenate([in_adv, ex_adv], axis=-1)
            else:
                advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        # TODO: seems like normalize two times
        # if self._use_adv_normalize:
        #     in_adv = advantages[:,:,:,:1] 
        #     advantages[:,:,:,:1] = (in_adv - in_adv.mean()) / (in_adv.std() + 1e-5)
        #     out_adv = advantages[:,:,:,1:]
        #     advantages[:,:,:,1:] = (out_adv - out_adv.mean()) / (out_adv.std() + 1e-5)

        in_advantages_copy = advantages[:,:,:,:1].copy()
        in_advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(in_advantages_copy)
        std_advantages = np.nanstd(in_advantages_copy)
        advantages[:,:,:,:1] = (in_advantages_copy - mean_advantages) / (std_advantages + 1e-5)
        
        ex_advantages_copy = advantages[:,:,:,1:].copy()
        ex_advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(ex_advantages_copy)
        std_advantages = np.nanstd(ex_advantages_copy)
        advantages[:,:,:,1:] = (ex_advantages_copy - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info["value_loss"] = 0
        train_info["ex_value_loss"] = 0
        train_info["policy_loss"] = 0

        train_info["dist_entropy"] = 0
        train_info["task_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["critic_grad_norm"] = 0
        train_info["ratio"] = 0
        if self.world_size > 1:
            train_info["reduced_value_loss"] = 0
            train_info["reduced_policy_loss"] = 0

        for _ in range(self.ppo_epoch):
            data_generator = self.get_data_generator(buffer, advantages)

            for sample in data_generator:
                (
                    in_value_loss,
                    ex_value_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    ratio,
                    task_entropy,
                ) = self.ppo_update(sample, turn_on)

                if self.world_size > 1:
                    train_info["reduced_value_loss"] += reduce_tensor(
                        value_loss.data, self.world_size
                    )
                    train_info["reduced_policy_loss"] += reduce_tensor(
                        policy_loss.data, self.world_size
                    )

                train_info["value_loss"] += in_value_loss.item()
                train_info["ex_value_loss"] += ex_value_loss.item()
                train_info["policy_loss"] += policy_loss.item()

                train_info["dist_entropy"] += dist_entropy.item()
                train_info["task_entropy"] += task_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["critic_grad_norm"] += critic_grad_norm
                train_info["ratio"] += ratio.mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def train(self, buffer, turn_on=True):
        train_info = {}
        for train_func in self.train_list:
            train_info.update(train_func(buffer, turn_on))

        for optimizer in self.algo_module.optimizers.values():
            if hasattr(optimizer, "sync_lookahead"):
                optimizer.sync_lookahead()

        return train_info
