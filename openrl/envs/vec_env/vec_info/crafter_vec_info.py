import time
from typing import Any, Dict

import numpy as np

from openrl.envs.vec_env.vec_info.base_vec_info import BaseVecInfo

from collections import deque


class CrafterVecInfo(BaseVecInfo):
    def __init__(self, parallel_env_num: int, agent_num: int):
        super().__init__(parallel_env_num, agent_num)

        self.episode_infos = []

        self.start_time = time.time()
        self.total_step = 0

        self.log_items = [
            "alpha",
            "kl_div",
            "kl_rewards",
            "original_rewards",
            "instruction_following_rewards",
        ]
        
        self.episode_rewards = deque(maxlen=256)
        self.episode_length = deque(maxlen=256)
        
        # achievement
        self.diamond_suc = deque(maxlen=256)
        self.iron_sword_suc = deque(maxlen=256)
        self.iron_pickaxe_suc = deque(maxlen=256)

    def statistics(self, buffer: Any) -> Dict[str, Any]:
        # get agent's episode reward
        rewards = buffer.data.rewards.copy()
        self.total_step += np.prod(rewards.shape[:2])
        rewards = rewards.transpose(2, 1, 0, 3)
        info_dict = {}
        ep_rewards = []
        for i in range(self.agent_num):
            agent_reward = rewards[i].mean(0).sum()
            ep_rewards.append(agent_reward)
            info_dict["agent_{}/episode_reward".format(i)] = agent_reward

        # get episode reward
        info_dict["FPS"] = int(self.total_step / (time.time() - self.start_time))
        info_dict["episode_reward"] = np.mean(ep_rewards)

        # get env info
        new_infos = dict()
        for step_infos in self.episode_infos:
            for env_info in step_infos:
                for key in env_info:
                    if key in self.log_items:
                        if key in new_infos:
                            new_infos[key].append(env_info[key])
                        else:
                            new_infos[key] = [env_info[key]]
                if "final_info" in env_info:
                    for key in env_info["final_info"]:
                        if key in self.log_items:
                            if key in new_infos:
                                new_infos[key].append(env_info["final_info"][key])
                            else:
                                new_infos[key] = [env_info["final_info"][key]]
        for key in new_infos:
            new_infos[key] = np.array(new_infos[key]).mean()
        info_dict.update(new_infos)
        
        for step_info in self.episode_infos:
            for singe_env_info in step_info:
                assert isinstance(singe_env_info, dict), "singe_env_info must be dict"

                if (
                    "final_info" in singe_env_info.keys()
                    and "episode" in singe_env_info["final_info"].keys()
                ):
                    self.episode_rewards.append(
                        singe_env_info["final_info"]["episode"]["r"]
                    )   
                    self.episode_length.append(
                        singe_env_info["final_info"]["episode"]["l"]
                    )
                    self.diamond_suc.append(
                        (singe_env_info["final_info"]["dict_obs"]["inventory"]["diamond"] > 0) * 1.
                    )
                    self.iron_sword_suc.append(
                        (singe_env_info["final_info"]["dict_obs"]["inventory"]["iron_sword"] > 0) * 1.
                    )
                    self.iron_pickaxe_suc.append(
                        (singe_env_info["final_info"]["dict_obs"]["inventory"]["iron_pickaxe"] > 0) * 1.
                    )
        
        if len(self.episode_rewards) > 0:
            info_dict["episode_rewards_mean"] = np.mean(self.episode_rewards)
            info_dict["episode_rewards_median"] = np.median(self.episode_rewards)
            info_dict["episode_rewards_min"] = np.min(self.episode_rewards)
            info_dict["episode_rewards_max"] = np.max(self.episode_rewards)
            info_dict["episode_length_mean"] = np.mean(self.episode_length)
            info_dict["episode_length_median"] = np.median(self.episode_length)
            info_dict["episode_length_min"] = np.min(self.episode_length)
            info_dict["episode_length_max"] = np.max(self.episode_length)
            info_dict["diamond_suc_rate"] = np.mean(self.diamond_suc)
            info_dict["iron_sword_suc_rate"] = np.mean(self.iron_sword_suc)
            info_dict["iron_pickaxe_suc_rate"] = np.mean(self.iron_pickaxe_suc)

        return info_dict

    def append(self, info: Dict[str, Any]) -> None:
        self.episode_infos.append(info)

    def reset(self) -> None:
        self.episode_infos = []
        self.rewards = []
