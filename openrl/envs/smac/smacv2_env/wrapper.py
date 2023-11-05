from openrl.envs.smac.smacv2_env.distributions import get_distribution
from openrl.envs.smac.smacv2_env.starcraft2 import StarCraft2Env, CannotResetException
from openrl.envs.smac.smacv2_env.multiagentenv import MultiAgentEnv

import numpy as np
import json

class StarCraftCapabilityEnvWrapper(MultiAgentEnv):
    def __init__(self, is_eval, env_id, **kwargs):
        self.distribution_config = kwargs["capability_config"]
        self.env_key_to_distribution_map = {}
        self._parse_distribution_config()
        self.env = StarCraft2Env(**kwargs)
        assert (
            self.distribution_config.keys()
            == kwargs["capability_config"].keys()
        ), "Must give distribution config and capability config the same keys"
        self.env_id = env_id
        self.is_eval = is_eval
        # get reset config 
        with open("config.json", "r") as f:
            data = json.load(f)
        self.all_config = data
        # get task embeding
        # data = np.load('task.npy')
        data = np.load('result.npy')
        self.task_emb = data.copy()
        del data

    def _parse_distribution_config(self):
        for env_key, config in self.distribution_config.items():
            if env_key == "n_units" or env_key == "n_enemies":
                continue
            config["env_key"] = env_key
            # add n_units key
            config["n_units"] = self.distribution_config["n_units"]
            config["n_enemies"] = self.distribution_config["n_enemies"]
            distribution = get_distribution(config["dist_type"])(config)
            self.env_key_to_distribution_map[env_key] = distribution

    def reset(self):
        try:
            # reset_config = {}
            # for distribution in self.env_key_to_distribution_map.values():
            #     reset_config = {**reset_config, **distribution.generate()}
            self.task_idx = self.env_id * 100 + np.random.randint(100)
            reset_config = self.all_config[str(self.task_idx)]
            reset_config["ally_start_positions"]["item"] = np.array(reset_config["ally_start_positions"]["item"])
            reset_config["enemy_start_positions"]["item"] = np.array(reset_config["enemy_start_positions"]["item"])
            self.env.reset(reset_config)
            return self.get_obs(), self.get_state()
        except CannotResetException as cre:
            raise NotImplementedError
            # # just retry
            # self.reset()

    def step(self, actions, extra_data=None):
        return self.env.step(actions)

    def __getattr__(self, name):
        if hasattr(self.env, name):
            return getattr(self.env, name)
        else:
            raise AttributeError

    def get_obs(self):
        obs = self.env.get_obs()
        task_emb = self.task_emb[self.task_idx:self.task_idx+1]
        task_emb = np.repeat(task_emb, len(obs), 0)
        obs = np.concatenate([task_emb, obs], -1)
        return obs

    def get_obs_feature_names(self):
        return self.env.get_obs_feature_names()

    def get_state(self):
        state = self.env.get_state()
        task_emb = self.task_emb[self.task_idx]
        state = np.concatenate([task_emb, state], -1)
        return state

    def get_state_feature_names(self):
        return self.env.get_state_feature_names()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_env_info(self):
        return self.env.get_env_info()

    def get_obs_size(self):
        return self.env.get_obs_size()

    def get_state_size(self):
        return self.env.get_state_size()

    def get_total_actions(self):
        return self.env.get_total_actions()

    def get_capabilities(self):
        return self.env.get_capabilities()

    def get_obs_agent(self, agent_id):
        return self.env.get_obs_agent(agent_id)

    def get_avail_agent_actions(self, agent_id):
        return self.env.get_avail_agent_actions(agent_id)

    def render(self, mode):
        return self.env.render(mode)

    def get_stats(self):
        return self.env.get_stats()

    def full_restart(self):
        return self.env.full_restart()

    def save_replay(self):
        self.env.save_replay()

    def close(self):
        return self.env.close()
         
