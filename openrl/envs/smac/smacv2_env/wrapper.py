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
        with open("config.json", "r") as f:
            data = json.load(f)
        self.task_potential = np.zeros(100)
        self.task_times = np.zeros(100)
        self.all_config = data
        self.value_list = []
        self.reward_list = []

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
            if self.is_eval:
                idx = self.env_id * 100 + np.random.randint(100)
                reset_config = self.all_config[str(idx)]
                reset_config["ally_start_positions"]["item"] = np.array(reset_config["ally_start_positions"]["item"])
                reset_config["enemy_start_positions"]["item"] = np.array(reset_config["enemy_start_positions"]["item"])
                return self.env.reset(reset_config)
            if len(self.reward_list) > 0:
                self.reward_list = np.array(self.reward_list)
                self.value_list = np.array(self.value_list)
                return2go = np.zeros_like(self.reward_list)
                return2go[-1] = self.reward_list[-1]
                gamma = 0.99
                for i in range(len(self.reward_list)-2,-1,-1):
                    return2go[i] = self.reward_list[i] + gamma * return2go[i+1]
                td_error = ((return2go-self.value_list)**2).mean()
                self.task_potential[self.idx] += td_error
                self.task_times[self.idx] += 1.
                self.reward_list = []
                self.value_list = []
            
            if (self.task_times<1.).any():
                prob = np.eye(100)[np.argmin(self.task_times)]
            else:
                beta = 0.1
                rho = 0.3
                avg_err = self.task_potential / self.task_times
                rank = np.zeros(100)
                sorted_idx = np.argsort(avg_err)
                rank[sorted_idx] = np.arange(100) + 1
                hs = (1/rank)**(1/beta)
                ps = hs / hs.sum()
                hc = self.task_times.sum() - self.task_times
                pc = hc / hc.sum()
                prob = (1-rho) * ps + rho * pc
                
            self.idx = np.random.choice(np.arange(100), p=prob)
            if self.env_id==0 and self.task_times.sum()%50==0:
                print("prob", prob)
            idx = self.env_id * 100 + self.idx
            reset_config = self.all_config[str(idx)]
            reset_config["ally_start_positions"]["item"] = np.array(reset_config["ally_start_positions"]["item"])
            reset_config["enemy_start_positions"]["item"] = np.array(reset_config["enemy_start_positions"]["item"])

            return self.env.reset(reset_config)
        except CannotResetException as cre:
            # just retry
            self.reset()

    def step(self, actions, extra_data=None):
        reward, terminated, info = self.env.step(actions)
        if extra_data is not None:
            value = extra_data["values"][self.env_id][0,0]
            self.value_list.append(value)
            self.reward_list.append(reward)
        return reward, terminated, info

    def __getattr__(self, name):
        if hasattr(self.env, name):
            return getattr(self.env, name)
        else:
            raise AttributeError

    def get_obs(self):
        return self.env.get_obs()

    def get_obs_feature_names(self):
        return self.env.get_obs_feature_names()

    def get_state(self):
        return self.env.get_state()

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
         
