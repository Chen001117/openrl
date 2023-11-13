from openrl.envs.smac.smacv2_env.distributions import get_distribution
from openrl.envs.smac.smacv2_env.starcraft2 import StarCraft2Env, CannotResetException
from openrl.envs.smac.smacv2_env.multiagentenv import MultiAgentEnv

import numpy as np

class StarCraftCapabilityEnvWrapper(MultiAgentEnv):
    def __init__(self, is_eval, **kwargs):
        self.distribution_config = kwargs["capability_config"]
        self.env_key_to_distribution_map = {}
        self._parse_distribution_config()
        self.env = StarCraft2Env(**kwargs)
        assert (
            self.distribution_config.keys()
            == kwargs["capability_config"].keys()
        ), "Must give distribution config and capability config the same keys"
        
        self.first_time = True
        self._is_eval = is_eval
        
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self._Nmin = 200
        self._p = 0.5 # prob to choose new task
        self._rho = 0.1 # importance of visit cnt
        self._beta = 0.1
        self._score = []
        self._cnt = []
        self._config = []
        
        assert self._Nmin > 1
        

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
        if self._is_eval:
            reset_config = {}
            for distribution in self.env_key_to_distribution_map.values():
                reset_config = {**reset_config, **distribution.generate()}
            self._config_idx = len(self._config)
        else:
            # calculate td error
            if len(self._config) > 0:
                td_error = self.td_error / self.episode_len
                prev_score = self._score[self._config_idx] * self._cnt[self._config_idx]
                self._score[self._config_idx] = prev_score + td_error
                self._cnt[self._config_idx] += 1.
                self._score[self._config_idx] /= self._cnt[self._config_idx]
            # collect new level
            if np.random.rand() < self._p or len(self._cnt) < self._Nmin:
                reset_config = {}
                for distribution in self.env_key_to_distribution_map.values():
                    reset_config = {**reset_config, **distribution.generate()}
                if not self.first_time:
                    self._config_idx = len(self._config)
                    self._config.append(reset_config)
                    self._score.append(0.)
                    self._cnt.append(0.)
                self.first_time = False
            # sample from buffer
            else:
                score = np.array(self._score)
                score = np.argsort(np.argsort(-score))
                score = (1/(score+1))**(1/self._beta)
                p_s = score / score.sum()
                cnt = np.array(self._cnt)
                cnt = cnt.sum() - cnt
                p_c = cnt / cnt.sum()
                prob = (1-self._rho) * p_s + self._rho * p_c
                idx = np.arange(len(prob))
                self._config_idx = np.random.choice(idx, size=1, p=prob)[0]
                reset_config = self._config[self._config_idx]
            
            self.last_value = None
            self.last_reward = None
            self.td_error = 0.
            self.episode_len = 0.
            
        return self.env.reset(reset_config)

    def step(self, step_inputs):
        actions, values = step_inputs
        reward, terminated, info = self.env.step(actions)
        if not self._is_eval:
            if self.last_value is not None:
                target = self.last_reward + self.gamma * values.mean()
                self.td_error += np.abs(target - self.last_value)
                self.episode_len += 1.
            self.last_value = values.mean()
            self.last_reward = reward
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

    def render(self):
        return self.env.render()

    def get_stats(self):
        return self.env.get_stats()

    def full_restart(self):
        return self.env.full_restart()

    def save_replay(self):
        self.env.save_replay()

    def close(self):
        return self.env.close()
