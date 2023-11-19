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
        self.is_eval = is_eval

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
        
        reset_config = {}
        for distribution in self.env_key_to_distribution_map.values():
            reset_config = {**reset_config, **distribution.generate()}
        self.env.reset(reset_config)
        
        self.state_orders = []
        for _ in range(self.env.n_agents*2):
            order = np.arange(5)
            np.random.shuffle(order)
            self.state_orders.append(order)
        self.obs_orders = []
        for _ in range(self.env.n_agents*2):
            order = np.arange(5)
            np.random.shuffle(order)
            self.obs_orders.append(order)

    def __getattr__(self, name):
        if hasattr(self.env, name):
            return getattr(self.env, name)
        else:
            raise AttributeError

    def get_state(self):
        
        state = self.env.get_state()
        
        if self.is_eval:
            state = [[state] for _ in range(self.env.n_agents)]
            return np.concatenate(state, 0)
        
        states = []
        for order in self.state_orders:
            ally = state[:40].reshape([5,8])
            ally = ally[order].flatten()
            enemy = state[40:75].reshape([5,7])
            enemy = enemy[order].flatten()
            act = state[75:].reshape([5,11])
            act = act[order].flatten()
            s = np.concatenate([ally, enemy, act])
            states.append(s)
        states = np.array(states)
        
        return states
    
    def get_obs(self):
        
        obs = self.env.get_obs()
        
        if self.is_eval:
            return obs
        
        obs = np.concatenate([obs, obs])
        for i, order in enumerate(self.obs_orders):
            if i < self.env.n_agents:
                continue
            agent_idx = i % self.env.n_agents
            enemy = obs[i,4:49]
            enemy = enemy.reshape([self.env.n_agents,9])
            enemy = enemy[order].flatten()
            obs[i,4:49] = enemy
            aorder = order.copy()
            aorder = np.delete(aorder, np.argwhere(aorder==agent_idx))
            aorder = np.where(aorder>agent_idx,aorder-1,aorder)
            ally = obs[i,49:85]
            ally = ally.reshape([self.env.n_agents-1,9])
            ally = ally[aorder].flatten()
            obs[i,49:85] = ally
        
        return obs
    
    def get_avail_actions(self):
        
        action_mask = self.env.get_avail_actions()
        
        if self.is_eval:
            return action_mask
        
        action_mask = np.concatenate([action_mask, action_mask])
        
        return action_mask

    def get_obs_feature_names(self):
        return self.env.get_obs_feature_names()

    def get_state_feature_names(self):
        return self.env.get_state_feature_names()

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

    def step(self, actions):
        return self.env.step(actions)

    def get_stats(self):
        return self.env.get_stats()

    def full_restart(self):
        return self.env.full_restart()

    def save_replay(self):
        self.env.save_replay()

    def close(self):
        return self.env.close()
