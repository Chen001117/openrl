from dataclasses import dataclass

import numpy as np
from os import path
from gymnasium import utils
from openrl.envs.mujoco.base_env import BaseEnv
from openrl.envs.mujoco.xml_gen import get_xml
from openrl.envs.mujoco.astar import find_path
from gymnasium.spaces import Box, Tuple, Dict
import time

@dataclass
class EnvSpec:
    id: str

class NavigationEnv(BaseEnv):
    spec = EnvSpec("")

    def __init__(self, num_agents, is_eval, **kwargs):
        
        # ablation
        self._re_order = True
        self._use_priveledge_info = True
        self._filter_prob = 0.75

        # hyper params:
        self._stucked_num = 100 # stucked steps
        self._stuck_threshold = 0.1 # m
        self._reach_threshold = 1. # m
        self._num_agents = num_agents # number of agents
        self._num_obstacles = 10 # number of obstacles
        self._domain_random_scale = 1e-1 # domain randomization scale
        self._measure_random_scale = 1e-2 # measurement randomization scale
        self._num_frame_skip = 10 # 100/frame_skip = decision_freq (Hz)
        self._map_real_size = 10. # map size (m)
        self._warm_step = 2 # warm-up: let everything stable (steps)
        self._num_astar_nodes = 20 # number of nodes for rendering astar path

        # simulator params
        self._init_kp = np.array([[2000, 2000, 800]]) # kp for PD controller
        self._init_kd = np.array([[0.02, 0.02, 1e-6]]) # kd for PD controller
        self._robot_size = np.array([0.65, 0.3, 0.3]) # m
        self._obstacle_size = np.array([1., 1., 1.]) # m
        self._load_size = np.array([0.6, 0.6, 0.6]) # m

        # initialize simulator
        self._anchor_id = np.random.randint(0, 4, self._num_agents)
        init_load_mass = 3. # kg
        self._load_mass = init_load_mass * (1 + self._rand(self._domain_random_scale))
        init_cable_len = np.ones(self._num_agents) # m
        self._cable_len = init_cable_len * (1 + self._rand(self._domain_random_scale))
        init_fric_coef = 1. # friction coefficient
        self._fric_coef = init_fric_coef * (1 + self._rand(self._domain_random_scale))
        self._astar_node = 20 # for rendering astar path
        model = get_xml(
            dog_num = self._num_agents, 
            obs_num = self._num_obstacles, 
            anchor_id = self._anchor_id,
            load_mass = self._load_mass,
            cable_len = self._cable_len,
            fric_coef = self._fric_coef,
            astar_node = self._num_astar_nodes
        )
        super().__init__(model, **kwargs)

        # RL space
        self._max_num_agents = 4
        local_observation_size = 5+11+self._max_num_agents*9+self._num_obstacles*5+20  
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(local_observation_size,), dtype=np.float64
        )
        
        global_state_size = 5+self._max_num_agents*9+self._num_obstacles*5+20
        priviledge_size = 10
        global_state_size = global_state_size+priviledge_size if self._use_priveledge_info else global_state_size
        share_observation_space = Box(
            low=-np.inf, high=np.inf, shape=(global_state_size,), dtype=np.float64
        )
        
        self.observation_space = Dict({
            "policy": observation_space,
            "critic": share_observation_space,
        })
        
        action_low = np.array([-0.15, -0.06, -np.pi/10])
        action_high = np.array([0.55, 0.06, np.pi/10])
        self.action_space = Box(
            low=action_low, high=action_high, shape=(3,), dtype=np.float64
        )
        
        # variables
        self._is_eval = is_eval
        self._max_time = 1e6
        self._prev_output_vel = np.zeros([self._num_agents, 3])
        bounds = self.model.actuator_ctrlrange.copy()
        torque_low, torque_high = bounds.astype(np.float32).T
        self._torque_low = torque_low.reshape([self._num_agents, self.action_space.shape[0]])
        self._torque_high = torque_high.reshape([self._num_agents, self.action_space.shape[0]])

    def reset(self, seed=None):
        # seed
        if seed:
            self.seed(seed)
        # init task specific parameters
        self._kp  = self._init_kp * (1 + self._rand(self._domain_random_scale, 3))
        self._kd  = self._init_kd * (1 + self._rand(self._domain_random_scale, 3))
        self._order = np.arange(self._num_agents)
        if self._re_order:
            np.random.shuffle(self._order)
        self._inverse_order = np.argsort(self._order)
        # reset simulator
        self._t = 0.
        self._hist_load_pos = np.zeros([self._stucked_num, 2]) - 100
        self._reset_simulator()
        # get rl info
        observation = self._get_obs()
        info = dict()
        # post process
        self._post_update()
        return observation, info
    
    def step(self, action):
        # step simulator
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = action[self._inverse_order]
        done = self._do_simulation(action, self._num_frame_skip)
        done = [done for _ in range(self._num_agents)]
        # get rl info
        observation = self._get_obs()
        reward, info = self._get_reward()
        reward = [[reward] for _ in range(self._num_agents)]
        # post process
        self._post_update()
        return observation, reward, done, info
        
    def _reset_simulator(self):
        # initialize obstacles
        init_obstacles_pos = self._rand(self._map_real_size, [self._num_obstacles, 2])
        init_obstacles_yaw = self._rand(2*np.pi, [self._num_obstacles, 1])
        init_obstacles_z = np.ones([self._num_obstacles, 1]) * self._obstacle_size[2]/2
        # initialize load
        init_load_pos = self._rand(self._map_real_size, [1, 2])
        init_load_yaw = self._rand(2*np.pi, [1, 1])
        init_load_z = np.ones([1, 1]) * self._load_size[2]/2
        # initialize dogs
        init_anchor_robot_len = self._rand(self._domain_random_scale, [self._num_agents, 1]) + 1.
        init_anchor_robot_yaw = self._rand(np.pi, [self._num_agents, 1])
        init_anchor_id = self._anchor_id.reshape([self._num_agents, 1])
        init_anchor_pos = self._get_toward(init_load_yaw)[0] * 0.3 * (init_anchor_id==0)
        init_anchor_pos += self._get_toward(init_load_yaw)[1] * 0.3 * (init_anchor_id==1)
        init_anchor_pos += self._get_toward(init_load_yaw)[0] * (-0.3) * (init_anchor_id==2)
        init_anchor_pos += self._get_toward(init_load_yaw)[1] * (-0.3) * (init_anchor_id==3)
        init_load_robot_yaw = init_load_yaw + init_anchor_id * np.pi/2 + init_anchor_robot_yaw
        init_anchor_robot_pos = self._get_toward(init_load_robot_yaw)[0] * init_anchor_robot_len
        init_robot_pos = init_load_pos + init_anchor_pos + init_anchor_robot_pos
        init_robot_yaw = self._rand(2*np.pi, [self._num_agents, 1])
        init_robot_z = np.ones([self._num_agents, 1]) * self._robot_size[2]/2
        # initialize goal
        dist = 0
        while dist < 1.:
            self._goal = self._rand(self._map_real_size-1, [1, 2])
            dist = np.expand_dims(init_obstacles_pos, 0) - np.expand_dims(self._goal, 1)
            dist = np.linalg.norm(dist, axis=-1).min()
        # astar search
        # astar_path = self._astar_search(init_load_pos, self._goal, init_obstacles_pos) # TODO
        astar_path = np.arange(self._num_astar_nodes)
        astar_path = astar_path.reshape([self._num_astar_nodes,1])
        astar_path = astar_path / (self._num_astar_nodes-1.)
        astar_path = init_load_pos + (self._goal-init_load_pos) * astar_path
        if astar_path is None:
            return self._reset_simulator()
        # filter simple path
        dist = np.expand_dims(astar_path, 0) - np.expand_dims(init_obstacles_pos, 1)
        dist = np.linalg.norm(dist, axis=-1)
        if dist.min() > 0.8 and np.random.rand() < self._filter_prob:
            return self._reset_simulator()
        # set state
        qpos = self.sim.data.qpos.copy()
        load = np.concatenate([init_load_pos, init_load_yaw, init_load_z], axis=-1).flatten()
        robots = np.concatenate([init_robot_pos, init_robot_yaw, init_robot_z], axis=-1).flatten()
        obstacles = np.concatenate([init_obstacles_pos, init_obstacles_yaw, init_obstacles_z], axis=-1).flatten()
        qpos[:-self._num_astar_nodes*2-12] = np.concatenate([load, robots, obstacles]) 
        qpos[-self._num_astar_nodes*2:] = astar_path.flatten()
        self.set_state(qpos, np.zeros_like(qpos))
        # warm-up
        for i in range(self._warm_step):
            terminated = self._do_simulation(0, self._num_frame_skip, add_time=False)
            if terminated:
                return self._reset_simulator()
        # astar check
        if self._is_eval:
            obs_map = self._draw_obs_map(
                init_obstacles_pos, init_obstacles_yaw, self._obstacle_size[:2]
            )
            astar_path = find_path(init_load_pos[0], self._goal[0], obs_map)
            if astar_path is None:
                return self._reset_simulator()
        dist = np.linalg.norm(self._goal-init_load_pos, axis=-1)[0]
        self._max_time = dist * 7 + 7
    
    def _do_simulation(self, action, num_frame_skip, add_time=True):
        for i in range(num_frame_skip):
            if self._get_done():
                return True
            robot_global_velocity = self._get_state("robot", vel=True)
            local_vel = self._global2local(robot_global_velocity)
            torque = self._pd_controller(target=action, cur_vel=local_vel)
            self.sim.data.ctrl[:] = self._local2global(torque).flatten()
            self.sim.step()
            self._t = self._t + self.model.opt.timestep if add_time else self._t
        terminated = self._get_done()
        return terminated

    def _get_obs(self):
        # get load info
        load_state = self._get_state("load")
        load_pos = load_state[:,:2] - self._goal
        load_pos += self._rand(self._measure_random_scale, load_pos.shape)
        load_pos /= self._map_real_size
        load_pos = self._cart2polar(load_pos)
        load_yaw = load_state[:,2:3]
        load_yaw += self._rand(self._measure_random_scale, load_yaw.shape)
        load = np.concatenate([load_pos, np.cos(load_yaw), np.sin(load_yaw)], axis=-1)
        load = load.reshape([1, -1])
        load = load.repeat(self._num_agents, axis=0)
        # get robot info
        robot_state = self._get_state("robot")
        robot_pos = robot_state[:,:2] - self._goal
        robot_pos += self._rand(self._measure_random_scale, robot_pos.shape)
        robot_pos /= self._map_real_size
        robot_pos = self._cart2polar(robot_pos)
        robot_yaw = robot_state[:,2:3]
        robot_yaw += self._rand(self._measure_random_scale, robot_yaw.shape)
        robot = np.concatenate([robot_pos, np.cos(robot_yaw), np.sin(robot_yaw)], axis=-1)
        # get anchor info
        anchor = np.eye(4)[self._anchor_id]
        anchor = anchor.reshape([self._num_agents, 4])
        # get obstacle info
        obstacle_state = self._get_state("obstacle")
        obstacle_pos = obstacle_state[:,:2] - self._goal
        obstacle_pos += self._rand(self._measure_random_scale, obstacle_pos.shape)
        obstacle_pos /= self._map_real_size
        obstacle_pos = self._cart2polar(obstacle_pos)
        obstacle_yaw = obstacle_state[:,2:3]
        obstacle_yaw += self._rand(self._measure_random_scale, obstacle_yaw.shape)
        obstacle = np.concatenate([obstacle_pos, np.cos(obstacle_yaw), np.sin(obstacle_yaw)], axis=-1)
        obstacle = obstacle.reshape([1, -1])
        obstacle = obstacle.repeat(self._num_agents, axis=0)
        # get wall info
        wall_state = self._get_state("wall")
        wall_pos = wall_state[:,:2] - self._goal
        wall_pos += self._rand(self._measure_random_scale, wall_pos.shape)
        wall_pos /= self._map_real_size
        wall_pos = self._cart2polar(wall_pos)
        wall_yaw = wall_state[:,2:3]
        wall_yaw += self._rand(self._measure_random_scale, wall_yaw.shape)
        wall = np.concatenate([wall_pos, np.cos(wall_yaw), np.sin(wall_yaw)], axis=-1)
        wall = wall.reshape([1, -1])
        wall = wall.repeat(self._num_agents, axis=0)
        
        # get all robots info
        robot_state = robot.copy()[self._order].reshape([1, -1])
        anchor_state = anchor.copy()[self._order].reshape([1, -1])
        place_holder = np.zeros([1, 9*(4-self._num_agents)])
        robot_state = np.concatenate([robot_state, anchor_state, place_holder], axis=-1)
        robot_state = np.repeat(robot_state, self._num_agents, axis=0)
        
        # theta(goal2robot) - phi(robot_yaw)
        robot_pos = self._get_state("robot")[:,:2] - self._goal
        theta = np.arctan2(robot_pos[:,1:2], robot_pos[:,0:1])
        phi = self._get_state("robot")[:,2:3]
        robot_rotate = np.concatenate([np.cos(theta-phi),np.sin(theta-phi)], axis=-1)
        
        # get local observation
        local_observation = np.concatenate([
            load, robot, robot_rotate, anchor, obstacle, wall, robot_state
        ], axis=-1)[self._order]
        # get global state
        global_state = np.concatenate([load, robot_state, obstacle, wall], axis=-1)
        # priviledge info
        if self._use_priveledge_info:
            priviledge = self._get_priviledge_info()
            global_state = np.concatenate([global_state, priviledge], axis=-1)
        
        observation = {
            "policy": local_observation,
            "critic": global_state,
        }

        return observation
    
    def _get_reward(self):
        
        if self._is_eval:
            done = self._get_done()
            load_pos = self._get_state("load")[:,:2]
            load_dist = np.linalg.norm(load_pos-self._goal)
            rewards = (load_dist<self._reach_threshold)*done*1.
            info = {"reach": rewards}
            return rewards, info
        
        # weights
        weight = np.array([1., 1.])
        weight = weight / weight.sum()
        # get rewards
        rewards = []
        # sparse reward
        duration = self._num_frame_skip / 100
        max_speed = np.linalg.norm(self.action_space.high[:2])
        load_pos = self._get_state("load")[:,:2]
        load_dist = np.linalg.norm(load_pos-self._goal)
        reach_goal = load_dist < self._reach_threshold
        rewards.append(reach_goal*duration*max_speed)
        # dense reward
        dense_rew = (self._last_load_dist-load_dist)*(1-reach_goal)
        rewards.append(dense_rew)
        # calculate reward
        info = {
            "dense": rewards[0],
            "sparse": rewards[1],
            "final_dist": load_dist,
        }
        rewards = np.array(rewards)
        rewards = np.dot(weight, rewards)
        return rewards, info

    def _get_done(self):
        # contact done
        for i in range(self.sim.data.ncon):
            con = self.sim.data.contact[i]
            obj1 = self.sim.model.geom_id2name(con.geom1)
            obj2 = self.sim.model.geom_id2name(con.geom2)
            if obj1=="floor" or obj2=="floor":
                continue
            else:
                # print("contact with {} and {}".format(obj1, obj2))
                return True
        # out of time
        if self._t > self._max_time:
            return True
        # stucked
        load_pos = self._get_state("load")[:,:2]
        load_move = np.linalg.norm(load_pos[0]-self._hist_load_pos[-1])
        if load_move < self._stuck_threshold:
            load_dist = np.linalg.norm(load_pos-self._goal)
            if load_dist > self._reach_threshold:
                return True
        # load rope contact done
        load_pos = self._get_state("load")[:,:2]
        load_yaw = self._get_state("load")[:,2]
        robot_pos = self._get_state("robot")[:,:2]
        anchor_yaw = self._anchor_id * np.pi/2 + load_yaw
        anchor_pos = np.stack([
            np.cos(anchor_yaw)*self._load_size[0]/2+load_pos[:,0],
            np.sin(anchor_yaw)*self._load_size[1]/2+load_pos[:,1],
        ], axis=-1)
        anchor2robot = robot_pos - anchor_pos
        anchor2robot_yaw = np.arctan2(anchor2robot[:,1], anchor2robot[:,0])
        cos_dist = np.cos(anchor2robot_yaw)*np.cos(anchor_yaw)
        cos_dist += np.sin(anchor2robot_yaw)*np.sin(anchor_yaw)
        if cos_dist < 0:
            return True
        return False

    def _post_update(self):
        load_pos = self._get_state("load")[:,:2]
        self._last_load_dist = np.linalg.norm(load_pos-self._goal)
        self._hist_load_pos[1:] = self._hist_load_pos[:-1]
        self._hist_load_pos[0] = load_pos

    # return x ~ U[-0.5*scale, 0.5*scale]
    def _rand(self, scale, size=None):
        if size is None:
            return (np.random.random()-0.5)*scale
        else:
            return (np.random.random(size)-0.5)*scale

    def _cart2polar(self, x):
        r = np.linalg.norm(x, axis=-1, keepdims=True)
        theta = np.arctan2(x[:,1:2], x[:,0:1])
        cos, sin = np.cos(theta), np.sin(theta)
        return np.concatenate([r, cos, sin], axis=-1)

    def _get_state(self, data_name, vel=False):

        if vel:
            state = self.sim.data.qvel.copy()
        else:
            state = self.sim.data.qpos.copy()

        if data_name == 'load':
            state = state[:4]
            state = state.reshape([1, 4])
        elif data_name == 'robot':
            state = state[4:4+self._num_agents*4]
            state = state.reshape([self._num_agents, 4])
        elif data_name == 'obstacle':
            state = state[4+self._num_agents*4:4+self._num_agents*4+self._num_obstacles*4]
            state = state.reshape([self._num_obstacles, 4])
        elif data_name == "wall":
            state = state[4+self._num_agents*4+self._num_obstacles*4:4+self._num_agents*4+self._num_obstacles*4+12]
            state = state.reshape([4, 3])
        else:
            raise NotImplementedError

        return state[:,:3]

    def _get_toward(self, theta):
        forwards = np.concatenate([np.cos(theta), np.sin(theta)], axis=-1)
        verticals = np.concatenate([-np.sin(theta), np.cos(theta)], axis=-1)
        return forwards, verticals

    def _local2global(self, actions):
        actions = actions.reshape([self._num_agents, 3])
        robot_yaw = self._get_state("robot")[:,2:3]
        forwards, verticals = self._get_toward(robot_yaw)
        actions_linear = actions[:,0:1] * forwards + actions[:,1:2] * verticals 
        actions_angular = actions[:,2:3]
        actions = np.concatenate([actions_linear, actions_angular], axis=-1)
        return actions

    def _global2local(self, actions):
        actions = actions.reshape([self._num_agents, 3])
        actions_linear = actions[:,:2]
        actions_angular = actions[:,2:3]
        robot_yaw = self._get_state("robot")[:,2]
        rot_mat = np.array([[np.cos(robot_yaw), np.sin(robot_yaw)],[-np.sin(robot_yaw), np.cos(robot_yaw)]])
        rot_mat = np.transpose(rot_mat, [2,0,1])
        actions_linear = np.expand_dims(actions_linear, 2)
        actions_linear = rot_mat @ actions_linear
        actions_linear = actions_linear.squeeze(2) 
        actions = np.concatenate([actions_linear, actions_angular], -1)
        return actions

    def _pd_controller(self, target, cur_vel):
        self._error_vel = target - cur_vel
        self._d_output_vel = cur_vel - self._prev_output_vel
        torque = self._kp * self._error_vel 
        torque += self._kd * self._d_output_vel/self.model.opt.timestep
        torque = np.clip(torque, self._torque_low, self._torque_high)
        self._prev_output_vel = cur_vel.copy()
        return torque
    
    def _get_priviledge_info(self):
        # simulation info
        time = (self._max_time - self._t) / 20.
        load = self._load_mass / 3.
        fric = self._fric_coef
        sim_info = np.array([[time, load, fric]])
        sim_info = np.repeat(sim_info, self._num_agents, axis=0)
        cable = self._cable_len.reshape([-1, 1])[self._order]
        sim_info = np.concatenate([sim_info, cable], axis=-1)
        # controller info
        control_info = np.concatenate([
            self._kp/self._init_kp, 
            self._kd/self._init_kd
        ], axis=-1)
        control_info = np.repeat(control_info, self._num_agents, axis=0)
        priviledge_info = np.concatenate([sim_info, control_info], axis=-1)
        return priviledge_info
        
    
    def seed(self, seed):
        np.random.seed(seed)

    @property
    def agent_num(self):
        return self._num_agents






    
    def _draw_obs_map(self, box_pos, box_yaw, box_len):
        """ draw dog/box/obstacle local map

        Args:
            dog_pos (np.array): size = [2]
            dog_theta (float): 
            box_pos (np.array): size = [box_num, 2]
            box_yaw (np.array): size = [box_num, 1]
            box_len (np.array): size = [2]
        """
        box_len /= 2
        self.lmlen, self.mlen = 47, 500
        box_pos = box_pos
        box_tow, box_ver = self._get_toward(box_yaw)
        box_rect = self._get_rect(box_pos, box_tow, box_ver, box_len)
        box_map = np.zeros([self.mlen, self.mlen])
        min_idx = -self.lmlen*3//2
        max_idx = self.mlen + self.lmlen*3//2 + 1
        for rect in box_rect:
            norm_rect = (rect.copy() / self._map_real_size + .5) * self.mlen
            norm_rect = norm_rect.astype('long')
            tmp_map = self._draw_rect(norm_rect, min_idx, max_idx)
            box_map += tmp_map[-min_idx:self.mlen-min_idx,-min_idx:self.mlen-min_idx]
        box_map = (box_map!=0) * 1.
        
        return box_map

    def _draw_map(self, dog_pos, dog_theta, box_pos, box_yaw, box_len):
        """ draw dog/box/obstacle local map

        Args:
            dog_pos (np.array): size = [2]
            dog_theta (float): 
            box_pos (np.array): size = [box_num, 2]
            box_yaw (np.array): size = [box_num, 1]
            box_len (np.array): size = [2]
        """
        box_len /= 2
        
        box_pos = box_pos - dog_pos.reshape([1,2])
        box_tow, box_ver = self._get_toward(box_yaw)
        box_rect = self._get_rect(box_pos, box_tow, box_ver, box_len)
        rotate_mat = np.array([[
            [np.cos(dog_theta), -np.sin(dog_theta)], 
            [np.sin(dog_theta), np.cos(dog_theta)]
        ]])
        box_rect = box_rect @ rotate_mat
        box_map = np.zeros([self.lmlen, self.lmlen])
        min_idx = self.mlen//2 - self.lmlen*3//2
        max_idx = self.mlen//2 + self.lmlen*3//2 + 1
        for rect in box_rect:
            norm_rect = (rect.copy() / self.msize + .5) * self.mlen
            norm_rect = norm_rect.astype('long')
            m = self._draw_rect(norm_rect, min_idx, max_idx)
            box_map += \
                m[0::3,0::3] + m[1::3,0::3] + m[2::3,0::3] + \
                m[0::3,1::3] + m[1::3,1::3] + m[2::3,1::3] + \
                m[0::3,2::3] + m[1::3,2::3] + m[2::3,2::3] 
        box_map = (box_map!=0) * 1.
        
        return box_map
        
    def _get_toward(self, theta):
        # theta : [env_num, 1]

        forward = np.zeros([*theta.shape[:-1], 1, 2])
        vertical = np.zeros([*theta.shape[:-1], 1, 2])
        forward[...,0] = 1
        vertical[...,1] = 1
        cos = np.cos(theta)
        sin = np.sin(theta)
        rotate = np.concatenate([cos, sin, -sin, cos], -1)
        rotate = rotate.reshape([*theta.shape[:-1], 2, 2])
        forward = forward @ rotate
        vertical = vertical @ rotate
        forward = forward.squeeze(-2)
        vertical = vertical.squeeze(-2)

        return forward, vertical
    
    def _get_rect(self, cen, forw, vert, length):
        '''
        cen(input): [*shape, 2] 
        forw(input): [*shape, 2]
        vert(input): [*shape, 2]
        length(input): [*shape, 2]
        vertice(output): [*shape, 4, 2]
        '''

        vertice = np.stack([
            cen + forw * length[...,0:1] + vert * length[...,1:2],
            cen - forw * length[...,0:1] + vert * length[...,1:2],
            cen - forw * length[...,0:1] - vert * length[...,1:2],
            cen + forw * length[...,0:1] - vert * length[...,1:2],
        ], axis=-2)
        
        return vertice

    def _draw_rect(self, rect, min_idx, max_idx):
        
        # get line function y = ax + b
        lines = [[rect[i], rect[(i+1)%4]] for i in range(4)]
        lines = np.array(lines)
        a = (lines[:,0,1] - lines[:,1,1]) / (lines[:,0,0] - lines[:,1,0])
        b = lines[:,0,1] - a * lines[:,0,0]
        # get x,y coor idx
        length = max_idx - min_idx
        x_axis = np.arange(length) + .5 + min_idx #[0]
        y_axis = np.arange(length) + .5 + min_idx #[1]
        y_map, x_map = np.meshgrid(y_axis, x_axis)
        if (a > 10).any():
            x_min = np.min(lines[:,:,0])
            x_max = np.max(lines[:,:,0])
            y_min = np.min(lines[:,:,1])
            y_max = np.max(lines[:,:,1])
            b_map = (x_map>x_min) & (x_map<x_max) \
                & (y_map>y_min) & (y_map<y_max)
            return b_map.astype('long')
        x_map = np.expand_dims(x_map, 0) 
        x_map = np.repeat(x_map, 4, 0)
        y_map = np.expand_dims(y_map, 0) 
        y_map = np.repeat(y_map, 4, 0)
        b_map = np.zeros_like(y_map).astype('bool')
        # fill rect 
        for i in range(4):
            b_map[i] = y_map[i] > a[i] * x_map[i] + b[i]
        b_map[0] = b_map[0] ^ b_map[2]
        b_map[1] = b_map[1] ^ b_map[3]
        b_map[0] = b_map[0] & b_map[1]
        return b_map[0].astype('long')
