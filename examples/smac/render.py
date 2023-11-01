""""""

import numpy as np
from openrl.envs.smac.custom_vecinfo import SMACInfo
from openrl.envs.smac.smacv2_env import make_smac_envs

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.envs.vec_env.vec_info import VecInfoFactory
from openrl.envs.wrappers.monitor import Monitor
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent

import imageio
import cv2
import os
import time

VecInfoFactory.register("SMACInfo", SMACInfo)

env_wrappers = [
    Monitor,
]


def evaluation():
    
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()
    
    env_num = 1
    env = make(
        "10gen_protoss",
        env_num=env_num,
        make_custom_envs=make_smac_envs,
        cfg=cfg,
        env_wrappers=env_wrappers,
    )
    
    net = Net(env, cfg=cfg, device="cuda")
    agent = Agent(net, use_wandb=False, project_name="SMAC")
    agent.load("./results/baseline/best_model")
    agent.set_env(env)
    
    first_time = True
    
   
    
    for stalker_num in range(6):
        for zealot_num in range(6):
            for colossus_num in range(6):
                if stalker_num+zealot_num+colossus_num!=5:
                    continue
                
                team = []
                for _ in range(stalker_num):
                    team.append("stalker")
                for _ in range(zealot_num):
                    team.append("zealot")
                for _ in range(colossus_num):
                    team.append("colossus")
                team_name = str(stalker_num)+str(zealot_num)+str(colossus_num)
                env_dict = env.envs[0].env.env.env_key_to_distribution_map
                env_dict["team_gen"].set_team(team)
                
                if first_time:
                    first_time = False
                    obs, info = env.reset(seed=0)
    
                num_episodes = 100
                images, win = [], []
                for i in range(num_episodes):
                    total_reward = 0
                    done = False
                    while not np.any(done):
                        # Based on environmental observation input, predict next action.
                        action, _ = agent.act(obs, info=info, deterministic=True)
                        obs, r, done, info = env.step(action)
                        image = env.envs[0].env.env.render("rgb_array")
                        images.append(image)
                        total_reward += np.mean(r)
                    # print(f"total_reward: {total_reward}, win/lose:", info[0]["final_info"]["battle_won"])
                    win.append(info[0]["final_info"]["battle_won"])
                    
                    team_dir = 'results/images/{}'.format(team_name)
                    os.makedirs(team_dir, exist_ok=True)
                    imageio.mimsave(team_dir+'/result_{:02d}.gif'.format(i), images[:-1])
                    cv2.imwrite(team_dir+"/result_{:02d}.png".format(i), images[0])
                    
                    images = [images[-1]]
                    
                print("team", team, "win rate", np.array(win).mean())

    env.close()


def evaluation_single():
    
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()
    
    env_num = 1
    env = make(
        "10gen_protoss",
        env_num=env_num,
        make_custom_envs=make_smac_envs,
        cfg=cfg,
        env_wrappers=env_wrappers,
    )
    
    net = Net(env, cfg=cfg, device="cuda")
    agent = Agent(net, use_wandb=False, project_name="SMAC")
    agent.load("./results/single/best_model")
    agent.set_env(env)
    
    total_reward = 0
    done = False
    images = []
    obs, info = env.reset(seed=0)
    image = env.envs[0].env.env.render("rgb_array")
    images.append(image)    
    while not np.any(done):
        action, _ = agent.act(obs, info=info, deterministic=False)
        obs, r, done, info = env.step(action)
        image = env.envs[0].env.env.render("rgb_array")
        images.append(image)
        total_reward += np.mean(r)
    
    imageio.mimsave('result.gif', images[:-1])
    cv2.imwrite("result.png", images[0])

    env.close()


if __name__ == "__main__":
    from absl import flags

    FLAGS = flags.FLAGS
    FLAGS([""])
    start = time.time()
    evaluation_single()
    print("time", time.time()-start)
