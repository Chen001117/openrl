""""""

import numpy as np
from openrl.envs.smac.custom_vecinfo import SMACInfo
from openrl.envs.smac.smacv2_env import make_smac_envs

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.envs.vec_env.vec_info import VecInfoFactory
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent

VecInfoFactory.register("SMACInfo", SMACInfo)

def evaluation():
    
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()

    env_num = 1
    env = make(
        "10gen_protoss",
        env_num=env_num,
        cfg=cfg,
        make_custom_envs=make_smac_envs,
    )

    # create the neural network
    net = Net(env, cfg=cfg, device="cuda")

    # initialize the trainer
    agent = Agent(net, use_wandb=False, project_name="SMAC")
    agent.load("results/best_model/best_model/module.pt")
    
    # agent.load("./ppo_agent/")
    agent.set_env(env)
    obs, info = env.reset(seed=0)
    done = False
    step = 0
    total_reward = 0
    imgs = []
    while not np.any(done):
        # Based on environmental observation input, predict next action.
        imgs.append(env.render())
        action, _ = agent.act(obs, info=info, deterministic=True)
        obs, r, done, info = env.step(action)
        step += 1
        total_reward += np.mean(r)
        # print(f"step:{step}, total_reward: {total_reward}")
    
    import imageio
    imageio.mimsave('render.gif', imgs, duration=10)
    
    print(f"total_reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    from absl import flags

    FLAGS = flags.FLAGS
    FLAGS([""])

    evaluation()
