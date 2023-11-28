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

VecInfoFactory.register("SMACInfo", SMACInfo)

env_wrappers = [
    Monitor,
]


def train():
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()

    # create environment
    env_num = 8
    env = make(
        "10gen_protoss",
        env_num=env_num,
        asynchronous=True,
        cfg=cfg,
        env_wrappers=env_wrappers,
    )

    # create the neural network

    net = Net(env, cfg=cfg, device="cuda")

    # initialize the trainer
    agent = Agent(net, use_wandb=True, project_name="SMAC")
    # start training, set total number of training steps to 5000000
    agent.train(total_time_steps=10000000)
    # agent.train(total_time_steps=2000)
    env.close()
    print("Saving agent to ./ppo_agent/")
    agent.save("./ppo_agent/")

    return agent


if __name__ == "__main__":
    from absl import flags

    FLAGS = flags.FLAGS
    FLAGS([""])

    agent = train()
