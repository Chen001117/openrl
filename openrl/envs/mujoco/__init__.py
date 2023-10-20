import copy
from typing import Callable, List, Optional, Union

from openrl.envs.common import build_envs
from openrl.envs.mujoco.navigation import NavigationEnv

from gymnasium import Env

def make_env(id: str, render_mode, disable_env_checker, **kwargs):
    if id.startswith("navigation"):
        num_agents = int(id[-1])
        env = NavigationEnv(num_agents=num_agents)
    else:
        raise ValueError("Unknown env {}".format(id))
    return env

def make_mujoco_envs(
    id: str,
    env_num: int = 1,
    render_mode: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> List[Callable[[], Env]]:

    env_wrappers = []
    
    env_fns = build_envs(
        make=make_env,
        id=id,
        env_num=env_num,
        render_mode=render_mode,
        wrappers=env_wrappers,
        **kwargs,
    )
    return env_fns