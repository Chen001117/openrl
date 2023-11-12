import copy
from typing import Callable, List, Optional, Union

from openrl.envs.common import build_envs
from openrl.envs.mujoco.navigation import NavigationEnv

from gymnasium import Env

def make_env(id: str, env_id, render_mode, disable_env_checker, **kwargs):
    if id.startswith("navigation"):
        is_eval = False
        if id[-5:] == "-eval":
            id = id[:-5]
            env_id += 1000 # TODO
            is_eval = True
        num_agents = int(id[-1])
        env = NavigationEnv(num_agents=num_agents, is_eval=is_eval, env_id=env_id)
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