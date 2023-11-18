from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

FLAGS = flags.FLAGS
FLAGS(["main.py"])

import copy
from typing import Callable, List, Optional, Union

from gymnasium import Env

from openrl.configs.config import create_config_parser
from openrl.envs.common import build_envs

from openrl.envs.smac.smacv2_env.smac_env import SMACEnv


def smac_make(id, render_mode, disable_env_checker, **kwargs):
    
    is_eval = False
    if id[-5:] == "-eval":
        id = id[:-5]
        is_eval = True
    
    cfg_parser = create_config_parser()
    cfg_parser.add_argument(
        "--map_name", type=str, default=id, help="Which smac map to run on"
    )

    cfg = cfg_parser.parse_args([])

    env = SMACEnv(is_eval, cfg=cfg)

    return env


def make_smac_envs(
    id: str,
    env_num: int = 1,
    render_mode: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> List[Callable[[], Env]]:
    env_wrappers = copy.copy(kwargs.pop("env_wrappers", []))
    env_fns = build_envs(
        make=smac_make,
        id=id,
        env_num=env_num,
        render_mode=render_mode,
        wrappers=env_wrappers,
        **kwargs,
    )
    return env_fns
