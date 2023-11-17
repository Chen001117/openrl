from openrl.envs.smac.smacv2_env.smac_env import SMACEnv

class cfg():
    def __init__(self):
        self.map_name = "10gen_protoss"

my_cfg = cfg()
env = SMACEnv(False, my_cfg).env
_, obs = env.reset()
print("state shape", obs.shape)
env.close()