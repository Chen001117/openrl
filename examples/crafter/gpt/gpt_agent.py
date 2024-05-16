import json

available_actions = [
    # "Find cows.", 
    # "Find water.", 
    # "Find stone.", 
    # "Find tree.",
    "Collect sapling.",
    "Place sapling.",
    "Chop tree.", 
    "Kill the cow.", 
    "Mine stone.", 
    "Drink water.",
    "Mine coal.", 
    "Mine iron.", 
    "Mine diamond.", 
    "Kill the zombie.",
    "Kill the skeleton.", 
    "Craft wood_pickaxe.", 
    "Craft wood_sword.",
    "Place crafting table.", 
    "Place furnace.", 
    "Craft stone_pickaxe.",
    "Craft stone_sword.", 
    "Craft iron_pickaxe.", 
    "Craft iron_sword.",
    "Sleep."
]

class GPTAgent():
    
    def __init__(self, env, agent):
        
        self.env = env
        self.agent = agent
        
        self.code = None
        self.trajectory = []
        
    def do(self, current_task):
        
        if current_task not in available_actions:
            raise ValueError(f"Invalid task: {current_task}")
        
        # get action 
        obs = self.env.env.set_task(self.env.obs, [current_task])
        action, _ = self.agent.act(obs, info=self.env.infos, deterministic=True, render=True)
        
        # save trajectory
        traj = {
            "text_obs": self.env.infos[0]["text_obs"],
            "action": int(action[0,0,0]),
            "task": current_task,
            "code": self.code,
        }
        self.trajectory.append(traj)
        
        # step forward
        text_obs, dict_obs = self.env.step(action, task=current_task)
        
        return dict_obs
    
    def save_trajectory(self, save_path):
        
        with open(save_path, 'w') as f:
            json.dump(self.trajectory, f)