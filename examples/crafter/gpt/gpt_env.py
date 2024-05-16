from PIL import Image, ImageDraw, ImageFont
import json

class GPTEnv():
    
    def __init__(self, env):
        
        self.env = env
        self.img_history = [] 
        self.task_history = []
        self.total_reward = 0.
        
        self.gpt_counter = 0
        self._need_reset = False
        self.is_done = False
        
        self.obs = None
        self.infos = None
        
        self.env_steps = 0
    
    def reset(self):
        
        self.obs, self.infos = self.env.reset()
        self.img_history = [] 
        self.task_history = []
        self.obs_history = []
        self.img_history.append(self.obs["policy"]["image"][0, 0])
        
        self.gpt_counter = 0
        self._need_reset = False
        self.is_done = False
        
        self.env_steps = 0
        
        text_obs = self.infos[0]["text_obs"] 
        dict_obs = self.infos[0]["dict_obs"]
        
        self.obs_history.append(dict_obs)
        
        return text_obs, dict_obs
    
    def step(self, action, task):
        
        self.obs, r, done, self.infos = self.env.step(action)
        self.img_history.append(self.obs["policy"]["image"][0, 0])
        self.task_history.append(task)
        
        self.total_reward += r[0][0,0]
        self.gpt_counter += 1
        self._need_reset = self.gpt_counter > 20 or done[0]
        self.is_done = done[0]
        
        text_obs = self.infos[0]["text_obs"] 
        dict_obs = self.infos[0]["dict_obs"]
        
        self.obs_history.append(dict_obs)
        
        self.env_steps += 1
        
        return text_obs, dict_obs
    
    @property
    def need_reset(self):
        self.query_counter += 1
        if self.query_counter > 100:
            assert False, "infinite loops"
        return self._need_reset
    
    def reset_query_counter(self):
        
        self.gpt_counter = 0
        self.query_counter = 0
        self._need_reset = False
        
    def save_obs(self, save_gif):
        # save self.history_obs as json fileqq
        with open(save_gif, "w") as f:
            json.dump(self.obs_history, f)
    
    def save_gif(self, save_path):
        self.task_history.append("You are dead.")
        imgs = []
        for img, task in zip(self.img_history, self.task_history):
            img = img.transpose((1, 2, 0))
            img = Image.fromarray(img)
            img = img.resize((256, 256))
            draw = ImageDraw.Draw(img)
            draw.text((10,10), task, fill=(255,0,0))
            imgs.append(img)
        imgs[0].save(
            save_path, 
            save_all=True,
            append_images=imgs[1:],
            duration=100,
            loop=0
        )