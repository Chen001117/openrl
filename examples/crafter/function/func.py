
from PIL import Image, ImageDraw, ImageFont

def save_img(obs, task):
    img = obs["policy"]["image"][0, 0]
    img = img.transpose((1, 2, 0))
    img = Image.fromarray(img)
    img = img.resize((256, 256))
    draw = ImageDraw.Draw(img)
    draw.text((10,10), task, fill=(255,0,0))
    img.save("run_results/image.png")
    return img

def do(language, agent, env, info, trajectory):
    current_task = language
    action, _ = agent.act(info[0], deterministic=True)
    obs, r, done, info = env.step(action, given_task=[current_task])
    img = save_img(obs, current_task)
    trajectory.append(img)
    if all(done):
        trajectory[0].save(
            "run_results/crafter.gif", 
            save_all=True, 
            append_images=trajectory[1:], 
            duration=100, 
            loop=0
        )
        exit()
    return (obs, r, done, info)

def get_obs(info):
    return info[-1][0]

def CrafterAgent(agent, env, env_info, trajectory):
    loop_counter = 0

    # Check for water and drink if drink level is low
    while "water" not in get_obs(env_info) and loop_counter < 30:
        env_info = do("Find water.", agent, env, env_info, trajectory)
        loop_counter += 1
        if loop_counter >= 30:
            return env_info

    while "You drink water." not in get_obs(env_info) and loop_counter < 30:
        env_info = do("Drink water.", agent, env, env_info, trajectory)
        loop_counter += 1
        if loop_counter >= 30:
            return env_info

    # Sleep to restore energy
    while "You feel rested." not in get_obs(env_info) and loop_counter < 30:
        env_info = do("Sleep.", agent, env, env_info, trajectory)
        loop_counter += 1
        if loop_counter >= 30:
            return env_info
    
    # Kill the skeleton to prevent damage to health
    while "skeleton" in get_obs(env_info) and loop_counter < 30:
        env_info = do("Kill the skeleton.", agent, env, env_info, trajectory)
        loop_counter += 1
        if loop_counter >= 30:
            return env_info

    return env_info
