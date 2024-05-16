def survival_critical_state(env, agent, observation):
    while not env.need_reset:
        # Check if there's a zombie and kill it to prevent health loss
        if any(obj['type'] == 'zombie' for obj in observation['surrounding']):
            observation = agent.do("Kill the zombie.")
            continue  # Re-evaluate the situation after the action
        
        # If health is critically low, killing a cow is a priority to restore food level
        if observation['inner']['food'] == 0:
            observation = agent.do("Find cows.")
            if not observation['surrounding']:  # If no cows found, try again
                continue
            observation = agent.do("Kill the cow.")
            continue  # Re-evaluate the situation after the action
        
        # If drink level is critically low, finding water and drinking is a priority
        if observation['inner']['drink'] == 1:
            observation = agent.do("Find water.")
            if not any(obj['type'] == 'water' for obj in observation['surrounding']):
                continue  # If no water found, try again
            observation = agent.do("Drink water.")
            continue  # Re-evaluate the situation after the action
        
        # If all critical levels are addressed, return from the function
        if observation['inner']['health'] > 1 and observation['inner']['food'] > 0 and observation['inner']['drink'] > 1:
            return env, agent

    return env, agent
