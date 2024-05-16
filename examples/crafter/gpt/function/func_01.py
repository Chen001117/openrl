def survival(env, agent, observation):
    if not hasattr(agent, 'inventory'):
        agent.inventory = {"plant": 1}
    if "plant" in agent.inventory:
        agent.do("Eat plant.")
    else:
        agent.do("Find water.")
    return env, agent
