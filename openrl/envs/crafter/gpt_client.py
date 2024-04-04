from openai import AsyncOpenAI
from openai import OpenAI

import asyncio

class GPTClient:
    
    def __init__(self, api_key, api_base, model):
        openai_api_key = api_key #"isQQQqPJUUSWXvz4NqG36Q6v5pxdPTkG"
        openai_api_base = api_base #"https://azure-openai-api.shenmishajing.workers.dev/v1"
        self.model_name = model
        self.client = AsyncOpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        
        self.cal_approx_cost = True
        self.num_prompt = 0
        self.num_output = 0

    async def query(self, msg):
        """
        prompt: list of dict, each dict contains "role" and "content"
        """
        
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=msg,
            # max_tokens=100,
            stop=['.\n\n'],
            # temperature=0.8,
            # frequency_penalty=0
        )
        
        if self.cal_approx_cost:
            for sentence in msg:
                self.num_prompt += len(sentence["content"])
            self.num_output += len(response.choices[0].message.content)
        
        return response
    
    async def async_query(self, prompts):
        """
        prompts: list of prompt
        """
        tasks = [self.query(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        return results
    
    def approx_cost(self):
        return self.num_prompt * 0.06 / 1000 + self.num_output * 0.12 / 1000
  
p1 = "\
You are a helpful assistant that tells me the next immediate task to do in Crafter game. \
Here are some tips: \
You have to worry about food, drink, and energy levels when they are low. \
Killing cows and eating plants will increase food level. Tree is not edible. \
Drinking water will increase drink level. \
Sleeping in a safe place (surrounded by blocks) will increase energy. \
Health level will decrease when attacked by monsters. \
Discovering new things when food, drink, and energy levels are high. \
Chop Trees to collect Wood. \
Use the wood to create a crafting table. \
Crafting pickaxe and sword near the crafting table. \
The pickaxe allows you to mine stone, while the sword is for attack. \
My ultimate goal is to discover as many diverse things as possible, accomplish as many diverse tasks as possible and become the best Crafter player in the world. \
Desired format: Reasoning: <reasoning>. Task: <task description>.\n\n\
Here is an example:\nReasoning: Your food level is low. You need to eat to restore your food level, and the cow is the only food source available. Task: Kill the cow.\n\n\
Reasoning: Your food level is low. You need to eat to restore your food level, and the cow is the only food source available. Task: Kill the cow.\n\n"
p2 = "\
Your health level is high, food level is high, drink level is high, energy is high. \
You see tree. \
You have in your inventory: wood, rock, plant. \
What do you do?"


p1 = "\
You are a helpful assistant that tells me whether the given task has been completed." 
# Desired format: Completion Criteria: <reasoning>. Answer: yes or no.\n\n\
# Here is an example:\n\
# Completion Criteria: The task's completion would be indicated by an increase in the drink property, as the objective involves consuming water to address thirst. Answer: no.\n\n\
# Completion Criteria: The task's completion would be indicated by an increase in the drink property, as the objective involves consuming water to address thirst. Answer: no.\n\n"
p2 = "\
The task at hand is to eat the plant to increase your food level. \
Initially, You see tree. You have in your inventory: plant,  Your health level is high, food level is high, drink level is high, energy is high. \
Afterwards, you move, attack cow, and chop tree. \
Currently, You see cow, lava, path, skeleton, stone, and tree. \
You have in your inventory: stone, wood_pickaxe, wood_sword,  \
Your health level is high, food level is high, drink level is high, energy is high. \
Has the task been completed? \
Desired format: Completion Criteria: <reasoning>; Answer: yes or no.\n\n"
 
    
if __name__ == "__main__":
    
    client = GPTClient(
        api_key = "EMPTY",
        api_base = "http://localhost:11017/v1",
        model = "meta-llama/Llama-2-70b-chat-hf",
    )
    
    
    prompts = [
        [
            {"role": "system", "content": p1},
            {"role": "user", "content": p2},
        ] 
    ]
    
    responses = asyncio.run(client.async_query(prompts))
    
    print(responses[0].choices[0].message.content)