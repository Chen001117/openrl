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
            temperature=0.6,
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
    
    
if __name__ == "__main__":
    
    client = GPTClient(
        api_key = "EMPTY",
        api_base = "http://localhost:11017/v1",
        model = "meta-llama/Llama-2-70b-chat-hf",
    )
    
    prompts = [
        [
            {"role": "system", "content": "You are a helpful agent."*60},
            {"role": "user", "content": "Tell me a joke."},
            {"role": "assistant", "content": "Sure, here's a joke for you: Why did the scarecrow win an award? Because he was outstanding in his field! (get it?)"},
            {"role": "user", "content": "Tell me another joke."},
        ] for _ in range(100)
    ]
    
    for i in range(100):
        responses = asyncio.run(client.async_query(prompts))
    
        print(i, "responses:", responses[0].choices[0].message.content)