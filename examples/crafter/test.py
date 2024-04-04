# Load model directly
import time 
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_utils import unwrap_model

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", padding_size="left")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

def query(prompt, past_key_values):

    input_pt = tokenizer(prompt, return_tensors="pt", padding=True)

    begin = time.time()
    output = model.generate(
        use_cache=True,
        max_new_tokens=10, 
        return_dict_in_generate=True, 
        input_ids=input_pt["input_ids"], 
        attention_mask=input_pt["attention_mask"],
        past_key_values=None,
    )
    print("Time taken: ", time.time() - begin)

    results = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
    
    past_key_values = output.past_key_values
    
    return results, past_key_values


prompt = [
    "You are a helpful agent."*10 + "Hi.",
    "You are a helpful agent."*10 + "Hi.",
]

result, past_key_values = query(prompt, past_key_values=None)

prompt = [
    result[0] + "How are you?",
    result[1] + "Who was Jim Henson?",
]

results, _ = query(prompt, past_key_values=past_key_values)


for result in results:
    print("="*60)
    print("Result: ", result)
