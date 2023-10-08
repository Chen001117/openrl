import transformers
import torch
        
config = transformers.GPT2Config(
                n_embd=128,
                n_positions=1024,
                n_layer=3,
                n_head=1,
                n_inner=128,
                activation_function="tanh",
                resid_pdrop=0.,
                attn_pdrop=0.,
                embd_pdrop=0.,
)
transformer = transformers.GPT2Model(config)

actor_features = torch.zeros([3,10,128])
actor_features[:,-3:] += torch.normal(mean=torch.zeros([3,3,128]))
attention_mask = torch.zeros([3,10])
attention_mask[:,-3:] += 1.
env_step = torch.ones([3,10]).long()
env_step[:,-3] = 1
env_step[:,-2] = 2
env_step[:,-1] = 3

x = transformer(
    inputs_embeds=actor_features,
    attention_mask=attention_mask,
    position_ids=env_step,
)
x = x['last_hidden_state']
x = x[:,-1]
print(x[0,:3])

actor_features[:,:3] += torch.normal(mean=torch.zeros([3,3,128]))
env_step[:,-4] += 1
x = transformer(
    inputs_embeds=actor_features,
    attention_mask=attention_mask,
    position_ids=env_step,
)
x = x['last_hidden_state']
x = x[:,-1]
print(x[0,:3])