import torch.nn as nn 
import torch
from transformers import AutoModel
import torch.nn.functional as F
import os
import pathlib
import pickle as pkl
import time
import numpy as np
import fasteners
from sentence_transformers import SentenceTransformer

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

class BertEncoder(nn.Module):
    
    def __init__(self, device="cuda"):
        super().__init__()
        
        self.model = SentenceTransformer(
            'sentence-transformers/paraphrase-MiniLM-L3-v2'
        )
        
        self.cache = {}
        
        self.device = device
        self.eval()
        self.to(self.device)
        
        self.encode(["Survive."])

    def forward(self, prompts, convert_to_numpy=True):
        
        # embeddings = []
        # eye_matrix = np.eye(22) if convert_to_numpy else torch.eye(22).to(self.device)
        
        # for prompt in prompts:
        #     if prompt == "Survive.":
        #         embeddings.append(eye_matrix[0]*0)
        #     else:
        #         embeddings.append(eye_matrix[available_actions.index(prompt)])
        
        # return embeddings
        
        # encode those are not in cache
        in_cache_idx = []
        not_in_cache_sentences = []
        for idx, prompt in enumerate(prompts):
            if prompt in self.cache:
                in_cache_idx.append(idx)
            else:
                not_in_cache_sentences.append(prompt)
        
        # encode those are not in cache
        n_embeddings = self.encode(not_in_cache_sentences, convert_to_numpy)
        
        # get embeddings from cache
        y_embeddings = []
        for idx in in_cache_idx:
            y_embeddings.append(self.cache[prompts[idx]])
        
        # simple case
        if len(in_cache_idx) == len(prompts):
            return y_embeddings
        
        # combine embeddings
        embeddings = []
        for idx in range(len(prompts)):
            if idx in in_cache_idx:
                next_embedding = y_embeddings[0]
                y_embeddings[:-1] = y_embeddings[1:]
                embeddings.append(next_embedding)
            else:
                next_embedding = n_embeddings[0]
                n_embeddings[:-1] = n_embeddings[1:]
                embeddings.append(next_embedding)
        
        return embeddings

    def encode(self, sentences, convert_to_numpy=True):
        self.eval()
        with torch.no_grad():
            if convert_to_numpy:
                embeddings = self.model.encode(
                    sentences, 
                    convert_to_numpy=True
                )
            else:
                embeddings = self.model.encode(
                    sentences, 
                    convert_to_numpy=False, 
                    device=self.device
                )
        
        for sentence, embedding in zip(sentences, embeddings):
            if sentence not in self.cache:
                self.cache[sentence] = embedding
        
        return embeddings
    

if __name__ == '__main__':
    
    encoder = BertEncoder(32, 'cpu')
    prompts = ["Mine stone.", "Eat a cow.", "Kill a cow."]
    embeddings = encoder(prompts)
    print("cosine similarity between 1st and 2nd prompt: ", torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0))
    print("cosine similarity between 1st and 3rd prompt: ", torch.nn.functional.cosine_similarity(embeddings[0], embeddings[2], dim=0))
    print("cosine similarity between 2nd and 3rd prompt: ", torch.nn.functional.cosine_similarity(embeddings[1], embeddings[2], dim=0))