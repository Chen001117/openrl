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

class BertEncoder(nn.Module):
    
    def __init__(self, device="cuda"):
        super().__init__()
        
        self.model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
        # self.output_head = nn.Sequential(
        #     nn.Linear(384, embedding_dim) # for minilm
        # )
        
        self.cache = {}
        
        self.device = device
        self.eval()
        self.to(self.device)
        
        self.encode("Survive.")

    def forward(self, prompts):
        
        # encode those are not in cache
        in_cache_idx = []
        not_in_cache_sentences = []
        for idx, prompt in enumerate(prompts):
            if prompt in self.cache:
                in_cache_idx.append(idx)
            else:
                not_in_cache_sentences.append(prompt)
        
        # encode those are not in cache
        n_embeddings = self.encode(not_in_cache_sentences)
        
        # get embeddings from cache
        y_embeddings = []
        for idx in in_cache_idx:
            y_embeddings.append(self.cache[prompts[idx]])
        
        # combine embeddings
        embeddings = []
        for idx in range(len(prompts)):
            if idx in in_cache_idx:
                embeddings.append(y_embeddings.pop(0))
            else:
                embeddings.append(n_embeddings.pop(0))
        
        return embeddings

    def encode(self, sentence):
        
        self.eval()
        with torch.no_grad():
            embeddings = self.model.encode(sentence, convert_to_numpy=True)#, device=self.device)
            # embeddings = self.output_head(embeddings)
            
        # if sentence not in self.cache:
        #     self.cache[sentence] = embeddings
        
        return embeddings
    

if __name__ == '__main__':
    
    encoder = BertEncoder(32, 'cpu')
    prompts = ["Mine stone.", "Eat a cow.", "Kill a cow."]
    embeddings = encoder(prompts)
    print("cosine similarity between 1st and 2nd prompt: ", torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0))
    print("cosine similarity between 1st and 3rd prompt: ", torch.nn.functional.cosine_similarity(embeddings[0], embeddings[2], dim=0))
    print("cosine similarity between 2nd and 3rd prompt: ", torch.nn.functional.cosine_similarity(embeddings[1], embeddings[2], dim=0))