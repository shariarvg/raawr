'''
Save embeddings for analyses, so I don't have to recompute them everytime
'''

import numpy as np
import torch
from parsing_tools import matching, parse_coinco_with_attributes_and_make_substitutions

device = 'cuda'

from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Model, GPT2Tokenizer
model = GPT2Model.from_pretrained("gpt2", output_hidden_states = True).to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

def get_substitution(cdp, i):
    if len(cdp['sub_versions']) >= i:
        return cdp['sub_versions'][i]
    return ""

def get_preserve(cdp, tokenizer):
    '''
    This vector, or stack of vectors, chooses the embeddings for the specific token index or sequence
    of indices, so that we have an (D,) vector instead of an LxD matrix
    '''
    l = cdp['list_tokens']
    sentence = " ".join(l)
    inputs = tokenizer(sentence, truncation = True, return_tensors="pt", add_special_tokens=True, padding = 'max_length', max_length = 100)
    token_number = cdp['token_number']
    l2 = [tokenizer.decode([a]) for a in inputs['input_ids'][0]]
    ind = matching(l, l2, token_number)
    out = torch.zeros(100)
    out[ind[0]] = 1
    out = out/torch.sum(out)
    return out
        

def cache(coinco_data_points, embedding_model, embedding_tokenizer, model_dim, max_replace, batch_size = 16):
    preserves = torch.stack([get_preserve(cdp, embedding_tokenizer) for cdp in coinco_data_points]).to(device)
    
    all_og_sentences = [c['sentence'] for c in coinco_data_points]
    all_subs_1 = [c['sub_version'] for c in coinco_data_points]
    
    all_embeddings_og = torch.zeros((len(coinco_data_points), model_dim)).to(device)
    all_embeddings_sub1 = torch.zeros((len(coinco_data_points), model_dim)).to(device)
    
    for i in range(0, len(coinco_data_points), batch_size):
        batch_og = all_og_sentences[i:i+batch_size]
        batch_sub1 = all_subs_1[i:i+batch_size]
        #batch_sub2 = all_subs_2[i:i+batch_size]
        
        #inputs_og = tokenizer(batch_og, truncation = True, return_tensors="pt", padding = 'max_length', max_length = 100).to(device)
        inputs_sub1 = tokenizer(batch_sub1, truncation = True, return_tensors="pt", padding= 'max_length', max_length = 100).to(device)
        #inputs_sub2 = tokenizer(batch_sub2, padding=True, truncation=True, return_tensors="pt").to(device)
        
        preserve = preserves[i:i+batch_size]
        
        
        with torch.no_grad():
            #embeddings_og = (model(**inputs_og).last_hidden_state*preserve.unsqueeze(-1)).sum(dim=1)
            embeddings_sub1 = (model(**inputs_sub1).last_hidden_state*preserve.unsqueeze(-1)).sum(dim=1)
            #embeddings_sub2 = (model(**inputs_sub2).last_hidden_state*preserve.unsqueeze(-1)).sum(dim=1)

        #all_embeddings_og[i:i+batch_size] = embeddings_og
        all_embeddings_sub1[i:i+batch_size] = embeddings_sub1
        
    #torch.save(all_embeddings_og, 'all_embeddings_og.pt')
    torch.save(all_embeddings_sub1, 'all_embeddings_replacement.pt')
    
_ , coinco_data_points = parse_coinco_with_attributes_and_make_substitutions('coinco.xml')
cache(coinco_data_points, model, tokenizer, 768, 2, 16)