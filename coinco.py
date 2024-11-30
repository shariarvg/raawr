from torch import nn
import torch
import numpy as np
from parsing_tools import matching
from tools import make_sparse_vec

class CoinCo(nn.Module):
    def __init__(self, coinco_data_points, tokenizer):
        super(CoinCo, self).__init__()
        self.data = coinco_data_points
        self.tokenizer = tokenizer
    def __getitem__(self, i):
        dic = self.data[i]
        subs_lemmas = [a['lemma'] for a in dic['subs']]
        subs_freqs = [a['freq'] for a in dic['subs']]
        l = dic['list_tokens']
        og_word = l[dic['token_number']]
        sentence = " ".join(l).replace("-","")
        inputs = self.tokenizer(sentence, return_tensors="pt", add_special_tokens=True, padding = 'max_length', max_length = 100)
        l1 = l
        l2 = [self.tokenizer.decode([a]) for a in inputs['input_ids'][0]]
        start_match, end_match = matching(l1, l2, dic['token_number'])
        preserve = torch.zeros(100)
        preserve[start_match:end_match+1] = 1
        preserve = preserve/(torch.sum(preserve)+0.001)
        one_hot_vector_og = torch.zeros(self.tokenizer.vocab_size+1)
        one_hot_vector_og[self.tokenizer(og_word)['input_ids'][0]] = 1
        return inputs, preserve, make_sparse_vec(subs_lemmas, subs_freqs, self.tokenizer), torch.ones_like(one_hot_vector_og) - one_hot_vector_og
    def __len__(self):
        return len(self.data)
    
class CoinCO2(nn.Module):
    def __init__(self, coinco_data_points, cached_embeddings, tokenizer):
        super(CoinCo, self).__init__()
        self.data = coinco_data_points
        self.tokenizer = tokenizer
        
    def __getitem__(self, i):
        dic = self.data[i]
        