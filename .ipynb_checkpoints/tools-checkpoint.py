import numpy as np
import pandas as pd
from torch import nn
from statsmodels.stats.proportion import proportions_ztest 
import torch

def matching(l1, l2, ind_l1):
    pointer_l1 = 0
    pointer_l2 = 0
    found = []
    iterations = 0
    while pointer_l1 < ind_l1 and iterations < 50:
        iterations += 1
        word1 = l1[pointer_l1].strip()
        word2 = l2[pointer_l2].strip()
        #print(word1)
        #print(word2)
        #print("---")
        if word1 == word2:
            pointer_l1 += 1
            pointer_l2 += 1
        elif word2 in word1:
            while word2 in word1:
                pointer_l2 += 1
                word2 = l2[pointer_l2]
            pointer_l1 += 1
        elif word1 in word2:
            while word1 in word2:
                pointer_l1 += 1
                word1 = l1[pointer_l1]
            pointer_l2 += 1
        else:
            found_option1 = matching(l1, l2[1:], ind_l1 - pointer_l1)
            found_option2 = matching(l1[1:], l2, ind_l1 - pointer_l1 - 1)
            found_option1 = [x + pointer_l1 + 2 for x in found_option1]
            found_option2 = [x + pointer_l1 + 2 for x in found_option2]
            return found_option1, found_option2
    word1 = l1[pointer_l1].strip()
    word2 = l2[pointer_l2].strip()
    while word2 in word1:
        found.append(pointer_l2)
        pointer_l2 += 1
        word2 = l2[pointer_l2]
    return min(found), max(found)

def p_statistic(num_violations, len_sequence, REMOVE_FACTOR):
    return proportions_ztest(count = num_violations, nobs = len_sequence - 1, value = 1.0/REMOVE_FACTOR)
    
def make_sparse_vec(lemmas, freqs, tokenizer):
    out = torch.zeros(len(tokenizer), dtype=float)
    encodings_and_freqs = [(tokenizer(lemma)['input_ids'], freqs[i]) for i, lemma in enumerate(lemmas)]
    encodings_and_freqs = [(encoding[0], freq) for (encoding, freq) in encodings_and_freqs if len(encoding) == 1]
    out[[e[0] for e in encodings_and_freqs]] = torch.tensor([e[1] for e in encodings_and_freqs], dtype=torch.double)
    return out/(torch.sum(out) + 0.001)
