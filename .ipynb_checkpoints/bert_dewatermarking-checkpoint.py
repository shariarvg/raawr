'''
Methods for dewatermarkering text
'''

import pandas as pd
import numpy as np
import torch
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import BertForMaskedLM, BertTokenizer
from transformers import RobertaForMaskedLM, RobertaTokenizer, RobertaConfig
from generation_and_recovery import get_watermark_adherences, z_statistic_watermark
from gen_watermarked import violations
import torch.nn.functional as F
import tiktoken

from parsing_tools import parse_coinco_with_attributes, matching
from replacementprediction import ReplacementPrediction

model_name = "roberta-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = RobertaConfig.from_pretrained("roberta-base")
config.is_decoder = False  # Ensure bidirectional self-attention for MLM

#semantic_tokenizer = RobertaTokenizer.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForMaskedLM.from_pretrained(model_name, config = config).to(device)
semantic_model = AutoModel.from_pretrained(model_name).to(device)
gpt2_model = GPT2Model.from_pretrained("gpt2", output_hidden_states = True)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
gpt2_model.resize_token_embeddings(len(tokenizer))

gpt4_tokenizer = tiktoken.encoding_for_model("gpt-4")



def perform_inference(model, tokenizer, sentence, r, word = None, word_index = None):
    '''
    Sentence modification for Algo 2
    '''
    #sentence = sentence.replace("-", "")
    #sentence_split = sentence.split(" ")
    #if word is None:
    #    word = sentence_split[word_index]
    #elif word_index is None:
    #    word_index = sentence_split.index(word)
    #print(word)
    tokens = tokenize_text(sentence)
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True, padding = 'max_length', max_length = 100)
    #l2 = [tokenizer.decode(a).strip() for a in inputs['input_ids'][0]]
    
    #inds = list(matching(sentence_split, l2, word_index))
    #print(l2[inds[0]])
    preserve = torch.zeros(100)
    preserve[word_index] = 1
    preserve = preserve/(torch.sum(preserve)+0.001)
    
    one_hot_vector_og = torch.zeros(tokenizer.vocab_size+1)
    #print(inputs['input_ids'][0][word_index])
    one_hot_vector_og[int(inputs['input_ids'][0][word_index])] = 1
    one_cold_vector_og = torch.ones_like(one_hot_vector_og) - one_hot_vector_og
    
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state
    embeddings = embeddings.squeeze().T @ preserve#(embeddings.squeeze() * preserve.unsqueeze(dim=1))#.sum(dim = 0)
    probs = r(embeddings, one_cold_vector_og)
    sample = torch.multinomial(probs, num_samples = 1)
    #print(tokenizer.decode(sample))
    
    replacement = tokenizer.convert_ids_to_tokens([sample])[0]
    tokens[word_index] = replacement
    #inputs[word_index] = tokenizer.decode(sample)
    return tokenizer.convert_tokens_to_string(tokens)

def compute_sentence_similarity(original_text, modified_text, cosine = True):
    '''
    Compute cosine similarity between two sentences using RoBERTa embeddings.
    '''
    with torch.no_grad():
        original_tokens = semantic_tokenizer(original_text, return_tensors="pt", padding=True, truncation=True).to(device)
        modified_tokens = semantic_tokenizer(modified_text, return_tensors="pt", padding=True, truncation=True).to(device)
        
        original_embedding = semantic_model(**original_tokens).last_hidden_state.mean(dim=1)
        modified_embedding = semantic_model(**modified_tokens).last_hidden_state.mean(dim=1)
        if cosine == True:
            similarity = F.cosine_similarity(original_embedding, modified_embedding)
    return similarity.item()

def compute_distance(original_text, modified_text):
    with torch.no_grad():
        original_tokens = gpt2_tokenizer(original_text, return_tensors="pt", padding=True, truncation=True).to(device)
        modified_tokens = gpt2_tokenizer(modified_text, return_tensors="pt", padding=True, truncation=True).to(device)
        
        original_embedding = gpt2_model(**original_tokens).last_hidden_state.mean(dim=1)
        modified_embedding = gpt2_model(**modified_tokens).last_hidden_state.mean(dim=1)
        return torch.sqrt(((original_embedding - modified_embedding)*(original_embedding - modified_embedding)).sum())

def tokenize_text(text):
    """Tokenize the input text."""
    return tokenizer.tokenize(text)

def replace_masked_word(tokens, word_index, predicted_token):
    """Replace the masked word with the predicted token."""
    tokens[word_index] = predicted_token.capitalize()
    return tokenizer.convert_tokens_to_string(tokens)

def replace_word_with_bert(text, word_index):
    """Main function to replace a random word in the text using BERT."""
    tokens = tokenize_text(text)
    masked_tokens, word_index = mask_word(tokens.copy(), word_index)
    masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
    
    predicted_token = predict_best_replacement(masked_text, tokens[word_index], text)
    replaced_text = replace_masked_word(tokens, word_index, predicted_token)
    
    return replaced_text

def tokenize_by_words(text):
    """Tokenize text into words."""
    return text.split()  # Simple word-level tokenization


def replacement_score(sentence_og, token_index, replacement_id, sim = True):
    '''
    Get the replacement score of replacing token_index with the token id of replacement_id
    If sim = True, return the cosine similarity. (higher the better)
    Otherwise, return the L2 distance. (lower the better)
    '''
    replacement_token = tokenizer.convert_ids_to_tokens([replacement_id])[0]
    #sentence to words
    tokens = tokenize_text(sentence_og)
    
    #copy words
    tokens2 = tokens.copy()
    #add replacement
    tokens2[token_index] = replacement_token 
    #back to sentence
    sentence2 = tokenizer.convert_tokens_to_string(tokens2)
    
    #similarity between sentences
    if sim:
        sentence_similarity = compute_sentence_similarity(sentence_og, sentence2)
        if sentence_similarity == 1:
            return 0
        return sentence_similarity
    else:
        return compute_distance(sentence_og, sentence2)

def predict_word_replacement(masked_text, unmasked_text, original_word, k = 10):
    """Predict the best replacement for the masked word."""
    input_ids = tokenizer.encode(masked_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    mask_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    mask_logits = logits[0, mask_index, :].squeeze()
    original_word_id = tokenizer.convert_tokens_to_ids(original_word)
    mask_logits[original_word_id] = float('-inf')  # Exclude original word

    # Get top replacement
    top_token_indices = torch.topk(mask_logits, k = 10).indices
    
    token_ids_and_scores = {token_id: replacement_score(unmasked_text, mask_index, token_id) for token_id in top_token_indices}
    max_score = max(token_ids_and_scores.values())
    return tokenizer.convert_ids_to_tokens([max(token_ids_and_scores, key = token_ids_and_scores.get)])[0], max_score

def best_candidates(masked_text, unmasked_text, original_word, k = 10):
    '''
    Should be very similar to method above. just gives the best candidates, instead of the best candidate.
    '''
    input_ids = tokenizer.encode(masked_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    mask_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    mask_logits = logits[0, mask_index, :].squeeze()
    original_word_id = tokenizer.convert_tokens_to_ids(original_word)
    mask_logits[original_word_id] = float('-inf')  # Exclude original word
    
    top_token_indices = torch.topk(mask_logits, k = 10).indices
    
    token_ids_and_dists = {tokenizer.decode(token_id)[0]: replacement_score(unmasked_text, mask_index, token_id, sim = False) for token_id in top_token_indices}
    
    return token_ids_and_dists

def generate_til_threshold(masked_text, unmasked_text, original_word, threshold_dist, max_k = 15):
    '''
    Generate top k, check the distances of the sentence modification to the original setence embedding.
    Once we find a replacement with distance less than threshold_dist, return that
    '''
    input_ids = tokenizer.encode(masked_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    mask_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    mask_logits = logits[0, mask_index, :].squeeze()
    original_word_id = tokenizer.convert_tokens_to_ids(original_word)
    mask_logits[original_word_id] = float('-inf')  # Exclude original word
    
    top_token_indices = torch.topk(mask_logits, k = 10).indices
    
    min_dist = float('inf')
    min_dist_id = -1
    start = 0
    for token_id in top_token_indices:
        dist = replacement_score(unmasked_text, mask_index, token_id, sim = False)
        if dist < threshold_dist:
            return tokenizer.convert_ids_to_tokens([token_id])[0], dist
        elif dist < min_dist:
            min_dist = dist
            min_dist_id = token_id
    return tokenizer.convert_ids_to_tokens([min_dist_id])[0], min_dist
        

def mask_tokens(tokens, token_index):
    """Mask a random word in the list of words."""
    tokens[token_index] = tokenizer.mask_token
    return tokens, token_index

def add_mask_to_sentence(sentence, token_index):
    tokens = tokenize_text(sentence)
    masked_tokens, word_index = mask_tokens(tokens.copy(), token_index)
    masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
    
    return masked_text, tokens[token_index]

def replacement_at_index(sentence, index, method = generate_til_threshold):
    '''
    Replace a random word in the sentence with a model-predicted word.
    If method is generate_til_threshold, then generate until you've hit the threshold (Approach 3)
    Otherwise, if the method is predict_word_replacement, execute Approach 1
    '''
    tokens = tokenize_text(sentence)
    
    masked_sentence, token_under_mask = add_mask_to_sentence(sentence, index)

    replacement_token, replacement_score = method(masked_sentence, sentence, token_under_mask, 2, 15)

    tokens[index] = replacement_token.capitalize() if token_under_mask.istitle() else replacement_token
    return tokenizer.convert_tokens_to_string(tokens), replacement_score

def replacement_coinco(cdp):
    '''
    Specific to an element of the coinco_data_points list
    '''
    l = cdp['list_tokens']
    sentence = " ".join(l).replace("-","")
    
    inputs = gpt2_tokenizer(sentence, truncation = True, return_tensors="pt", add_special_tokens=True, padding = 'max_length', max_length = 100)
    token_number = cdp['token_number']
    l2 = [gpt2_tokenizer.decode([a]) for a in inputs['input_ids'][0]]
    index = matching(l, l2, token_number)[0]
    
    good_replacement_dists = []
    bad_replacement_dists = []
    
    masked_sentence, token_under_mask = add_mask_to_sentence(sentence, index)
    
    tokens_and_dists = best_candidates(masked_sentence, sentence, token_under_mask)
    
    good_replacement_dists = [item for key, item in tokens_and_dists.items() if key in cdp['subs']]
    bad_replacement_dists = [item for key, item in tokens_and_dists.items() if key not in cdp['subs']]
    return good_replacement_dists, bad_replacement_dists

def iterated_replacements(sentence, iterations=20):
    '''
    Iteratively make replacements
    '''
    unreplaced = list(range(len(sentence.split(" "))))
    for count in range(iterations):
        ind = random.sample(unreplaced, 1)[0]
        unreplaced.remove(ind)
        sentence, _ = replacement_at_index(sentence, ind)
        print(f"Iteration {count+1} done")
    return sentence

def iterated_replacements_r(sentence, iterations = 20):
    '''
    Iteratively make replacements with Approach 2
    '''
    r = ReplacementPrediction(768, len(gpt2_tokenizer))
    sd = torch.load("r_checkpoint_1_nosoftmax_v4.pt", map_location = torch.device('cpu'))
    r.load_state_dict(sd)
    
    unreplaced = list(range(len(sentence.split(" "))))
    for count in range(iterations):
        ind = random.sample(unreplaced, 1)[0]
        print(ind)
        unreplaced.remove(ind)
        sentence = perform_inference(gpt2_model, gpt2_tokenizer, sentence, r = r, word_index = ind)
        print(f"Iteration {count+1} done")
    return sentence

def tokenize_replacement(sentence, rf = None):
    all_tokens = []
    for word in sentence.split(" "):
        tokens = gpt4_tokenizer.encode(word)
        all_tokens.extend(tokens)
    if rf is None:
        return all_tokens
    return all_tokens, violations(all_tokens, rf)
#'''
s = 'It was a dark and stormy night but that didn\'t stop me from standing," New Guard chief Beverbert told SBS News in Paris the past few days during another encounter there at 2pm that ended five miles away and'

print(iterated_replacements_r(s, iterations = 10))
'''

df = pd.read_csv("watermark_test_viz.csv")
df['Regen'] = df['Gen'].apply(lambda x: iterated_replacements(x))
df['Z_Regen'] = df['Regen'].apply(lambda x: z_statistic_watermark(x))
df.to_csv("dewatermark_test_viz.csv")
'''