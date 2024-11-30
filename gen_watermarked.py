import os
from openai import OpenAI
import tiktoken
import random
import pandas as pd
import sys
from tools import p_statistic
tokenizer = tiktoken.encoding_for_model("gpt-4")

client = OpenAI(
    api_key=APIKEYHERE
)

#REMOVE_FACTOR = float(sys.argv[1])
##TEMP = float(sys.argv[2])
#BLACK = int(sys.argv[3])
#LEN_SEQ = int(sys.argv[4])

def get_seed(sentence):
    return tokenizer.encode(sentence)[-1]

def gen_next(sentence):
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": sentence,
        }
    ],
    model="gpt-4",
    max_tokens = 1,
    temperature = 1.5
)
    return chat_completion.choices[0].message.content

def get_blacklist(seed, rf):
    
    random.seed(seed)
    blacklist = random.sample(range(0, 100000), int(100000/rf))
    
    return blacklist

def violations(seq, rf):
    out = 0
    for i in range(len(seq) - 1):
        encoding = seq[i]
        blacklist = set(get_blacklist(encoding, rf))
        out += int(seq[i+1] in blacklist)
                        
    return out

def gen_with_blacklist(sentence, rf, black=1, iterations = 20):
    token_ids = tokenizer.encode(sentence)
    generated_tokens = []
    last_token = token_ids[-1]
    
    for count in range(iterations):
        blacklist = get_blacklist(last_token, rf) if black == 1 else set()
        next_token = gen_next(sentence)
        fail_count = 1
        while tokenizer.encode(next_token)[0] in blacklist and fail_count < 5:
            next_token = gen_next(sentence)
            fail_count += 1
        
        sentence +=  " " + next_token
        last_token = tokenizer.encode(next_token)[0]
        generated_tokens.append(last_token)
        token_ids = tokenizer.encode(sentence)
    vios = violations(generated_tokens, rf)
    return sentence, generated_tokens, vios, p_statistic(vios, iterations - 1, 1.0/rf)
'''
starter = "The quick brown fox jumped over the lazy dog"
df = pd.DataFrame()
all_sentences = []
all_vios = []
s, g, vios = gen_with_blacklist(starter, LEN_SEQ)
print("Sentence: ",s)
print("G: ",g)
print("Number of violations: ",vios)
print("P value: ", p_statistic(vios, LEN_SEQ, REMOVE_FACTOR)[1])
'''