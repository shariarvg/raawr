import numpy as np
import pandas as pd
import torch
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)#'meta-llama/Meta-Llama-3-8B-Instruct')
model = AutoModelForCausalLM.from_pretrained(model_name)#'meta-llama/Meta-Llama-3-8B-Instruct')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

def get_blacklist(word_id, vocab_size, alpha):
    """Generate a blacklist for a given word_id."""
    random.seed(word_id)
    blacklist_size = int(alpha * vocab_size)
    return random.sample(range(vocab_size), blacklist_size)

def generate_without_blacklist(prompt, max_tokens = 50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(device)
    generated_ids = input_ids.clone()
    for _ in range(max_tokens):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def generate_with_blacklist(prompt, alpha=0.1, max_tokens=50):
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(device)
    generated_ids = input_ids.clone()

    # Loop to generate tokens
    for _ in range(max_tokens):
        # Forward pass through GPT-2
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]  # Logits for the last token

        # Blacklist alpha% of the vocabulary
        vocab_size = logits.shape[-1]
        last_token = int(input_ids[0, -1])  # Get the last token's ID

        # Seed the random generator with the last token
        blacklist = get_blacklist(last_token, vocab_size, alpha)
        # Mask blacklisted logits
        mask = torch.full_like(logits, float("-inf"))
        mask[:, blacklist] = logits[:, blacklist]
        logits = logits.masked_fill(mask == float("-inf"), float("-inf"))

        # Sample the next token
        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)

        # Append the next token to the input_ids
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    # Decode the generated tokens
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def get_watermark_adherences(text, alpha=0.5):
    """Check if a given text was generated with the watermark."""
    # Tokenize the input text
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0].to(device)
    vocab_size = tokenizer.vocab_size
    
    adherences = 0  # Count how many word pairs do not match the watermark pattern
    total_pairs = len(input_ids) - 1
    
    for i in range(total_pairs):
        word_id = input_ids[i].item()
        next_word_id = input_ids[i + 1].item()
        
        # Generate blacklist for the current word
        blacklist = set(get_blacklist(word_id, vocab_size, alpha))
        
        # Check if the next word is in the blacklist
        if next_word_id not in blacklist:
            adherences += 1
    
    # Calculate the watermark adherence percentage
    return adherences, total_pairs

def z_statistic_watermark(text, alpha=0.5):
    """Check if a given text was generated with the watermark."""
    adherences, total_pairs = get_watermark_adherences(text, alpha)
    return 2*(adherences - total_pairs/2)/np.sqrt(total_pairs)
'''
# Example usage
prompt = "It was a dark and stormy night."
generations = []
watermarked = []
z_statistics = []


#always assume alpha = 0.5
for count in range(50):
    if random.randint(0,1) == 0:
        gen = generate_without_blacklist(prompt)
        watermarked.append(0)
    else:
        gen = generate_with_blacklist(prompt)
        watermarked.append(1)
    z = z_statistic_watermark(gen)
    generations.append(gen)
    z_statistics.append(z)
        
pd.DataFrame({"Gen": generations, "Watermarked": watermarked, "Z": z_statistics}).to_csv("watermark_test_viz.csv")
'''