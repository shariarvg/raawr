from torch import nn
import torch
import numpy as np
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from parsing_tools import matching, parse_coinco_with_attributes
from tools import make_sparse_vec
from coinco import CoinCo
from replacementprediction import ReplacementPrediction

from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Model, GPT2Tokenizer
model = GPT2Model.from_pretrained("gpt2", output_hidden_states = True)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# Example usage
xml_path = "coinco.xml"  # Replace with the actual path to your CoInCo XML file
coinco_data, coinco_data_points = parse_coinco_with_attributes(xml_path)
cdata = CoinCo(coinco_data_points, tokenizer)
dataloader = torch.utils.data.DataLoader(cdata, batch_size = 128)

loss_fn = nn.CrossEntropyLoss()
r = ReplacementPrediction(768, len(tokenizer)).to('cuda')
optimizer = torch.optim.Adam(r.parameters(), lr = 0.001)
#sd = torch.load("r_checkpoint_1_nosoftmax.pt")#, map_location = torch.device('cpu'))
#r.load_state_dict(sd)

model = model.to('cuda') ## for transformer embeddings

print("Continuining training for no softmax model")

for epoch in range(0, 1000):
    epoch_loss = 0.0
    for batch in dataloader:
        inputs, preserve, subs, og_word_encoding = batch
        inputs = inputs.to('cuda')
        preserve = preserve.to('cuda')
        subs = subs.to('cuda')
        og_word_encoding = og_word_encoding.to('cuda')
        
        with torch.no_grad():
            
            embeddings = model(**inputs).last_hidden_state
        
        optimizer.zero_grad()
        
        embeddings = (embeddings.squeeze() * preserve.unsqueeze(-1)).sum(dim = 1)

        prediction = r(embeddings, og_word_encoding)
        loss = loss_fn(prediction, subs)
        epoch_loss += loss.item()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(r.parameters(), max_norm=1.0)
        optimizer.step()
    print(f"loss over epoch {epoch}: {epoch_loss}")
        
torch.save(r.state_dict(), "r_checkpoint_1_nosoftmax_v4.pt")