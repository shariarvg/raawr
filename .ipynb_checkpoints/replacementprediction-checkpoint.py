from torch import nn

class ReplacementPrediction(nn.Module):
    def __init__(self, dim_embedding, len_vocab):
        super(ReplacementPrediction, self).__init__()
        self.linear = nn.Linear(dim_embedding, len_vocab)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, embedding, one_hot_encodings):#, encoding_of_og_word):
        # Linear transformation
        logits = one_hot_encodings * self.linear(embedding)  # Shape: (batch_size, len_vocab)

        # Apply softmax
        probabilities = self.softmax(logits)
        return probabilities