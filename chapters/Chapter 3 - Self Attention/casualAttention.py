# now, we implement the casual attention mechanism
# casual attention is simply the technique of masking or hidings the tokens after the current one, such that the model will 
# only use the tokens before the current one for calculating context vector 

import torch 
import torch.nn as nn
import math

# we use torch,triu for casual attention, and torch.tril for backpropoagation

class Casual_Attention(nn.Module):
    def __init__(self, dimension_in, dimension_out, qkv_bias = False):
        super().__init__()
        self.dimension_out = dimension_out 
        self.w_query = nn.Linear(dimension_in, dimension_out, bias=qkv_bias)
        self.w_key = nn.Linear(dimension_in, dimension_out, bias=qkv_bias)
        self.w_value = nn.Linear(dimension_in, dimension_out, bias=qkv_bias)

    def forward(self, inputs):
        query = self.w_query(inputs)
        key = self.w_key(inputs)
        value = self.w_value(inputs)
        attention_scores = query @ key.transpose(-1, -2)
        scaled_attention_scores = attention_scores / math.sqrt(self.dimension_out)

        seq_len = scaled_attention_scores.size(-1)
        casual_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        # print(casual_mask)

        scaled_attention_scores.masked_fill_(casual_mask, float('-inf'))

        attention_weights = torch.softmax(scaled_attention_scores, dim=-1)
        # print(attention_weights)

        context_vectors = attention_weights @ value 
        return context_vectors 
    

                    # -------------------------------- Data Input -------------------------------- #    


inputs = torch.tensor(
    [[0.43, 0.15, 0.89],    # Your     (x^1)
    [0.55, 0.87, 0.66],     # journey  (x^2)
    [0.57, 0.85, 0.64],     # starts   (x^3)
    [0.22, 0.58, 0.33],     # with     (x^4)
    [0.77, 0.25, 0.10],     # one      (x^5)
    [0.05, 0.80, 0.55]]     # step     (x^6)
).unsqueeze(0) 

torch.manual_seed(123)
dimension_in = 3
dimension_out = 2

casual_attention = Casual_Attention(dimension_in, dimension_out)
context_vectors = casual_attention(inputs)

print("Context Vectors:", context_vectors)

