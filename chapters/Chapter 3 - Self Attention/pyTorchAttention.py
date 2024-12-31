                # -------------- Implementation of Self Attention using Class x -------------- #

import torch
import torch.nn as nn
import math 

class PyTorchAttention_v1(nn.Module):
    def __init__(self, dimension_in, dimension_out):
        super().__init__()
        self.dimension_out = dimension_out
        self.w_query = nn.Parameter(torch.rand(dimension_in, dimension_out), requires_grad=False)
        self.w_key = nn.Parameter(torch.rand(dimension_in, dimension_out), requires_grad=False)
        self.w_value = nn.Parameter(torch.rand(dimension_in, dimension_out), requires_grad=False)

    def forward(self, inputs):
        keys = inputs @ self.w_key 
        query = inputs @ self.w_query 
        value = inputs @ self.w_value 
        attention_scores = query @ keys.T 
        scaled_attention_scores = attention_scores / math.sqrt(self.dimension_out)
        attention_weights = torch.softmax(scaled_attention_scores, dim=-1)
        context_vectors = attention_weights @ value 
        return context_vectors


                    # ----------- Implementation of Self Attention using Linear Layers ----------- #


class Linear_PytorchTokenizer(nn.Module):
    def __init__(self, dimension_in, dimension_out, qkv_bias = False):
        super().__init__()
        self.dimension_out = dimension_out 
        self.w_query = nn.Linear(dimension_in, dimension_out, bias=qkv_bias)
        self.w_key = nn.Linear(dimension_in, dimension_out, bias=qkv_bias)
        self.w_value = nn.Linear(dimension_in, dimension_out, bias=qkv_bias)

    def forward(self, inputs):
        query = self.w_query(inputs)
        keys = self.w_key(inputs)
        value = self.w_value(inputs) 
        attention_scores = query @ keys.T 
        scaled_attention_scores = attention_scores / math.sqrt(dimension_out)
        attention_weights = torch.softmax(scaled_attention_scores, dim=-1)
        context_vectors = attention_weights @ value 
        return context_vectors



                    # -------------------------------- Data input -------------------------------- #


inputs = torch.tensor(
    [[0.43, 0.15, 0.89],    # Your     (x^1)
    [0.55, 0.87, 0.66],     # journey  (x^2)
    [0.57, 0.85, 0.64],     # starts   (x^3)
    [0.22, 0.58, 0.33],     # with     (x^4)
    [0.77, 0.25, 0.10],     # one      (x^5)
    [0.05, 0.80, 0.55]]     # step     (x^6)
)

torch.manual_seed(123)
dimension_in = 3 
dimension_out = 2
pytorch_attention = PyTorchAttention_v1(dimension_in, dimension_out)
linear_pytorch_attention = Linear_PytorchTokenizer(dimension_in, dimension_out)
print(pytorch_attention(inputs))
print(linear_pytorch_attention(inputs))


