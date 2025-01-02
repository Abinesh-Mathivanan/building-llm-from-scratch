# now we implement the multi-head attention where the data is split acorss multiple heads to be processed independently
# we set the dropout to 0.1, essentially 10% of the data

import torch 
import torch.nn as nn 
import math

class Multi_head_Attetion(nn.Module):
    def __init__(self, dimension_in, num_heads, dropout = 0.1, qkv_bias = False):
        super().__init__()
        assert dimension_in % num_heads == 0, "dim must be divisible"
        self.num_heads = num_heads
        self.head_dim = dimension_in // num_heads
        self.w_query = nn.Linear(dimension_in, dimension_in, bias=qkv_bias)
        self.w_key = nn.Linear(dimension_in, dimension_in, bias=qkv_bias)
        self.w_value = nn.Linear(dimension_in, dimension_in, bias=qkv_bias)
        self.output_value = nn.Linear(dimension_in, dimension_in)
        self.dropout_value = nn.Dropout(dropout)

    def forward(self, inputs):
        batch_size, sample_len, dimension_in = inputs.size()

        query = self.w_query(inputs).view(batch_size, sample_len, self.num_heads, self.head_dim)
        key = self.w_key(inputs).view(batch_size, sample_len, self.num_heads, self.head_dim)
        value = self.w_value(inputs).view(batch_size, sample_len, self.num_heads, self.head_dim)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout_value(attention_weights)
        
        context = torch.matmul(attention_weights, value).view(batch_size, sample_len, dimension_in)
        # the output layer is used to concat the data in the given output dimensions
        output = self.output_value(context)

        return output
    

                    # -------------------------------- Data Input -------------------------------- #


dim_in = 128
num_heads = 8
inputs = torch.rand(32, 10, dim_in)  
multi_head = Multi_head_Attetion(dim_in, num_heads)
outputs = multi_head(inputs)
print("Sample Attention value:", outputs[1])

