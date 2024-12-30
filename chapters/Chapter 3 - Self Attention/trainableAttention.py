#* We introduce a new thing called 'weight matrices', to efficiently compute context vectors for corpus data.
#* Let's consider three matrices, w_k, w_q, and w_v (k- key, q - query, v - value)
# Query Matrix - encodes the query into tensor and enables us to determine the associativity of query with all other tokens
# Key Matrix - contains the encoding of all the other tokens and supports query matrix for associativity
# Value Matrix - contains the resultant vectors for the query.

import torch 

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],    # Your     (x^1)
    [0.55, 0.87, 0.66],     # journey  (x^2)
    [0.57, 0.85, 0.64],     # starts   (x^3)
    [0.22, 0.58, 0.33],     # with     (x^4)
    [0.77, 0.25, 0.10],     # one      (x^5)
    [0.05, 0.80, 0.55]]     # step     (x^6)
)

query = inputs[1]

# using query.shape would return a tensor value, not an integer value 
dimension_in = inputs.shape[1]
dimension_out = 2 
# print("Dimension in:", dimension_in)
# print("Dimension out:", dimension_out)

#* Let's perform attention score calculation using w_q, w_k, w_v for a single query.

# manual_seed ensures that same set of random numbers are generated for each run.
torch.manual_seed(123)

# requires_grad could be set to true in training phase.
w_q = torch.nn.Parameter(torch.rand(dimension_in, dimension_out), requires_grad=False)
w_k = torch.nn.Parameter(torch.rand(dimension_in, dimension_out), requires_grad=False)
w_v = torch.nn.Parameter(torch.rand(dimension_in, dimension_out), requires_grad=False)

query_vector = query @ w_q
key_vector = query @ w_k 
value_vector = query @ w_v 

print("Query Vector:", query_vector)
print("Value Vector:", value_vector)
print("Key Vector:", key_vector)

# now, let's compute the key and value vector for the single query we've chosen.
keys = inputs @ w_k 
values = inputs @ w_v 
single_key = keys[1]
single_attention_score = single_key.dot(query_vector)

# now, we got the attention score for the query input
print("Attention score for single query:", single_attention_score)




