# This part focuses on embedding the tokens obtained from the tokenizer.
#* Embedding is nothing but the way of representing a token in the 3-D Tensor Matrix.

import torch 
import torch.nn as nn
from bytePair import verdict_sample

# Vocab size defines the number of rows in the embedding matrix. 
# The minimum number of rows must be (max token ID + 1).
# If your token data contains 5000 as the max token ID, then the rows must be at least 5001.

# Embedding dimensions represent the number of columns in the matrix.
# It's highly configurable, and higher dimensions typically improve accuracy but increase computation.

#* In our case, the matrix is 27076 x 50.
vocab_size = max(verdict_sample) + 1
embedding_dim = len(verdict_sample)    # Using the length of token data is not a feasible approach. Use standard embedding dimensions.

tensor_verdict_sample = torch.tensor(verdict_sample)
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

embedding_matrix = embedding_layer(tensor_verdict_sample)

# embedding_matrix[40] returns the embedding vector for the token ID 40.
print("Output Embedding:", embedding_matrix[40])




