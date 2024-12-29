# The Attention Mechanism is at the heart of LLM.
#* Attention determines the importance of each word relative to the others.
#* Attention scores are computed by taking the dot product of the input token with all other tokens in the corpus.

import torch 
input_embedding = torch.tensor(
    [[0.43, 0.15, 0.89],    # Your     (x^1)
    [0.55, 0.87, 0.66],     # journey  (x^2)
    [0.57, 0.85, 0.64],     # starts   (x^3)
    [0.22, 0.58, 0.33],     # with     (x^4)
    [0.77, 0.25, 0.10],     # one      (x^5)
    [0.05, 0.80, 0.55]]     # step     (x^6)
)

query = input_embedding[1]
attention_scores = torch.empty(input_embedding.shape[0])
for i, x_i in enumerate(input_embedding):
    attention_scores[i] = torch.dot(x_i, query)

#* A higher attention score indicates a higher likelihood that the token will appear after the given token.
print("The Attention scores:", attention_scores)

#* After computing the attention scores, it's time to normalize.
normalized_attention_scores = attention_scores / attention_scores.sum()
print("The Normalized attention score:", normalized_attention_scores)
print("Sum of Normalized scores:", normalized_attention_scores.sum())

#* The below is the implementation of context vector computation for input vector 2.
context_vectors = torch.zeros(query.shape)
for i, x_i in enumerate(input_embedding):
    context_vectors += normalized_attention_scores[i] * x_i
print("The context vectors:", context_vectors)


# TODO: Let's implement the simplified self attention for all the input vectors.

full_attention_scores = input_embedding @ input_embedding.T 
full_attention_weights = torch.softmax(full_attention_scores, dim=-1)
full_context_vector = full_attention_weights @ input_embedding 

print("Final Context Embeddings of the input:", full_context_vector)

#* To implement the context vector embeddings, simply follow the above three steps



