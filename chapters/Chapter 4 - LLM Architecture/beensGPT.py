# Creating a dummy GPT model named 'BeensGPT'

beensGPT_config = {
    "vocab_size": 50257,
    "context_length": 1024,
    "embedding_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "dropout_value": 0.1,
    "qkv_bias": False
}

import torch 
import torch.nn as nn 

class BeensGPTModel(nn.Module):
    def __init__(self, beens_config):
        super().__init__()
        self.token_embeddings = nn.Embedding(beens_config["vocab_size"], beens_config["embedding_dim"])
        self.positional_emeddings = nn.Embedding(beens_config["context_length"], beens_config["embedding_dim"])
        self.dropout = nn.Dropout(beens_config["dropout_value"])
        self.transformer_blocks = nn.Sequential(*[BeensTransformer(beens_config) for _ in range(beens_config["n_layers"])])
        self.normalized_layer = BeensLayerNorm(beens_config["embedding_dim"])
        self.output = nn.Linear(beens_config["embedding_dim"], beens_config["vocab_size"], bias=False)

    def forward(self, inputs):
        batch_size, sequence_length = inputs.shape
        token_embeds = self.token_embeddings(inputs)
        position_embeds = self.positional_emeddings(torch.arange(sequence_length, device=inputs.device))
        x = token_embeds + position_embeds 
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.normalized_layer(x)
        logits = self.normalized_layer(x)
        return logits 

class BeensTransformer(nn.Module):
    def __init__(self, beens_config):
        super().__init__()

    def forward(self, inputs):
        return inputs 
    
class BeensLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = 1e-5 
        self.scale = nn.Parameter(torch.ones(normalized_shape))
        self.shift = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, inputs):
        input_mean = inputs.mean(dim=1, keepdim=True)
        input_variance = inputs.var(dim=1, unbiased=True, keepdim=True)
        norm_input = (inputs - input_mean) / torch.sqrt(input_variance + self.eps)
        return self.scale * norm_input + self.shift 


# -------------------------------- Data Input -------------------------------- #

from tiktoken import get_encoding 
tokenizer = get_encoding("gpt2")
txt1 = "how are you doing today"
txt2 = "how are we gonna do"
batch = []
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
# print(batch)

torch.manual_seed(123)
model = BeensGPTModel(beensGPT_config)
logits = model(batch)
# print(logits.shape)
# print(logits)






