import torch

import torch.nn as nn
import torch.nn.functional as F

with open('./data/the-verdict.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

block_size = 8  
X, y = [], []
for i in range(len(text) - block_size):
    context = text[i:i+block_size]
    target = text[i+block_size]
    X.append(encode(context))
    y.append(stoi[target])

X = torch.tensor(X)
y = torch.tensor(y)

# Simple model
class TextGenerator(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.linear = nn.Linear(n_embd * block_size, vocab_size)

    def forward(self, idx):
        x = self.embedding(idx)
        x = x.view(x.shape[0], -1)
        logits = self.linear(x)
        return logits

model = TextGenerator(vocab_size=vocab_size, n_embd=32, block_size=block_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch_size = 32
for steps in range(10000):
    ix = torch.randint(0, len(X), (batch_size,))
    Xbatch, ybatch = X[ix], y[ix]
    
    logits = model(Xbatch)
    loss = F.cross_entropy(logits, ybatch)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if steps % 1000 == 0:
        print(f'step {steps}: loss {loss.item():.4f}')

def generate(model, start_text, max_tokens=100):
    model.eval()
    context = torch.tensor(encode(start_text))
    generated = start_text
    
    for _ in range(max_tokens):
        if len(context) > block_size:
            context = context[-block_size:]
        
        logits = model(context.unsqueeze(0))
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs[0], 1).item()
        
        generated += itos[next_token]
        context = torch.cat([context, torch.tensor([next_token])])
    
    return generated

print(generate(model, "The jury", 200))