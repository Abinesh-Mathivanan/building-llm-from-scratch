import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
BATCH_SIZE = 64
BLOCK_SIZE = 64  
EPOCHS = 50
LEARNING_RATE = 3e-4
EMBEDDING_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 6
DROPOUT = 0.1


enc = tiktoken.get_encoding("gpt2")

class TextDataset(Dataset):
    def __init__(self, file_path, block_size):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Encode the text using tiktoken
        self.tokens = enc.encode(text)
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config['num_heads']
        self.embedding_dim = config['embedding_dim']
        self.head_dim = self.embedding_dim // self.num_heads
        
        self.query = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.key = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.value = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        B, T, C = x.shape
        
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(self.head_dim))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        return out

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['embedding_dim'])
        self.ln2 = nn.LayerNorm(config['embedding_dim'])
        self.attn = MultiHeadAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config['embedding_dim'], 4 * config['embedding_dim']),
            nn.GELU(),
            nn.Linear(4 * config['embedding_dim'], config['embedding_dim']),
            nn.Dropout(config['dropout'])
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config['embedding_dim'])
        self.pos_embedding = nn.Parameter(torch.zeros(1, config['block_size'], config['embedding_dim']))
        self.dropout = nn.Dropout(config['dropout'])
        self.blocks = nn.ModuleList([Block(config) for _ in range(config['num_layers'])])
        self.ln_f = nn.LayerNorm(config['embedding_dim'])
        self.head = nn.Linear(config['embedding_dim'], vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        tok_emb = self.embedding(idx)
        pos_emb = self.pos_embedding[:, :T, :]
        x = self.dropout(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

def train_model():
    config = {
        'vocab_size': enc.n_vocab,
        'block_size': BLOCK_SIZE,
        'embedding_dim': EMBEDDING_DIM,
        'num_heads': NUM_HEADS,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
    }

    # Initialize dataset and dataloader
    dataset = TextDataset('./data/the-verdict.txt', BLOCK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, optimizer, and loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT(config['vocab_size'], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits.view(-1, config['vocab_size']), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}')

    return model, config

def generate_text(model, config, prompt, max_tokens=100):
    model.eval()
    device = next(model.parameters()).device
    
    context = torch.tensor(enc.encode(prompt), dtype=torch.long)[None, ...].to(device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            if context.size(1) > config['block_size']:
                context = context[:, -config['block_size']:]
                
            logits = model(context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1)
    
    generated_tokens = context[0].tolist()
    return enc.decode(generated_tokens)

if __name__ == "__main__":
    model, config = train_model()
    
    prompt = "The court"
    generated_text = generate_text(model, config, prompt)
    print("\nGenerated text:")
    print(generated_text)