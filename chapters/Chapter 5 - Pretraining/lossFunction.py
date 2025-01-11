import torch
import torch.nn.functional as F

def loss_fn(logits, targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

def perplexity(loss):
    return torch.exp(loss)

def accuracy(logits, targets):
    predictions = torch.argmax(logits, dim=-1)
    return (predictions == targets).float().mean()