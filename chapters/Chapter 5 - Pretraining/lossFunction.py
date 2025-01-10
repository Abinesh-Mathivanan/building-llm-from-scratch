import torch
import torch.nn.functional as F

def loss_fn(logits, targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
