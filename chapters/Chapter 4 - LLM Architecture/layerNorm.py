# This file demonstrates the concept of Layer Normalization.
# Layer Normalization is a critical technique that helps stabilize training by normalizing activations in deep networks.
# We set up a simple tensor input to test LayerNorm-related methods.

import torch 
import torch.nn as nn

torch.manual_seed(123)
inputs = torch.rand(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
batch_example = layer(inputs)

mean_val = batch_example.mean(dim=1, keepdim=True)
var_val = batch_example.var(dim=1, keepdim=True)
print("Mean value (Regular Batch):", mean_val)
print("Variance value (Regular Batch):", var_val)

# we manually normalize the batch using standard normalization.
norm_batch = (batch_example - mean_val) / torch.sqrt(var_val)
norm_mean = norm_batch.mean(dim=1, keepdim=True)
norm_var = norm_batch.var(dim=1, keepdim=True)

# To round off the values for better readability
torch.set_printoptions(sci_mode=False)
print("Mean value (Normalized Batch):", norm_mean)
print("Variance value (Normalized Batch):", norm_var)
