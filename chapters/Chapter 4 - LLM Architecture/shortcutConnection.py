# we implement shortcut connection here.
# shortcut connection prevents vanishing gradient issue by skipping few layers of deep neural network 

import torch 
import torch.nn as nn 

# no of layers == layer_sizes.shape
class DeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, is_shortcut):
        super().__init__()
        self.is_shortcut = is_shortcut
        self.layers = nn.ModuleList([
                nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), nn.GELU()),
                nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), nn.GELU()),
                nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), nn.GELU()),
                nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), nn.GELU())
        ])

    def forward(self, inputs):
        for layer in self.layers:
            output = layer(inputs)
            # shortcut connection = element wise addition between addition of two layers (both must be of equal dim)
            if self.is_shortcut and output.shape == inputs.shape:
                inputs = inputs + output 
            else:
                inputs = output 
        return inputs 
    
    
def print_gradients(model, inputs):
    output = model(inputs)
    target = torch.tensor([[0., 0., 0.]])
    loss = nn.MSELoss()
    loss = loss(output, target)
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
             print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


layer_sizes = [3, 3, 3, 3, 3]
inputs = torch.tensor([[-1., 0., 1.]])
torch.manual_seed(123)
# set the shortcut initialization using Boolean value
model = DeepNeuralNetwork(layer_sizes, True)
print_gradients(model, inputs)


