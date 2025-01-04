import torch
import torch.nn as nn 
import matplotlib.pyplot as plt 

# we implement the GELU (Gaussian error linear unit) here
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
         # GELU formula: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.44715 * x^3)))
        return 0.5 * inputs * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi) ) * (inputs + 0.44715 * torch.pow(inputs, 3))))
    
# we generate 100 tensor points from -3 to 3
gelu = GELU()
relu = nn.ReLU()
inputs = torch.linspace(-3, 3, 100)
gelu_output, relu_output = gelu(inputs), relu(inputs)

# for i, (y, label) in enumerate(zip([gelu_output, relu_output], ["GELU", "ReLU"]), 1):
#     plt.subplot(1, 2, i)
#     plt.plot(inputs, y)
#     plt.title(f"{label} activation function")
#     plt.xlabel("x")
#     plt.ylabel(f"{label}(x)")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


# --------------------------------- Some info -------------------------------- #

# why do we use GELU, not anything like RELU or SWIGLU?
# GELU is preferred over ReLU for its smooth, probabilistic nature, improving gradient flow and preventing dead neurons.
# It outperforms ReLU in tasks like NLP by enhancing generalization and ensuring stable optimization.
# GELU is often more computationally efficient than SwiGLU while offering better performance in transformer-based architectures.

