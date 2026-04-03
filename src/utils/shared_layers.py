import torch
from torch import nn
import torch.nn.functional as F

class Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__();

        self.input_dim = input_dim;
        self.output_dim = output_dim;
    
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.layer_norm = nn.LayerNorm()
        self.activation = nn.ELU()
    
        self.linear_2 = nn.Linear()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.layer_norm(x)
        x = self.activation(x)

        x = self.linear_2(x)
        x = F.normalize(x, p=2, dim=1)
        return x