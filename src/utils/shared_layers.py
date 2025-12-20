import torch
from torch import nn

class Projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__();

        self.input_dim = input_dim;
        self.output_dim = output_dim;
    
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=output_dim)

        self.block = nn.Sequential(
            nn.ELU(),
            nn.Linear(output_dim, output_dim),
            nn.Dropout(p=0.5)
        )
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.linear_1(x);
        res = self.block(x);
        x = x + res;
        x = self.layer_norm(x)
        return x