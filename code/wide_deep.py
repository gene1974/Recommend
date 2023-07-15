import torch
import torch.nn as nn

class wide_deep(nn.Module):
    def __init__(self, wide_in_dim, deep_in_dim, output_size):
        self.wide_layer = nn.Linear(wide_in_dim, wide_out)
        self.deep_embedding = nn.Embedding(deep_in_dim)
        self.deep_layer = nn.Sequencial([
            nn.Linear(deep_in, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, deep_out),
        ])

    def forward(self, wide_input, deep_input):
        wide_out = self.wide_layer(wide_input)
        deep_input = self.deep_embedding(deep_input)
        deep_out = self.deep_layer(deep_input)
        return wide_out, deep_out


