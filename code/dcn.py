import torch
import torch.nn as nn
import torch.nn.functional as F


'''
if self.parameterization == 'vector':
    xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
    dot_ = torch.matmul(x_0, xl_w)
    x_l = dot_ + self.bias[i] + x_l
elif self.parameterization == 'matrix':
    xl_w = torch.matmul(self.kernels[i], x_l)  # W * xi  (bs, in_features, 1)
    dot_ = xl_w + self.bias[i]  # W * xi + b
    x_l = x_0 * dot_ + x_l  # x0 · (W * xi + b) +xl  Hadamard-product
'''

# DCN-v(vector)
class DCN(nn.Module):
    def __init__(self, in_feature, embedding_dim, num_cross_layer, deep_layers = [64, 32, 1]):
        super().__init__()
        self.embedding = nn.Embedding(in_feature, embedding_dim)

        deep_layer_list = [nn.Linear(embedding_dim, deep_layers[0])]
        for i in range(1, len(deep_layers)):
            deep_layer_list.append(nn.ReLU())
            deep_layer_list.append(nn.Linear(deep_layers[i - 1], deep_layers[i]))

        # cross_l(x) = xixjwij + b_l + x, wij is vector
        self.cross_layers = nn.ModuleList()
        for i in range(num_cross_layer):
            self.cross_layers.append(nn.Linear(embedding_dim, embedding_dim))

        self.out_layer = nn.Linear(embedding_dim + sum(deep_layers), 1)

    def forward(self, x):
        x_emb = self.embedding(x) # (batch, emb_dim)

        x_cross = x_emb
        for layer in num_cross_layer:
            x_cross = self.cross_layers(x_emb * x_cross) + x_cross

        x_deep = self.deep_layer(x_emb)
        
        x_out = torch.cat([x_deep, x_cross], -1)
        x_out = self.out_layer(x_out)

        # x_out = torch.sigmoid(x_out)

        return x_out


class CrossNetV1(nn.Module):
    def __init__(self, num_layer, embed_dim):
        super().__init__()
        self.num_layer = num_layer
        self.cross_layers = nn.ParameterList(torch.empty(embed_dim, embed_dim) for _ in range(num_layer))

    def forward(self, x):
        x_cross = x
        for i in range(self.num_layer - 1):
            x_cross = self.cross_layers[i](x_cross) * x + x_cross
            x_cross = F.tanh(x_cross)
        x_cross = self.cross_layers[self.num_layer - 1](x_cross) * x + x_cross
        return x_cross

class CrossNetV2(nn.Module):
    def __init__(self, num_layer, embed_dim):
        super().__init__()
        self.num_layer = num_layer
        self.cross_layers = nn.ModuleList(nn.Linear(embed_dim, embed_dim) for _ in range(num_layer))

    def forward(self, x):
        x_cross = x
        for i in range(self.num_layer - 1):
            x_cross = self.cross_layers[i](x_cross) * x + x_cross
            x_cross = F.tanh(x_cross)
        x_cross = self.cross_layers[self.num_layer - 1](x_cross) * x + x_cross
        return x_cross

class CrossNetMoE(nn.Module):
    def __init__(self, num_layer, embed_dim, num_expert, low_rank = 32):
        super().__init__()
        self.num_layer = num_layer
        self.num_expert = num_expert
        self.U_list = nn.ParameterList(
            [nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_expert, embed_dim, low_rank))) for _ in range(num_layer)]
        )
        self.V_list = nn.ParameterList(
            [nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_expert, low_rank, embed_dim))) for _ in range(num_layer)]
        )
        self.C_list = nn.ParameterList(
            [nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_expert, low_rank, low_rank))) for _ in range(num_layer)]
        )
        self.gate_list = nn.ModuleList(nn.Linear(embed_dim, 1) for _ in range(num_layer))
        self.bias = nn.ParameterList(
            [nn.Parameter(nn.init.zeros_(embed_dim, 1)) for _ in range(num_layer)]
        )

        self.bias = torch.nn.ParameterList([nn.Parameter(nn.init.zeros_(
            torch.empty(in_features, 1))) for i in range(self.layer_num)])

    def forward(self, x):
        x_cross = x
        for i in range(self.num_layer - 1):
            x_cross = self.cross_layers[i](x_cross) * x + x_cross
            x_cross = F.tanh(x_cross)
        x_cross = self.cross_layers[self.num_layer - 1](x_cross) * x + x_cross
        return x_cross


class DCNv2(nn.Module):
    def __init__(self, in_feature, embedding_dim, num_cross_layer, deep_layers = [64, 32, 1], low_rank = 32):
        super().__init__()
        self.embedding = nn.Embedding(in_feature, embedding_dim)

        deep_layer_list = [nn.Linear(embedding_dim, deep_layers[0])]
        for i in range(1, len(deep_layers)):
            deep_layer_list.append(nn.ReLU())
            deep_layer_list.append(nn.Linear(deep_layers[i - 1], deep_layers[i]))

        if low_rank is not None:
            self.V_list = nn.ModuleList()
        
        # cross_l(x) = x * (W_l * x) + b_l + x
        self.cross_layers = nn.ModuleList()
        for i in range(num_cross_layer):
            self.cross_layers.append(nn.Linear(embedding_dim, embedding_dim))

        self.out_layer = nn.Linear(embedding_dim + sum(deep_layers), 1)

    def forward(self, x):
        x_emb = self.embedding(x) # (batch, emb_dim)

        x_cross = x_emb
        for layer in num_cross_layer:
            x_cross = self.cross_layers(x_emb * x_cross) + x_cross

        x_deep = self.deep_layer(x_emb)
        
        x_out = torch.cat([x_deep, x_cross], -1)
        x_out = self.out_layer(x_out)

        # x_out = torch.sigmoid(x_out)

        return x_out







model = DCN()
criterion = nn.CrossEntropyLoss()
optimizer = nn.Adam(model.parameters(), lr = 0.001)

optimizer.zero_grad()
output = model(train_data)
loss = criterion(output, target) # pred, target
loss.backward()
optimizer.step()

