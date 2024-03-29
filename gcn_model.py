from typing import List, Dict
import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F


class GCNNet(torch.nn.Module):
    def __init__(self, model_params: Dict):
        super().__init__()
        self.gcn = GCN(**model_params['GCN'])
        self.mlp = MLP(**model_params['MLP'])

    def forward(self, x, edge_index, batch):
        x = self.gcn(x, edge_index)
        x = scatter(batch, x)
        x = self.mlp(x)
        return x
    

class GCN(torch.nn.Module):
    def __init__(self, channels: List[int], activation=None) -> None:
        super().__init__()
        self.convs = torch.nn.ModuleList(GCNConv(channels[i], channels[i + 1]) for i in range(len(channels) - 1))
        self.activation = activation

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                if self.activation:
                    x = self.activation(x)
                else:
                    x = F.relu(x)
        return x


class MLP(torch.nn.Module):
    def __init__(self, channels: List[int], activation=None) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(Linear(channels[i], channels[i + 1]) for i in range(len(channels) - 1))
        self.activation = activation

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                else:
                    x = F.relu(x)
        return x
    

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

        
def scatter(batch, x):
    y = torch.zeros(torch.max(batch)+1, x.size(1), dtype=x.dtype, device=x.device)
    for node, graph_node in enumerate(batch):
        y[graph_node, :] += x[node, :]
    _, counts = torch.unique_consecutive(batch, return_counts=True)
    y = y/counts.unsqueeze(1)
    return y



