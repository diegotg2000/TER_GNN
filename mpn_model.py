import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn
from torch_geometric.nn import MessagePassing





class GraphEncoder(nn.Module):

    def __init__(self, node_dim, edge_dim, encoding_size=64) -> None:
        super().__init__()
        # encoding with two linear layers with relu activation
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, encoding_size),
            nn.ReLU(),
            nn.Linear(encoding_size, encoding_size),
            nn.ReLU()
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, encoding_size),
            nn.ReLU(),
            nn.Linear(encoding_size, encoding_size),
            nn.ReLU()
        )

    def forward(self, x, edge_features):
        x = self.node_encoder(x)
        edge_features = self.edge_encoder(edge_features)
        return x, edge_features
    

class MessageLayer(MessagePassing):

    def __init__(self, encoding_size=64) -> None:
        super().__init__(aggr='add')

        self.edge_updater_2 = nn.Sequential(
            nn.Linear(3*2*encoding_size, encoding_size),
            nn.ReLU(),
            nn.Linear(encoding_size, encoding_size),
            nn.ReLU()
        )

        self.node_updater_2 = nn.Sequential(
            nn.Linear(2*2*encoding_size, encoding_size),
            nn.ReLU(),
            nn.Linear(encoding_size, encoding_size),
            nn.ReLU()
        )


    def forward(self, x, edge_features, edge_index):
        """Forward pass for the G step of the DeepMind
        paper. 

        Args:
            x (tensor): It should be the concatenation of the 
            previous node features and the node features at step 0.
            edge_index (tensor): connectivity of the graph
            edge_features (tensor): edge features, also concatenated with the 
            features at step 0
        """        

        extended_edge_features = torch.concat([
            edge_features, 
            x[edge_index[0]], 
            x[edge_index[1]]
        ], dim=1)

        out = self.propagate(edge_index, x=x, edge_features=edge_features)

        edge_features = self.edge_updater_2(extended_edge_features)


        return out, edge_features
    
    def message(self, edge_features):
        return edge_features

    def update(self, aggr_output, x):
        # x: [N, 2*encoding_size]
        # aggr_output: [N, 2*encoding_size]
        x = torch.concat([
            x, 
            aggr_output
        ], axis=1)

        x = self.node_updater_2(x)

        return x



class PaperGNN(nn.Module):
    
    def __init__(self, n_layers) -> None:
        super().__init__()

        self.graph_encoder = GraphEncoder(node_dim=2, edge_dim=3, encoding_size=64)

        self.layers = nn.ModuleList([
            MessageLayer(encoding_size=64) for _ in range(n_layers)
        ])

        self.node_decoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_features, edge_index):
        x_0, e_0 = self.graph_encoder(x, edge_features)

        x, edge_features = x_0, e_0

        for layer in self.layers:
            x = torch.concatenate([
                x, x_0
            ], dim=1)
    
            edge_features = torch.concatenate([
                edge_features, e_0
            ], dim=1)
            
            x, edge_features = layer(x, edge_features, edge_index)

        x = self.node_decoder(x)

        return x


    

