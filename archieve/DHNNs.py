import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import DeepGCNLayer, GENConv
from torch.nn import LayerNorm, Linear, ReLU
from dhg.nn import HGNNPConv
import pickle
from Hypergraph import custom_hyperedges
from utils import normalize_features
from data_preparation import load_data
from dhg.nn import UniGATConv


class DeeperHNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, edge_dim, num_layers):
        super().__init__()

        self.node_encoder = Linear(input_dim, hidden_dim)
        self.edge_encoder = Linear(edge_dim, hidden_dim)
        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):

            conv = HGNNPConv(hidden_dim, hidden_dim, use_bn=False, drop_rate=0.0)

            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(
                conv, norm, act, block="res+", dropout=0.1, ckpt_grad=i % 3
            )
            self.layers.append(layer)

        self.lin = Linear(hidden_dim, out_dim)

    def forward(self, data):
        x = data.x
        hg = data.hg

        x = self.node_encoder(x)

        x = self.layers[0].conv(x, hg)

        for layer in self.layers[1:]:
            x = layer(x, hg)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        return self.lin(x)


class DeeperGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, edge_dim, num_layers):
        super().__init__()

        self.node_encoder = Linear(input_dim, hidden_dim)
        self.edge_encoder = Linear(edge_dim, hidden_dim)
        # self.use_attention = False

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(
                hidden_dim,
                hidden_dim,
                aggr="softmax",
                t=1.0,
                learn_t=True,
                num_layers=2,
                norm="layer",
            )

            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(
                conv, norm, act, block="res+", dropout=0.1, ckpt_grad=i % 3
            )
            self.layers.append(layer)

        self.lin = Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # if data has no edge_attr, then edge_attr will automatically return None
        x = self.node_encoder(x)

        edge_attr = self.edge_encoder(edge_attr) if edge_attr is not None else None

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        return self.lin(x)


def DGNN():
    df, _ = load_data()
    # data = pickle.load(open("graph_data.pkl", "rb"))
    data = pickle.load(open("baseline_delete_edge_file.pkl", "rb"))
    data = custom_hyperedges(data, df)
    data = normalize_features(data)
    node_features = data.x

    print("number of node features:", node_features.shape[1])

    model = DeeperHNN(
        input_dim=node_features.shape[1],
        hidden_dim=300,
        out_dim=17,
        num_layers=2,
        edge_dim=12,
    )

    return model, data


def DHGCN():
    df, _ = load_data()
    data = pickle.load(open("graph_data.pkl", "rb"))
    # data = pickle.load(open("baseline_delete_edge_file.pkl", "rb"))
    data = custom_hyperedges(data, df)
    data = normalize_features(data)
    model = DeeperGNN(
        input_dim=data.x.shape[1],
        hidden_dim=300,
        out_dim=17,
        num_layers=2,
        edge_dim=12,
    )

    return model, data


if __name__ == "__main__":
    DHGCN()

    # save_hyperedges(data.hg, 'hypergraph.pkl')
    # save_model(model, 'deep_hgnnp_model.pth')
