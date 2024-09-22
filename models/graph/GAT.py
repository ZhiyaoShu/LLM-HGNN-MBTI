import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import pickle
import parse_arg
from dataloader import baseline_data_process
import os

args = parse_arg.parse_arguments()

if args.use_llm:
    data_path = "data_features.pkl"
else:
    data_path = "baseline_data.pkl"


class GAT_Net(torch.nn.Module):
    def __init__(self, features, hidden, classes, heads=1):
        super(GAT_Net, self).__init__()
        self.gat1 = GATConv(features, hidden, heads=heads, concat=True, dropout=0.0)
        self.gat2 = GATConv(hidden * heads, classes, concat=True, dropout=0.0)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)

        return x


def GAT():
    data = pickle.load(open(data_path, "rb"))
    model = GAT_Net(features=data.x.shape[1], hidden=200, classes=16, heads=1)

    return model, data
