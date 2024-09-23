import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pickle
import os
import parse_arg
from dataloader import baseline_data_process, data_preparation
import logging

args = parse_arg.parse_arguments()

if args.use_llm:
    data_path = "data_features.pkl"
else:
    data_path = "baseline_data.pkl"

class GCN_Net(torch.nn.Module):
    def __init__(self, features, hidden, classes):
        super (GCN_Net, self).__init__()
        self.conv1 = GCNConv(features, hidden)
        self.conv2 = GCNConv(hidden, classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def GCN():
    if not os.path.exists(data_path):
        if args.use_llm:
            data = data_preparation()
        else:
            data = baseline_data_process()
    else:
        logging.info(f"Loading data from {data_path}")
        data = pickle.load(open(data_path, "rb"))
    model = GCN_Net(features=data.x.shape[1], hidden=200, classes=16)

    return model, data

if __name__ == "__main__":
    GCN()
