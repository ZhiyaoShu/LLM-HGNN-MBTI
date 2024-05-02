import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv
import pickle


class GAT_Net(torch.nn.Module):
    def __init__(self, features, hidden, classes, heads=1):
        super(GAT_Net, self).__init__()
        self.gat1 = GATConv(features, hidden, heads=heads, concat=True, dropout=0.0)
        self.gat2 = GATConv(hidden*heads, classes, concat=True, dropout=0.0)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)

        return x

def GAT():
    # data = pickle.load(open('baseline_data1.pkl', 'rb'))
    data = pickle.load(open("checkout/test_train_change7.pkl", "rb"))
    # data = pickle.load(open('graph_with_embedding2.pkl', 'rb'))
    model = GAT_Net(features=data.x.shape[1], hidden=200, classes=17, heads=1)
    
    print(f"Data object: {data}")  
    print(f"Data x: {data.x}")
    print(f"Data y: {data.y}")
    
    # unique, counts = np.unique(data.y.numpy(), return_counts=True)
    # print("Label distribution:", dict(zip(unique, counts)))
    # print("Train, Val, Test masks counts:", data.train_mask.sum().item(), data.val_mask.sum().item(), data.test_mask.sum().item())
    
    return model, data

if __name__ == "__main__":
    GAT()