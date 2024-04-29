import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import random
import numpy as np
from torch_sparse import SparseTensor
from torch_geometric.utils import to_scipy_sparse_matrix, index_to_mask, dropout_node, remove_self_loops, to_dense_adj
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
from model.data_preparation import process


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
    data = pickle.load(open('baseline_data2.pkl', 'rb'))
    # data = pickle.load(open('baseline_delete_edge_file.pkl', 'rb'))
    model = GCN_Net(features=data.x.shape[1], hidden=200, classes=17)
    
    # print(f"Data object: {data}")  
    # print(f"Data x: {data.x}")
    # print(f"Data y: {data.y}")
    
    unique, counts = np.unique(data.y.numpy(), return_counts=True)
    # print("Label distribution:", dict(zip(unique, counts)))
    # print("Train, Val, Test masks counts:", data.train_mask.sum().item(), data.val_mask.sum().item(), data.test_mask.sum().item())
    # print(data)
    
    return model, data

if __name__ == "__main__":
    GCN()