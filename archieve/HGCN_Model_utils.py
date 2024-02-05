from typing import Optional
import os.path as osp
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import math
from sklearn.preprocessing import OneHotEncoder
import torch_geometric
import torch.nn as nn
import pandas as pd
import random
from torch_geometric.utils import to_scipy_sparse_matrix, index_to_mask, dropout_node, remove_self_loops, to_dense_adj
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.datasets import TUDataset
from torch_sparse import SparseTensor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from scipy.sparse import csr_matrix
from torch_geometric.nn import HypergraphConv
from torch_geometric.datasets import Planetoid
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_score, recall_score, f1_score


from HGCN_Model import  node_features, edge_index, data, y_follow_encode, y_follow_label

# Set seed value
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Define the HGC model
class HyperGCN_Net(torch.nn.Module):
    def __init__(self, features, hidden, classes, heads=1):
        """
        :param features: in channels
        :param hidden:
        :param classes:
        :param heads:
        """
        super(HyperGCN_Net, self).__init__()
        self.use_attention = False
        self.hcn1 = HypergraphConv(features, hidden, use_attention=self.use_attention, heads=heads, concat=True,
                                   dropout=0.0)

        self.hcn2 = HypergraphConv(hidden * heads, classes, use_attention=self.use_attention, concat=True, dropout=0.0)

    def forward(self, data):
            x, hyperedge_index = data.x, data.hyperedge_index
            x = self.hcn1(x=x, hyperedge_index=hyperedge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.hcn2(x=x, hyperedge_index=hyperedge_index)

            return x

# Define the hyperedges among notes
def get_hyperedge(data):

    node_num = data.x.shape[0]

    edge_index = data.edge_index
    assert edge_index.shape[0] == 2
    edge_index_2, edge_mask, ID_node_mask = dropout_node(edge_index, p=0.0, num_nodes=node_num)
    # print("Max node index in edge_index_2:", edge_index_2.max().item(), "Expected node_num:", node_num)
    
    adj = SparseTensor.from_edge_index(edge_index_2, sparse_sizes=(node_num, node_num))
    adj = adj + adj @ adj
    row, col, _ = adj.coo()
    edge_index_2hop = torch.stack([row, col], dim=0)
    edge_index_2hop, _ = remove_self_loops(edge_index_2hop)
    data.hyperedge_index=edge_index_2hop

    return data

def normalize_features(data):
    scaler = StandardScaler()
    data.x = torch.FloatTensor(scaler.fit_transform(data.x))
    return data

if __name__ == '__main__':
    node_features_tensor = torch.tensor(node_features, dtype=torch.float)
    data = Data(x=node_features_tensor, edge_index=edge_index)
    data = get_hyperedge(data)
    data = normalize_features(data)

# Define the parameters in the hypergraph networks
data = normalize_features(data)
hidden = 64
classes = 70
features_describe = data.x.shape[1]
model = HyperGCN_Net(features_describe, hidden, classes, heads=1)
output = model(data)
# data.y = y_follow_encode
data.y = torch.argmax(y_follow_encode,dim=1)

# Develop entropyloss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Split dataset to train, test, and train
def validate():
    model.eval()
    with torch.no_grad():
        out = model(data)
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
    return val_loss.item()

def train():
      model.train()
      optimizer.zero_grad()
      out = model(data)
      loss = criterion(out[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()
      return loss

def test():
      model.eval()
      out = model(data)
      pred = out.argmax(dim=1)
      test_correct = pred[data.test_mask] == data.y[data.test_mask]
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
      
      test_precision = precision_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), average='macro')
      test_recall = recall_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), average='macro')
      test_f1 = f1_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), average='macro')
    
    # Calculate precision, recall, and F1 score
      return test_acc, test_precision, test_recall, test_f1

# Create train, test, validate masks
def generate_masks(y, split=(2,1,1)):
    sp1 = split[2] * 1.0 / (split[0] + split[1] + split[2])
    sp2 = split[1] * 1.0 / (split[0] + split[1])

    id_list = np.array(range(y.shape[0]))
    [train_val, test_index] = train_test_split(id_list, test_size=sp1, shuffle=True)
    [train_index, val_index] = train_test_split(train_val, test_size=sp2, shuffle=False)

    train_mask = index_to_mask(torch.as_tensor(train_index), size=y.shape[0])
    val_mask = index_to_mask(torch.as_tensor(val_index), size=y.shape[0])
    test_mask = index_to_mask(torch.as_tensor(test_index), size=y.shape[0])

    return train_mask, val_mask, test_mask

train_mask, val_mask, test_mask = generate_masks(y=data.y)
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask


# train the dataset
for epoch in range(1, 1001):
    loss=train()
    val_loss = validate()
    scheduler.step(val_loss)
    print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')

# test_acc, prediction = test()
test_acc, test_precision, test_recall, test_f1 = test()


# print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}')
#隨即猜測

# Set earlystop function
# class Earlystop:
#     def __init__(self, patience = 1, min_delta = 0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.best_loss = None
#         self.early_stop = False

#     def __call__(self, val_loss):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#         elif val_loss < self.best_loss - self.min_delta:
#             self.best_loss = val_loss
#             self.counter = 0
#         else:
#             self.counter +=1
#             if self.counter >= self.patience:
#                 self.early_stop = True

# earlystop = Earlystop (patience=20, min_delta=0)

    # earlystop(val_loss)
    # if earlystop.early_stop:
    #     print("Early stopping triggered!")
    #     break