import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pickle


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
    data = pickle.load(open('baseline_data.pkl', 'rb'))
    model = GCN_Net(features=data.x.shape[1], hidden=200, classes=16)
    
    return model, data

if __name__ == "__main__":
    GCN()