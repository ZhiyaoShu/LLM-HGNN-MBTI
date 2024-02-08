import torch
import torch.nn as nn
import dhg
import pickle
from dhg.nn import HyperGCNConv
from dhg.structure.graphs import Graph
from dhg.structure.hypergraphs import Hypergraph
from typing import Optional
import numpy as np
from torch_sparse import SparseTensor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch_geometric.transforms as T
from data_preparation import load_data
import ast
from sklearn.cluster import KMeans
from torch_geometric.utils import dropout_node, remove_self_loops
from torch_geometric.data import Data
import numpy as np
from collections import defaultdict
import pandas as pd
import ast
import pickle
import random

class HyperGCN(nn.Module):
    r"""The HyperGCN model proposed in `HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs <https://papers.nips.cc/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf>`_ paper (NeurIPS 2019).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_mediator`` (``str``): Whether to use mediator to transform the hyperedges to edges in the graph. Defaults to ``False``.
        ``fast`` (``bool``): If set to ``True``, the transformed graph structure will be computed once from the input hypergraph and vertex features, and cached for future use. Defaults to ``True``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.5.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_mediator: bool = False,
        use_bn: bool = False,
        fast: bool = True,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.fast = fast
        self.cached_g = None
        self.with_mediator = use_mediator
        self.layers = nn.ModuleList()
        self.layers.append(
            HyperGCNConv(
                in_channels, hid_channels, use_mediator, use_bn=use_bn, drop_rate=drop_rate,
            )
        )
        self.layers.append(
            HyperGCNConv(
                hid_channels, num_classes, use_mediator, use_bn=use_bn, is_last=True
            )
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math
        """
        if self.fast:
            if self.cached_g is None:
                self.cached_g = Graph.from_hypergraph_hypergcn(
                    hg, X, self.with_mediator
                )
            for layer in self.layers:
                X = layer(X, hg, self.cached_g)
        else:
            for layer in self.layers:
                X = layer(X, hg)
        return X

def add_hyperedges(data, df):
    node_num = data.x.shape[0]
    edge_index = data.edge_index
    user_to_index = {username: i for i,
                     username in enumerate(df['Username'])}

    group_hyperedges = []
    group_to_hyperedge = {}
    hyperedge_id = 0

    for _, row in df.iterrows():
        user = row['Username']
        try:
            groups = ast.literal_eval(row['Groups'])
        except ValueError:
            groups = []
        for group in groups:
            if group not in group_to_hyperedge:
                group_to_hyperedge[group] = hyperedge_id
                hyperedge_id += 1
            group_hyperedges.append((group_to_hyperedge[group], user_to_index[user]))


    # Convert group_hyperedges to a tensor
    group_hyperedges_tensor = torch.tensor(
        group_hyperedges, dtype=torch.long).t().contiguous()

    print("Shape of group-edges:", group_hyperedges_tensor.shape)

    # Clustering Nodes with K-means
    k = 100
    node_features = data.x.numpy()  # Assuming data.x is a PyTorch tensor.
    kmeans = KMeans(n_clusters=100, random_state=5).fit(node_features)
    clusters = kmeans.labels_

    cluster_hyperedges = []
    for node, label in enumerate(clusters):
        cluster_hyperedges.append((hyperedge_id + label, node))
    hyperedge_id += k

    # 2-hop hyperedge
    adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(node_num, node_num))
    adj_sq = adj @ adj
    row, col, _ = adj_sq.coo()
    two_hop_hyperedges = []
    for idx in range(row.size(0)):
        if row[idx] != col[idx]:  # Removing self loops
            two_hop_hyperedges.append((hyperedge_id, row[idx].item()))
            two_hop_hyperedges.append((hyperedge_id, col[idx].item()))
        hyperedge_id += 1
    
    # Combine all hyperedges
    all_hyperedges = group_hyperedges + cluster_hyperedges + two_hop_hyperedges
    hyperedge_dict = {}
    for hyperedge_id, node_id in all_hyperedges:
        if hyperedge_id not in hyperedge_dict:
            hyperedge_dict[hyperedge_id] = []
        hyperedge_dict[hyperedge_id].append(node_id)
    
    hyperedge_list = list(hyperedge_dict.values())

    print("Number of hyperedges:", len(hyperedge_list))
    
    valid_hyperedges = [he for he in hyperedge_list if len(he) >= 2]
    if len(valid_hyperedges) != len(hyperedge_list):
        print(f"Warning: Removed {len(hyperedge_list) - len(valid_hyperedges)} hyperedges with fewer than 2 vertices.")
    hyperedge_list = valid_hyperedges

    print("Number of hyperedges after validation:", len(hyperedge_list))

    # Create the Hypergraph object
    hypergraph = Hypergraph(num_v=node_num, e_list=hyperedge_list)
    data.hg = hypergraph
    data.e = hyperedge_list
    print("Type of variable expected to be list:", type(hyperedge_list))
    data.num_v = node_num

    return data

def check_data_distribution(data):
    """
    Check the distribution of the features in the data.
    """
    from scipy.stats import shapiro
    sample = data.x[np.random.choice(
        data.x.size(0), 1000, replace=False)].numpy()

    stat, p = shapiro(sample)
    return 'normal' if p > 0.05 else 'non-normal'


def normalize_features(data):
    """
    Normalize the node features based on their distribution.
    """
    distribution = check_data_distribution(data)
    scaler = StandardScaler() if distribution == 'normal' else MinMaxScaler()
    data.x = torch.tensor(scaler.fit_transform(
        data.x.numpy()), dtype=torch.float)
    return data

def DHGCN():
    df, _ = load_data()
    data = pickle.load(open('edges_delete_file.pkl', 'rb'))
    data = add_hyperedges(data, df)
    data = normalize_features(data)
    in_channels = data.node_features.shape[1]
    model = HyperGCN(
        use_mediator=True, 
        hid_channels = 300,
        in_channels=in_channels,
        fast=True, 
        num_classes = 17,
        drop_rate=0.5)

    return model, data


if __name__ == "__main__":
    DHGCN()
