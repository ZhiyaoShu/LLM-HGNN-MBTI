import torch
import torch.nn as nn
import dhg
import pickle
from dhg.nn import HyperGCNConv
from dhg.structure.graphs import Graph
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
        return X, hg


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

class SimpleHypergraph:
    def __init__(self, num_vertices, hyperedges):
        self.num_v = num_vertices 
        self.e = hyperedges

def Hyperedges(data, df):
    num_v = data.x.shape[0]
    edge_index = data.edge_index
    user_to_index = {username: i for i, username in enumerate(df['Username'])}

    # Construct group hyperedges
    group_hyperedges_dict = {}
    for _, row in df.iterrows():
        user = row['Username']
        user_index = user_to_index[user]
        groups = ast.literal_eval(row['Groups']) if not pd.isna(row['Groups']) else []
        for group in groups:
            if group not in group_hyperedges_dict:
                group_hyperedges_dict[group] = []
            group_hyperedges_dict[group].append(user_index)
    
    group_hyperedges_list = list(group_hyperedges_dict.values())
    
    # Clustering Nodes with K-means
    k_hyperedges_dict = {}
    k=100
    node_features = data.node_features.numpy()  # Assuming data.node_features is a tensor
    kmeans = KMeans(n_clusters=k, random_state=10).fit(node_features)
    clusters = kmeans.labels_
    
    for node_index, cluster_id in enumerate(clusters):
        if cluster_id not in k_hyperedges_dict:
            k_hyperedges_dict[cluster_id] = []
        k_hyperedges_dict[cluster_id].append(node_index)
        
    k_hyperedges_list = list(k_hyperedges_dict.values())

    # 2-hop hyperedge construction
    adj_list = defaultdict(set)
    for i, j in edge_index.t().numpy():
        adj_list[i].add(j)
        adj_list[j].add(i)  # Assuming undirected graph

    two_hop_hyperedges = []
    for node in adj_list:
        two_hop_neighbors = set(adj_list[node])
        for neighbor in adj_list[node]:
            two_hop_neighbors.update(adj_list[neighbor])
        two_hop_neighbors.discard(node)  # Remove the node itself to avoid self-loops
        two_hop_hyperedges.append(list(two_hop_neighbors))

    # Combine all types of hyperedges
    all_hyperedges = group_hyperedges_list + k_hyperedges_list + two_hop_hyperedges
    
    # No need to convert to list using .tolist()
    hypergraph = SimpleHypergraph(num_vertices=num_v, hyperedges=all_hyperedges)

    data.hg = hypergraph

    return data
# def Hyperedges(data, df):
#     num_v = data.x.shape[0]
#     edge_index = data.edge_index
#     user_to_index = {username: i for i,
#                      username in enumerate(df['Username'])}

#     group_hyperedges = []
#     group_to_hyperedge = {}
#     hyperedge_id = 0

#     for _, row in df.iterrows():
#         user = row['Username']
#         try:
#             # Convert string to list
#             groups = ast.literal_eval(row['Groups'])
#         except ValueError:
#             groups = []
#         for group in groups:
#             if group not in group_to_hyperedge:
#                 group_to_hyperedge[group] = hyperedge_id
#                 hyperedge_id += 1
#             group_hyperedges.append(
#                 (group_to_hyperedge[group], user_to_index[user]))

#     # Convert group_hyperedges to a tensor
#     group_hyperedges_tensor = torch.tensor(
#         group_hyperedges, dtype=torch.long).t().contiguous()

#     print("Shape of group-edges:", group_hyperedges_tensor.shape)
#     print("Number of unique groups:", len(group_to_hyperedge))

#     # Clustering Nodes with K-means
#     k = 100
#     node_features = data.node_features
#     kmeans = KMeans(n_clusters=k, random_state=10).fit(
#         node_features.detach().numpy())
#     clusters = kmeans.labels_

#     # Map each node to its new hyperedge (cluster)
#     cluster_to_hyperedge = {i: hyperedge_id + i for i in range(k)}
#     k_hyperedges = [(cluster_to_hyperedge[label], node)
#                     for node, label in enumerate(clusters)]
#     k_hyperedges_tensor = torch.tensor(
#         k_hyperedges, dtype=torch.long).t().contiguous()

#     print("Shape of k_hyperedges:", k_hyperedges_tensor.shape)

#     # 2-hop hyperedge
#     assert edge_index.shape[0] == 2
#     group_hyperedges = []
#     group_to_hyperedge = {}
#     hyperedge_id = 0
#     edge_index_2, edge_mask, ID_node_mask = dropout_node(
#         edge_index, p=0.0, num_nodes=num_v)

#     adj = SparseTensor.from_edge_index(
#         edge_index_2, sparse_sizes=(num_v, num_v))
#     adj = adj + adj @ adj
#     row, col, _ = adj.coo()
#     edge_index_2hop = torch.stack([row, col], dim=0)
#     edge_index_2hop, _ = remove_self_loops(edge_index_2hop)

#     print("Shape of edge_index_2hop:", edge_index_2hop.shape)

#     group_hyperedges_tensor_shape = group_hyperedges_tensor.shape[1]
#     edge_index_2hop_shape = edge_index_2hop.shape[1]
    
#     # combined_tensor = torch.cat(
#     #     [group_hyperedges_tensor, k_hyperedges_tensor, edge_index_2hop], dim=1)
    
#     # data.hyperedge_index = combined_tensor

#     # print(f"hyperedge_size", combined_tensor.shape)
#     # hyperedges_list = [hyperedge.tolist() for hyperedge in combined_tensor]
    
#     # hypergraph = SimpleHypergraph(num_vertices=num_v, hyperedges=hyperedges_list)
#     # print(combined_tensor.shape)
#     # data.hg = hypergraph

#     return data


def DHGCN():
    df, _ = load_data()
    data = pickle.load(open('edges_delete_file.pkl', 'rb'))
    data = Hyperedges(data, df)
    data = normalize_features(data)
    model = HyperGCN(1, 1, 1, use_mediator=True, fast=True, drop_rate=0.5)

    return model, data


if __name__ == "__main__":
    DHGCN()
