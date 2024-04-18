import torch
import torch.nn as nn
import dhg
import pickle
from dhg.nn import HyperGCNConv
from dhg.structure.graphs import Graph
from dhg.structure.hypergraphs import Hypergraph
import numpy as np
from torch_sparse import SparseTensor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data_preparation import load_data
import ast
from sklearn.cluster import KMeans
from torch import nn, optim
import math
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.autograd import gradcheck
from typing import Optional
from dhg.nn import HGNNPConv
from dhg.nn import HNHNConv
from dhg.nn import UniGCNConv, UniGATConv, UniSAGEConv, UniGINConv, MultiHeadWrapper
from utils import normalize_features
from Hypergraph import get_dhg_hyperedges, custom_hyperedges


class AttentionHyperGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.5):
        super(AttentionHyperGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = nn.Dropout(drop_rate)

        # Define linear transformations
        self.query = nn.Linear(in_channels, out_channels)
        self.key = nn.Linear(in_channels, out_channels)
        self.value = nn.Linear(in_channels, out_channels)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X):
        # Calculate Q, K, and V
        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)

        # Calculate attention scores and weights
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.out_channels
        )
        attention_weights = self.softmax(attention_scores)

        # Apply dropout
        attention_output = torch.matmul(attention_weights, V)

        return attention_output


# class HyperGCN(nn.Module):
#     r"""The HyperGCN model proposed in `HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs <https://papers.nips.cc/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf>`_ paper (NeurIPS 2019).

#     Args:
#         ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
#         ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
#         ``num_classes`` (``int``): The Number of class of the classification task.
#         ``use_mediator`` (``str``): Whether to use mediator to transform the hyperedges to edges in the graph. Defaults to ``False``.
#         ``fast`` (``bool``): If set to ``True``, the transformed graph structure will be computed once from the input hypergraph and vertex features, and cached for future use. Defaults to ``True``.
#         ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.5.
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         hid_channels: int,
#         num_classes: int,
#         use_mediator: bool = False,
#         use_bn: bool = False,
#         fast: bool = True,
#         drop_rate: float = 0.5,
#     ) -> None:
#         super(HyperGCN, self).__init__()
#         self.fast = fast
#         self.cached_g = None
#         self.with_mediator = use_mediator
#         self.dropout = nn.Dropout(drop_rate)

#         self.layer1 = HyperGCNConv(
#             in_channels, hid_channels, use_mediator, use_bn=use_bn
#         )
#         # attention_layer accepts the output of the first HyperGCNConv layer
#         self.attention_layer = AttentionHyperGCNConv(
#             hid_channels, hid_channels, drop_rate
#         )
#         # Accepts the output of the attention layer
#         self.layer2 = HyperGCNConv(
#             hid_channels, num_classes, use_mediator, use_bn=use_bn, is_last=True
#         )
#         self.layers = nn.ModuleList([self.layer1, self.attention_layer, self.layer2])

#     def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
#         if self.fast and self.cached_g is None:
#             self.cached_g = Graph.from_hypergraph_hypergcn(
#                 hg, X, self.with_mediator
#             )

#         # The first layer accepts the input vertex features and the hypergraph
#         X = self.layer1(X, hg, self.cached_g)

#         # Attention layer accepts the output of the first layer
#         X = self.attention_layer(X)

#         # The second layer accepts the output of the attention layer
#         X = self.layer2(X, hg, self.cached_g)

#         return X


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
        self.skip_layers = nn.ModuleList()
        self.layers.append(
            HyperGCNConv(
                in_channels,
                hid_channels,
                use_mediator,
                use_bn=use_bn,
                drop_rate=drop_rate,
            )
        )
        # self.skip_layers.append(nn.Identity())
        self.layers.append(
            HyperGCNConv(
                hid_channels, num_classes, use_mediator, use_bn=use_bn, is_last=True
            )
        )
        # self.skip_layers.append(nn.Linear(hid_channels, num_classes))

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
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
        # if self.fast:
        #     if self.cached_g is None:
        #         self.cached_g = Graph.from_hypergraph_hypergcn(
        #             hg, X, self.with_mediator
        #         )
        #     for layer, skip_layer in zip(self.layers, self.skip_layers):
        #         skip_X = skip_layer(X)
        #         X = layer(X, hg, self.cached_g)
        #         X += skip_X
        # else:
        #     for layer, skip_layer in zip(self.layers, self.skip_layers):
        #         skip_X = skip_layer(X)
        #         X = layer(X, hg)
        #         X += skip_X
        return X


def DHGCN():
    df, _ = load_data()
    data = pickle.load(open("edges_delete_file.pkl", "rb"))
    data = get_dhg_hyperedges(data, df)
    data = normalize_features(data)
    # in_channels = data.x.shape[1]
    in_channels = 384
    num_classes = 17
    hid_channels = 384

    print("Before processing: ", data.num_v, "node numbers", len(data.e), "hyperedges.")

    model = HyperGCN(
        in_channels=in_channels,
        hid_channels=hid_channels,
        num_classes=num_classes,
        use_mediator=False,
        use_bn=True,
        fast=True,
    )

    return model, data


if __name__ == "__main__":
    DHGCN()


# def update_hypergraph_structure(data, new_hyperedges):
#     # update the hypergraph structure
#     for hyperedge_id, node_id in new_hyperedges:
#         if hyperedge_id not in data.hg.e_dict:
#             data.hg.e_dict[hyperedge_id] = []
#         data.hg.e_dict[hyperedge_id].append(node_id)

#     # Update the incidence matrix
#     data.hg.update_e_list_and_incidence_matrix()


# def dynamic_hyperedges(data, model, num_clusters=100):
#     """
#     Update the hypergraph structure with dynamic hyperedges.
#     """
#     #
#     with torch.no_grad():
#         node_embeddings = model.get_node_embeddings(data.x)  # 获取节点嵌入

#     # K-means clustering
#     kmeans = KMeans(n_clusters=num_clusters, random_state=5).fit(
#         node_embeddings.numpy()
#     )
#     clusters = kmeans.labels_

#     dynamic_hyperedges = []
#     hyperedge_id = max(data.hg.e_dict.keys()) + 1

#     for node_id, cluster_label in enumerate(clusters):
#         dynamic_hyperedges.append((hyperedge_id + cluster_label, node_id))

#     # Update the hypergraph structure
#     update_hypergraph_structure(data, dynamic_hyperedges)

# def test_hypergcn_model():
#     num_vertices = 10
#     num_classes = 3
#     in_channels = 10
#     hid_channels = 8

#     # Generate random feature matrix
#     X = torch.randn(num_vertices, in_channels)

#     # Generate simulated hypergraph structure
#     hyperedges = [[0, 1, 2], [2, 3, 4], [5, 6], [7, 8, 9]]
#     hg = Hypergraph(num_v=num_vertices, e_list=hyperedges)

#     # Initialize HyperGCN model
#     model = HyperGCN(
#         in_channels=in_channels,
#         hid_channels=hid_channels,
#         num_classes=num_classes,
#         use_mediator=False,
#         use_bn=False,
#         fast=True,
#         selfloops=False,
#         drop_rate=0.5,
#     )

#     data = type('Data', (object,), {})()
#     data.node_features = X
#     data.hg = hg
#     data.y = torch.randint(0, num_classes, (num_vertices,))  # Random target labels
#     data.train_mask = torch.rand(num_vertices) > 0.5  # Random train mask

#     optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

#     # Define the train function
#     def train(model, data, optimizer):
#         model.train()
#         optimizer.zero_grad()
#         out = model(data.node_features, data.hg)
#         target = data.y[data.train_mask].squeeze()
#         target = target.long()
#         loss = F.cross_entropy(out[data.train_mask], target)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#         return loss

#     # Call the train function
#     train_loss = train(model, data, optimizer)
#     print(f"Train loss: {train_loss.item()}")

#     print("HyperGCN model test passed successfully.")

# test_hypergcn_model()
