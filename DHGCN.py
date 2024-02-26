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


class HGNNP(nn.Module):
    r"""The HGNN :sup:`+` model proposed in `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        # use_skip_connections=False,
    ) -> None:
        super().__init__()
        # self.use_skip_connections = use_skip_connections
        self.layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        self.layers.append(
            HGNNPConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        # # add a linear layer to match the dimensions
        # if self.use_skip_connections:
        #     self.skip_layers.append(nn.Linear(in_channels, hid_channels))

        self.layers.append(
            HGNNPConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )
        # # add a linear layer to match the dimensions
        # if self.use_skip_connections and in_channels != num_classes:
        #     self.skip_layers.append(nn.Linear(hid_channels, num_classes))

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return X


class HNHN(nn.Module):
    r"""The HNHN model proposed in `HNHN: Hypergraph Networks with Hyperedge Neurons <https://arxiv.org/pdf/2006.12278.pdf>`_ paper (ICML 2020).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HNHNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            HNHNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return X


class UniGAT(nn.Module):
    r"""The UniGAT model proposed in `UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks <https://arxiv.org/pdf/2105.00956.pdf>`_ paper (IJCAI 2021).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``num_heads`` (``int``): The Number of attention head in each layer.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): The dropout probability. Defaults to ``0.5``.
        ``atten_neg_slope`` (``float``): Hyper-parameter of the ``LeakyReLU`` activation of edge attention. Defaults to 0.2.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        num_heads: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        atten_neg_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.drop_layer = nn.Dropout(drop_rate)
        self.multi_head_layer = MultiHeadWrapper(
            num_heads,
            "concat",
            UniGATConv,
            in_channels=in_channels,
            out_channels=hid_channels,
            use_bn=use_bn,
            drop_rate=drop_rate,
            atten_neg_slope=atten_neg_slope,
        )
        # The original implementation has applied activation layer after the final layer.
        # Thus, we donot set ``is_last`` to ``True``.
        self.out_layer = UniGATConv(
            hid_channels * num_heads,
            num_classes,
            use_bn=use_bn,
            drop_rate=drop_rate,
            atten_neg_slope=atten_neg_slope,
            is_last=False,
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        X = self.drop_layer(X)
        X = self.multi_head_layer(X=X, hg=hg)
        X = self.drop_layer(X)
        X = self.out_layer(X, hg)
        return X


# Hyperedges implementation
def get_hyperedges(data, df):
    """
    Dynamcially add hyperedges to the hypergraph based on node features and the dataframe.
    """
    node_num = data.x.shape[0]
    edge_index = data.edge_index
    user_to_index = {username: i for i, username in enumerate(df["Username"])}

    group_hyperedges = []
    group_to_hyperedge = {}
    hyperedge_id = 0

    for _, row in df.iterrows():
        user = row["Username"]
        try:
            groups = ast.literal_eval(row["Groups"])
        except ValueError:
            groups = []
        for group in groups:
            if group not in group_to_hyperedge:
                group_to_hyperedge[group] = hyperedge_id
                hyperedge_id += 1
            group_hyperedges.append((group_to_hyperedge[group], user_to_index[user]))

    # # Convert group_hyperedges to a tensor
    # group_hyperedges_tensor = (
    #     torch.tensor(group_hyperedges, dtype=torch.long).t().contiguous()
    # )

    # print("Shape of group-edges:", group_hyperedges_tensor.shape)

    # Clustering Nodes with K-means
    k = 100
    node_features = data.x.numpy()  # Assuming data.x is a PyTorch tensor.
    kmeans = KMeans(n_clusters=k, random_state=5).fit(node_features)
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

    # print("Number of hyperedges:", len(hyperedge_list))

    valid_hyperedges = [he for he in hyperedge_list if len(he) >= 2]
    if len(valid_hyperedges) != len(hyperedge_list):
        print(
            f"Warning: Removed {len(hyperedge_list) - len(valid_hyperedges)} hyperedges with fewer than 2 vertices."
        )
    hyperedge_list = valid_hyperedges

    # print("Number of hyperedges after validation:", len(hyperedge_list))

    # Create the Hypergraph object
    hypergraph = Hypergraph(num_v=node_num, e_list=hyperedge_list)
    data.hg = hypergraph
    data.e = hyperedge_list
    # print("Type of variable expected to be list:", type(hyperedge_list))
    data.num_v = node_num

    return data


def check_data_distribution(data):
    """
    Check the distribution of the features in the data.
    """
    from scipy.stats import shapiro

    sample = data.x[np.random.choice(data.x.size(0), 1000, replace=False)].numpy()

    stat, p = shapiro(sample)
    return "normal" if p > 0.05 else "non-normal"


def normalize_features(data):
    """
    Normalize the node features based on their distribution.
    """
    distribution = check_data_distribution(data)
    scaler = StandardScaler() if distribution == "normal" else MinMaxScaler()
    data.x = torch.tensor(scaler.fit_transform(data.x.numpy()), dtype=torch.float)
    return data


def DHGCN():
    df, _ = load_data()
    data = pickle.load(open("edges_delete_file.pkl", "rb"))
    data = get_hyperedges(data, df)
    data = normalize_features(data)
    # in_channels = data.x.shape[1]
    in_channels = 384
    num_classes = 17
    hid_channels = 384

    print("Before processing: ", data.num_v, "node numbers", len(data.e), "hyperedges.")

    # model = HyperGCN(
    #     in_channels=in_channels,
    #     hid_channels=hid_channels,
    #     num_classes=num_classes,
    #     use_mediator=False,
    #     use_bn=True,
    #     fast=True,
    # )

    model = HGNNP(
        in_channels=in_channels,
        hid_channels=hid_channels,
        num_classes=num_classes,
        use_bn=True,
    )

    # model = HNHN(
    #     in_channels=in_channels,
    #     hid_channels=hid_channels,
    #     num_classes=num_classes,
    #     use_bn=True,
    # )

    # model = UniGAT(
    #     in_channels=in_channels,
    #     hid_channels=hid_channels,
    #     num_classes=num_classes,
    #     num_heads=1,
    #     use_bn=True,
    #     drop_rate=0.5,
    #     atten_neg_slope=0.2,
    # )

    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         print(f"Parameter {name} does not require gradients.")
    #     else:
    #         print(f"Parameter {name} requires gradients.")

    # model_path = 'hypergcn_model_state_dict.pth'
    # torch.save(model.state_dict(), model_path)

    return model, data


if __name__ == "__main__":
    DHGCN()
    # test_hypergcn_model()

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
