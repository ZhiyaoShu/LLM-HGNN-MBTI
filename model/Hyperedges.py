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
import ast
from sklearn.cluster import KMeans
from torch import nn, optim
from model.data_preparation import load_data


# Define the self loop removal function
def remove_self_loops(edge_index: torch.Tensor) -> torch.Tensor:
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]

device = torch.device("cuda" if torch.cuda else "cpu")

def get_dhg_hyperedges(data, df):
    print("Checking for non-finite values in feature matrix...")
    if not torch.isfinite(data.x).all():
        print("Non-finite values found in data.x, applying fill strategy...")
        data.x[~torch.isfinite(data.x)] = 0
    # Define the hyperedges based on data
    edge_index = data.edge_index
    edge_index_no_self_loops = remove_self_loops(edge_index)
    print("Edge index shape:", edge_index_no_self_loops.shape)
    # Create a graph from the edge index
    _g = dhg.Graph(
        data.x.size(0), edge_index_no_self_loops.t().tolist(), merge_op="mean"
    )
    print("Graph:", _g)
    # Add nodes into the hypergraph
    hg = dhg.Hypergraph(data.x.size(0))

    # Add hyperedges into the hypergraph
    hg.add_hyperedges_from_graph_kHop(_g, k=2, only_kHop=False, group_name="kHop")

    # Clustering-based hyperedges
    k = 100
    hg.add_hyperedges_from_feature_kNN(data.x, k, group_name="feature_kNN")

    # Group-based hyperedges
    user_to_index = {username: i for i, username in enumerate(df["Username"])}

    # Initialize group to hyperedges mapping
    group_to_hyperedge = {
        group: idx + hg.num_e
        for idx, group in enumerate(set(sum([ast.literal_eval(row["Groups"]) for _, row in df.iterrows() if row["Groups"]], [])))
    }
    
    # Initialize a dictionary to hold nodes for each group
    group_nodes = {group_id: [] for group_id in group_to_hyperedge.values()}
    
    for _, row in df.iterrows():
        user = row["Username"]
        try:
            groups = ast.literal_eval(row["Groups"])
        except ValueError:
            continue  # Skip if groups cannot be parsed
        user_index = user_to_index[user]
        for group in groups:
            group_id = group_to_hyperedge[group]
            group_nodes[group_id].append(user_index)
    
    # Add each group's hyperedge with its nodes
    for group_id, nodes in group_nodes.items():
        hg.add_hyperedges(nodes, group_name=group_id)  

    data.hg = hg
    return data


# data = pickle.load(open("edges_delete_file.pkl", "rb"))
# df, _ = load_data()
# get_dhg_hyperedges(data, df)
# user_to_index = data.user_to_index
# user_to_index_items = list(user_to_index.items())
# for user, index in user_to_index_items[:10]: 
#     print(f"{user}: {index}")


# Hyperedges implementation
def custom_hyperedges(data, df):
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

    print("Number of hyperedges after validation:", len(hyperedge_list))

    # Create the Hypergraph object
    hypergraph = Hypergraph(num_v=node_num, e_list=hyperedge_list)

    data.hg = hypergraph
    data.e = hyperedge_list
    print("Type of variable expected to be list:", type(hyperedge_list))
    print("Number of hyperedges:", len(data.e))
    data.num_v = node_num

    return data

    # groups = data.groups

    # group_to_nodes = {}  # Maps each group to a list of node indices
    # for user, index in data.user_to_index.items():
    #     # Retrieve the groups for the current user
    #     for group in groups:
    #         if group not in group_to_nodes:
    #             group_to_nodes[group] = []
    #         group_to_nodes[group].append(index)

    # for nodes in group_to_nodes.values():
    #     if len(nodes) > 1:
    #         hg.add_hyperedges(nodes, group_name="group")

    # # Assign the constructed hypergraph back to the data object
    # data.hg = hg