import torch
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import dropout_node, remove_self_loops
from torch_sparse import SparseTensor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch_geometric.nn import HypergraphConv, LayerNorm
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
import ast
import pickle
import random
from data_preparation import process, load_data


# Define the HGC model
class HyperGCN_Net(torch.nn.Module):
    def __init__(self, features, hidden, classes, heads=1):
        """
        :param features: in channels
        :param hidden:
        :param classes:
        :param heads:
        :param dropout_rate:
        """
        super(HyperGCN_Net, self).__init__()
        self.use_attention = False
        self.dropout_rate = 0.5
        self.hcn1 = HypergraphConv(
            features, hidden, use_attention=self.use_attention, heads=heads, concat=True, dropout=0.0)
        self.norm1 = LayerNorm(hidden * heads)

        self.hcn2 = HypergraphConv(
            hidden * heads, classes, use_attention=self.use_attention, concat=True, dropout=0.0)
        self.norm2 = LayerNorm(classes)

    def forward(self, data):
        x, hyperedge_index = data.x, data.hyperedge_index
        x = self.hcn1(x, hyperedge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.hcn2(x, hyperedge_index)
        
        # # res = x
        # x = self.hcn1(x, hyperedge_index)
        # # x = self.norm1(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # x = self.hcn2(x, hyperedge_index)
        # # x = self.norm2(x)

        return F.log_softmax(x, dim=1)

# # Define the hyperedges among notes


def get_hyperedge(data, df):
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
            # Convert string to list
            groups = ast.literal_eval(row['Groups'])
        except ValueError:
            groups = []
        for group in groups:
            if group not in group_to_hyperedge:
                group_to_hyperedge[group] = hyperedge_id
                hyperedge_id += 1
            group_hyperedges.append(
                (group_to_hyperedge[group], user_to_index[user]))

    # Convert group_hyperedges to a tensor
    group_hyperedges_tensor = torch.tensor(
        group_hyperedges, dtype=torch.long).t().contiguous()
    
    print("Shape of group-edges:", group_hyperedges_tensor.shape)

    # Clustering Nodes with K-means
    k = 50 # 太少了
    node_features = data.node_features
    kmeans = KMeans(n_clusters=k, random_state=5).fit(
        node_features.detach().numpy())
    clusters = kmeans.labels_

    # Map each node to its new hyperedge (cluster)
    cluster_to_hyperedge = {i: hyperedge_id + i for i in range(k)}
    k_hyperedges = [(cluster_to_hyperedge[label], node)
                    for node, label in enumerate(clusters)]
    k_hyperedges_tensor = torch.tensor(
        k_hyperedges, dtype=torch.long).t().contiguous()
    
    print("Shape of k_hyperedges:", k_hyperedges_tensor.shape)
    # 2-hop hyperedge
    assert edge_index.shape[0] == 2
    group_hyperedges = []
    group_to_hyperedge = {}
    hyperedge_id = 0
    edge_index_2, edge_mask, ID_node_mask = dropout_node(
        edge_index, p=0.0, num_nodes=node_num)

    adj = SparseTensor.from_edge_index(
        edge_index_2, sparse_sizes=(node_num, node_num))
    adj = adj + adj @ adj
    row, col, _ = adj.coo()
    edge_index_2hop = torch.stack([row, col], dim=0)
    edge_index_2hop, _ = remove_self_loops(edge_index_2hop)
    # print("edge_index_2hop:", edge_index_2hop)
    # print("Type of edge_index_2hop:", type(edge_index_2hop))
    print("Shape of edge_index_2hop:", edge_index_2hop.shape)

    # combined_hyperedge_index = torch.stack([group_hyperedges_tensor, edge_index_2hop], dim=1)

    group_hyperedges_tensor_shape = group_hyperedges_tensor.shape[1]
    edge_index_2hop_shape = edge_index_2hop.shape[1]

    # Calculate the new shape
    new_shape = (2, group_hyperedges_tensor_shape +
                 edge_index_2hop_shape + k_hyperedges_tensor.shape[1])

    # Create a new tensor with the calculated shape, filled with zeros
    combined_tensor = torch.zeros(new_shape, dtype=torch.long)

    # Insert group_hyperedges_tensor into the new tensor
    combined_tensor[:,
                    :group_hyperedges_tensor_shape] = group_hyperedges_tensor

    # Insert edge_index_2hop into the new tensor
    combined_tensor[:, group_hyperedges_tensor_shape:group_hyperedges_tensor_shape +
                    edge_index_2hop_shape] = edge_index_2hop

    # Insert k_hyperedges_tensor into the new tensor
    combined_tensor[:, group_hyperedges_tensor_shape +
                    edge_index_2hop_shape:] = k_hyperedges_tensor

    data.hyperedge_index = combined_tensor

    print(f"hyperedge_size", combined_tensor.shape)

    return data


def check_data_distribution(data):
    """
    Check the distribution of the features in the data.

    :param data: Data object containing the features.
    :return: Distribution type ('normal' or 'non-normal').
    """
    from scipy.stats import shapiro

    # Use a sample of the data for the Shapiro-Wilk test
    sample = data.x[np.random.choice(data.x.shape[0], 1000, replace=False)]

    # Shapiro-Wilk test for normality
    stat, p = shapiro(sample)
    if p > 0.05:
        # Data looks Gaussian (normal)
        return 'normal'
    else:
        # Data does not look Gaussian (normal)
        return 'non-normal'


def normalize_features(data):
    distribution = check_data_distribution(data)
    if distribution == 'normal':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    data.x = torch.FloatTensor(scaler.fit_transform(data.x))
    return data


def HGCN():
    # Preprocess the data
    df, _ = load_data()
    # data = process()
    # with open('graph_data.pkl', 'rb') as file:
    #     data = pickle.load(file)
    
    data = pickle.load(open('edges_delete_file.pkl', 'rb'))
    data = get_hyperedge(data, df)
    # data = hyperedge_attr(data)
    # print("Hyperedge attributes:", data.hyperedge_attr)
    data = normalize_features(data)

    model = HyperGCN_Net(
        features=data.x.shape[1], hidden=200, classes=17, heads=1)
        
    # Move model and data to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    # print(f"Data object: {data}")
    # print(f"Data x: {data.x}")
    # print(f"Data y: {data.y}")

    # unique, counts = np.unique(data.y.numpy(), return_counts=True)
    # print("Label distribution:", dict(zip(unique, counts)))
    print("Train, Val, Test masks counts:", data.train_mask.sum().item(),
          data.val_mask.sum().item(), data.test_mask.sum().item())
    # total_edges = data.hyperedge_index.size(1)
    
    # num_edges_to_keep = total_edges - int(total_edges * 1)
    # # Shuffle indices
    # indices = list(range(total_edges))
    # # Select indices to keep
    # indices_to_keep = random.sample(indices, num_edges_to_keep)
    # # Select edges based on these indices
    # remaining_edges = data.hyperedge_index[:, indices_to_keep]
    # data.hyperedge_index = remaining_edges
    print(data)

    # if torch.isnan(data.x).any() or torch.isinf(data.x).any():
    #     raise ValueError("Data contains NaN or infinite values")
    # print(data.get_hyperedge())
    
    return model, data


if __name__ == "__main__":
    HGCN()

# def hyperedge_attr(data):
#     node_features = data.x
#     hyperedge_index = data.hyperedge_index

#     # Ensure the indices are within the range of node features
#     max_node_index = node_features.size(0) - 1

#     # Determine the number of unique hyperedges
#     num_hyperedges = hyperedge_index[0].max().item() + 1
#     unique_hyperedges = hyperedge_index[0].unique()
#     print("Unique hyperedges:", unique_hyperedges)
#     print("Number of unique hyperedges:", len(unique_hyperedges))

#     # Create a tensor to store hyperedge attributes
#     hyperedge_attr = torch.zeros(num_hyperedges, node_features.size(1))

#     # Calculate the mean of node features for each hyperedge
#     for edge in range(num_hyperedges):
#         nodes = (hyperedge_index[0] == edge).nonzero(as_tuple=True)[0]
#         # nodes = (hyperedge_index[0] == edge).nonzero(as_tuple=True)[0]
#         # if len(nodes) > 0:
#         #     hyperedge_attr[edge] = node_features[nodes].mean(dim=0)
#         valid_nodes = nodes[nodes <= max_node_index]

#         if len(valid_nodes) > 0:
#             hyperedge_attr[edge] = node_features[valid_nodes].mean(dim=0)

#     data.hyperedge_attr = hyperedge_attr
#     return data
