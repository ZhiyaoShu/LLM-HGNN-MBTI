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

class AttentionHyperGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_mediator=False, selfloops=False, drop_rate=0.5):
        super(AttentionHyperGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_mediator = use_mediator
        self.selfloops = selfloops
        self.dropout = nn.Dropout(drop_rate)
        
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.attention = nn.Parameter(torch.Tensor(out_channels, 1))
        
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.xavier_normal_(self.attention)
    
    def forward(self, X):
        # Apply linear transformation
        H = torch.matmul(X, self.weight)
        
        # Compute attention scores
        attention_scores = torch.matmul(torch.tanh(H), self.attention)
        attention_weights = torch.softmax(attention_scores, dim=0)
        
        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)
        
        # Update node features based on attention weights
        X_updated = torch.matmul(attention_weights.transpose(0, 1), H)
        
        return X_updated

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
        selfloops: bool = False,
        drop_rate: float = 0.5,  # 保留这个参数用于配置Dropout层
    ) -> None:
        super(HyperGCN, self).__init__()
        self.fast = fast
        self.cached_g = None
        self.with_mediator = use_mediator
        self.dropout = nn.Dropout(drop_rate)  # 定义一个Dropout层
        
        # 第一层：输入特征到隐藏层
        self.layer1 = HyperGCNConv(in_channels, hid_channels, use_mediator, use_bn=use_bn)
        # 注意力层：增强隐藏层特征
        self.attention_layer = AttentionHyperGCNConv(hid_channels, hid_channels, use_mediator, selfloops, drop_rate)
        # 第二层：隐藏层到输出类别
        self.layer2 = HyperGCNConv(hid_channels, num_classes, use_mediator, use_bn=use_bn, is_last=True)

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math
        """
        # self.layers.append(
        #     HyperGCNConv(
        #         in_channels, hid_channels, use_mediator, use_bn=use_bn, drop_rate=drop_rate,
        #     )
        # )
        # self.layers.append(
        #     HyperGCNConv(
        #         hid_channels, num_classes, use_mediator, use_bn=use_bn, is_last=True
        #     )
        # )
        print("Entering forward method")
        print("Feature matrix shape:", X.shape)
        print("Number of vertices in hypergraph:", hg.num_v)
        
        if self.fast and self.cached_g is None:
            print("Creating cached graph...")
            self.cached_g = Graph.from_hypergraph_hypergcn(hg, X, self.with_mediator)
            print("Cached graph created.")
        
        X = self.layer1(X, hg)
        X = self.dropout(X)  # 应用Dropout
        X = self.attention_layer(X)  # 注意：这里假设AttentionHyperGCNConv不直接处理超图结构
        X = self.dropout(X)  # 再次应用Dropout
        X = self.layer2(X, hg)
        
        print("Exiting forward method")
        return X

def simple_test():
    in_channels = 384 
    num_classes = 17   
    hid_channels = 300 
    model = HyperGCN(
        in_channels=in_channels,
        hid_channels=hid_channels,
        num_classes=num_classes,
        use_mediator=True, 
        use_bn=False,  
        fast=True,
    )
    
    test_input = torch.randn(10, 384)  # 现在特征矩阵的行数与超图中的顶点数量匹配
    test_hypergraph = Hypergraph(num_v=10, e_list=[[0, 1], [2, 4]])
    test_output = model(test_input, test_hypergraph) # 使用修正后的特征矩阵和超图
    print("Test output:", test_output)

if __name__ == "__main__":
    simple_test()


def add_hyperedges(data, df):
    """
    Dynamcially add hyperedges to the hypergraph based on node features and the dataframe.
    """
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

def normalize_features(data):
    """
    Normalize the node features based on their distribution.
    """
    distribution = check_data_distribution(data)
    scaler = StandardScaler() if distribution == 'normal' else MinMaxScaler()
    data.x = torch.tensor(scaler.fit_transform(
        data.x.numpy()), dtype=torch.float)
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

def update_hypergraph_structure(data, new_hyperedges):
    # 假设data.hg是当前的超图对象
    # 将新的动态超边添加到超图中
    for hyperedge_id, node_id in new_hyperedges:
        if hyperedge_id not in data.hg.e_dict:
            data.hg.e_dict[hyperedge_id] = []
        data.hg.e_dict[hyperedge_id].append(node_id)
    
    # 更新超图的边列表和节点-边关系等
    data.hg.update_e_list_and_incidence_matrix()


def dynamic_hyperedges(data, model, num_clusters=100):
    # 假设data.x是节点特征矩阵，model是您的HyperGCN模型
    with torch.no_grad():
        node_embeddings = model.get_node_embeddings(data.x)  # 获取节点嵌入

    # 使用聚类算法（如K-Means）基于节点嵌入动态构建超边
    kmeans = KMeans(n_clusters=num_clusters, random_state=5).fit(node_embeddings.numpy())
    clusters = kmeans.labels_

    dynamic_hyperedges = []
    hyperedge_id = max(data.hg.e_dict.keys()) + 1  

    for node_id, cluster_label in enumerate(clusters):
        dynamic_hyperedges.append((hyperedge_id + cluster_label, node_id))
    
    # 更新超图结构
    update_hypergraph_structure(data, dynamic_hyperedges)

# def DHGCN():
#     df, _ = load_data()
#     data = pickle.load(open('edges_delete_file.pkl', 'rb'))
#     data = add_hyperedges(data, df)
#     data = normalize_features(data)
#     in_channels = 384
#     # in_channels = data.node_features.shape[0]
#     num_classes = 17  
#     hid_channels = 300 

#     model = HyperGCN(
#         in_channels=in_channels,
#         hid_channels=hid_channels,
#         num_classes=num_classes,
#         use_mediator=True, 
#         use_bn=False,  
#         fast=True,
#     )
    
#     # print("顶点数量:", df.shape[0])
#     # print("特征矩阵的行数:", data.node_features.shape[0])

#     return model, data


# if __name__ == "__main__":
#     DHGCN()

