from typing import Optional
import os.path as osp
import torch
import torch.utils.data
import torch.nn.functional as F
import math
import pandas as pd
from torch_geometric.utils import to_scipy_sparse_matrix, index_to_mask, dropout_node, remove_self_loops, to_dense_adj
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, normalize
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils import shuffle
from scipy.sparse import csr_matrix
from torch_geometric.nn import HypergraphConv
from torch_geometric.datasets import Planetoid, TUDataset
from torch_sparse import SparseTensor
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# df = pd.read_csv('dataset/new_df.csv')
# reciprocity_df = pd.read_csv('dataset/reciprocity_df.csv')
updated_new_df = pd.read_csv('dataset/merge_new_df.csv')

drop_mbti_df = updated_new_df.drop(columns=["MBTI"])

# Function to encode MBTI to binary representation
def encode_mbti(mbti):
    encoding = {
        'I': '0', 'E': '1',
        'N': '0', 'S': '1',
        'T': '0', 'F': '1',
        'J': '0', 'P': '1',
    }
    default_encoding = [-1, -1, -1, -1]
    # Check if mbti is unknown or NaN
    if mbti in ['Unknown', 'nan', None] or pd.isna(mbti):
        return default_encoding
    encoded = [int(encoding.get(char, -1)) for char in mbti]
    return encoded if len(encoded) == 4 else default_encoding

# Function to encode MBTI to numerical representation
# Numbers 0-15
def encode_mbti_number(mbti):
    mbti_to_number = {
    'INTJ': 1, 'ENTJ': 2, 'INTP': 3, 'ENTP': 4,
    'INFJ': 5, 'ENFJ': 6, 'INFP': 7, 'ENFP': 8,
    'ISTJ': 9, 'ESTJ': 10, 'ISFJ': 11, 'ESFJ': 12,
    'ISTP': 13, 'ESTP': 14, 'ISFP': 15, 'ESFP': 16
    }
    default_number = 0
    if mbti in ['Unknown', 'nan', None] or pd.isna(mbti):
        return default_number
    ordered = [int(mbti_to_number.get(char, -1)) for char in mbti]
    return mbti_to_number.get(mbti, 0)

# Encoding MBTI categories
updated_new_df.loc[:, 'Encoded'] = updated_new_df['MBTI'].apply(encode_mbti)
updated_new_df.loc[:, 'Label'] = updated_new_df['MBTI'].apply(encode_mbti_number)

# Tensor conversions
y_follow_encode = torch.tensor(updated_new_df['Encoded'], dtype=torch.long)
y_follow_label = torch.tensor(updated_new_df['Label'].values, dtype=torch.long).unsqueeze(1)

# Load and process data
embeddings_df = pd.read_json("dataset/embedings/merged_embeddings.json")
usernames_df = drop_mbti_df[['Username']]

# Reset the index to ensure alignment
usernames_df = usernames_df.reset_index(drop=True)
embeddings_df = embeddings_df.reset_index(drop=True)

# Combine 'Username' and 'Embedding' into a new DataFrame
combined_df = pd.concat([usernames_df, embeddings_df], axis=1)

# Generate user-to-index mapping
user_to_index = {username: i for i, username in enumerate(drop_mbti_df['Username'])}

# Encode groups category to index
# groups = set()
# for group_list in drop_mbti_df['Groups']:
#     for group in group_list:
#         groups.add(group)
group_code = {group: i for i, group in enumerate(drop_mbti_df['Groups'])}
def get_group_indices(groups):
    if isinstance(groups, list):
        return [group_code.get(group, None) for group in groups]
    elif isinstance(groups, str):
        return group_code.get(groups, None)
    else:
        return None

groups_index = drop_mbti_df['Groups'].apply(get_group_indices)
# groups_index = drop_mbti_df['Groups'].apply(lambda g: [group_code.get(group, None) for group in g])

# Constructing edges for graph
edges = []

# Add edges from groups and following networks
for _, row in drop_mbti_df.iterrows():
    user = row['Username']
    groups = row['Groups']
    for group in groups:
        edges.append((user_to_index[user], groups_index[group]))

# input edges index with features of following networks
df_follow_names_sorted = drop_mbti_df.sort_values(by='Username').reset_index(drop=True)
for _, row in df_follow_names_sorted.iterrows():
    follower = row['Username']
    followed = row['Follower']
    if follower in user_to_index and followed in user_to_index:
        edges.append((user_to_index[follower], user_to_index[followed]))

# Node feature creation
node_features = embeddings_df.loc[embeddings_df.index.intersection(drop_mbti_df.index)].values

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Build model
data = Data(edge_index=edge_index, x=torch.tensor(node_features, dtype=torch.float), y = y_follow_encode)

# print(data.x.shape[0])
# print(len(usernames_df))
# all=pd.read_json("dataset/all_description_data.json")
# print(len(all))
# print(len(embeddings_df))
print(data)

# True labels
# true_labels = data.y[data.test_mask].cpu().numpy()  

# Generate the confusion matrix
# cm = confusion_matrix(true_labels, predictions.cpu().numpy())

# # Plotting the confusion matrix
# plt.figure(figsize=(10, 10))
# sns.heatmap(cm, annot=True, fmt='g')
# plt.xlabel('Predicted labels')
# plt.ylabel('True labels')
# plt.title('Confusion Matrix')
# plt.show()

# print(combined_df)
# print(embeddings_df)
# print(embeddings_df.shape)
# print(reciprocity_df['Group'].head())
