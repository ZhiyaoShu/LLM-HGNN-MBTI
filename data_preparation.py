import torch
import torch.utils.data
import numpy as np
import pandas as pd
from torch_geometric.utils import index_to_mask
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import ast
import pickle
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


def load_data():
    # Load merged data
    df = pd.read_csv('updated_merge_new_df.csv')

    # Load embeddings
    embeddings_df = pd.read_json("embeddings3.json")

    return df, embeddings_df


def preprocess_data(df, embeddings_df):
    # Encoding MBTI to binary and numerical representation
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

    def encode_mbti_number(mbti):
        mbti_to_number = {
            'INTJ': 1, 'ENTJ': 2, 'INTP': 3, 'ENTP': 4,
            'INFJ': 5, 'ENFJ': 6, 'INFP': 7, 'ENFP': 8,
            'ISTJ': 9, 'ESTJ': 10, 'ISFJ': 11, 'ESFJ': 12,
            'ISTP': 13, 'ESTP': 14, 'ISFP': 15, 'ESFP': 16
        }
        if mbti in ['Unknown', 'nan', None] or pd.isna(mbti):
            return 0
        return mbti_to_number.get(mbti, 0)

    # Apply encoding
    df.loc[:, 'Encoded'] = df['MBTI'].apply(encode_mbti)
    df.loc[:, 'Label'] = df['MBTI'].apply(
        encode_mbti_number)

    # Prepare class and label methods
    y_follow_encode = torch.tensor(
        df.loc[:, 'Encoded'].tolist(), dtype=torch.long)

    y_follow_label = torch.tensor(
        df.loc[:, 'Label'].values, dtype=torch.long).unsqueeze(1)

    # Prepare user index and embeddings
    usernames_df = df[['Username']].reset_index(drop=True)
    embeddings_df = embeddings_df.reset_index(drop=True)
    combined_df = pd.concat([usernames_df, embeddings_df], axis=1)

    return combined_df, y_follow_encode, y_follow_label


def save_edges_as_pickle(edge_index, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(edge_index, f)

    print(f"Edges saved to {file_name}.")


def prepare_graph_tensors(combined_df, df):

    # Convert 'Follower' and 'Groups' columns from string to list
    df['Follower'] = df['Follower'].apply(ast.literal_eval)
    # df['Groups'] = df['Groups'].apply(ast.literal_eval)

    # Node Features
    node_features = torch.tensor(
        combined_df.iloc[:, 1:].values, dtype=torch.float)

    # User to Index mapping
    user_to_index = {username: i for i,
                     username in enumerate(combined_df['Username'])}

    # Constructing Edges
    edges = []

    for _, row in df.iterrows():
        user = row['Username']
        # groups = row['Groups']

        # Add edges for followers
        followed = row['Follower']
        for follow in followed:
            if follow in user_to_index:
                edges.append((user_to_index[user], user_to_index[follow]))
                edges.append((user_to_index[follow], user_to_index[user]))

    # Converting edges list to a tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return node_features, edge_index, user_to_index

# Create train, test, validate masks
def generate_masks(y, split=(2, 1, 1)):
    sp1 = split[2] * 1.0 / (split[0] + split[1] + split[2])
    sp2 = split[1] * 1.0 / (split[0] + split[1])

    id_list = np.array(range(y.shape[0]))
    [train_val, test_index] = train_test_split(
        id_list, test_size=sp1, shuffle=True)
    [train_index, val_index] = train_test_split(
        train_val, test_size=sp2, shuffle=False)

    train_mask = index_to_mask(torch.as_tensor(train_index), size=y.shape[0])
    val_mask = index_to_mask(torch.as_tensor(val_index), size=y.shape[0])
    test_mask = index_to_mask(torch.as_tensor(test_index), size=y.shape[0])

    return train_mask, val_mask, test_mask


def process():

    df, embeddings_df = load_data()
    combined_df, y_follow_encode, y_follow_label = preprocess_data(
        df, embeddings_df)

    # 'Label' column is used as labels (y) for the model
    # y = y_follow_encode
    y = y_follow_label

    # mask = torch.all(y_follow_encode == torch.tensor([-1, -1, -1, -1]), dim=1)

    node_features, edge_index, user_to_index = prepare_graph_tensors(
        combined_df, df)
    
    data = Data(x=node_features, edge_index=edge_index)
    
    data.y = y.float()
    # data.y = torch.argmax(y, dim=1)

    # Generate masks
    train_mask, val_mask, test_mask = generate_masks(y)
    
    # # Convert node_features to NumPy for efficiency
    # node_features_np = data.x.detach().numpy()
    
    data.edge_index = edge_index
    data.node_features = node_features
    data.user_to_index = user_to_index
    
    # Add to the data object
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.groups = df['Groups'].tolist()
    # print(len(data))

    print("node features:", node_features.shape)
    print("edge index:", edge_index)
    print(node_features.shape)
    print(edge_index.shape)

    with open('graph_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    return data


if __name__ == "__main__":
    process()

 # def check_and_clean_data(data):
    # # Check and clean x (features)
    #     if torch.isnan(data.x).any() or torch.isinf(data.x).any():
    #         # Replace NaNs with the mean of the column (or consider removing them)
    #         mean_values = torch.nanmean(data.x, dim=0)
    #         data.x = torch.nan_to_num(data.x, nan=mean_values)

    # # Check and clean y (targets)
    # if torch.isnan(data.y).any() or torch.isinf(data.y).any():
    #     # It's usually better to remove rows with NaN in targets
    #     valid_indices = ~torch.isnan(data.y) & ~torch.isinf(data.y)
    #     data.x = data.x[valid_indices]
    #     data.y = data.y[valid_indices]
    # data = check_and_clean_data(data)

    # # Range of K to try
    # k_values = range(12, 49)
    # silhouette_scores = []

    # for k in k_values:
    #     kmeans = KMeans(n_clusters=k, random_state=42).fit(node_features_np)
    #     score = silhouette_score(node_features_np, kmeans.labels_)
    #     silhouette_scores.append(score)

    # # Find the optimal K
    # optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
    # print(f"Optimal number of clusters: {optimal_k}, Silhouette Score={max(silhouette_scores)}")