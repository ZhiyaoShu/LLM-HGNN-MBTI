import numpy as np
import pandas as pd
from torch_geometric.utils import index_to_mask
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import pickle

def load_onehot_data():
    df = pd.read_csv('data/updated_merge_new_df.csv')
    
    return df

def preprocess_data(df):
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
    df.loc[:, 'Label'] = df['MBTI'].apply(
        encode_mbti_number)

    # Prepare class and label methods
    y_follow_label = torch.tensor(
        df.loc[:, 'Label'].values, dtype=torch.long).unsqueeze(1)

    return y_follow_label

def one_hot_features(df):
    df.fillna({'Gender': 'Unknown', 'Sexual': 'Unknown', 'About': '', 'Location': 'Unknown'}, inplace=True)
    
    # One-hot encode the categorical features
    encoder = OneHotEncoder(handle_unknown='ignore')
    one_hot_encoded = encoder.fit_transform(df[['Gender', 'Sexual', 'Location']]).toarray()
    
    # Use all generated feature names for columns
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out())
    
    tfidf_vectorizer = TfidfVectorizer(max_features=100)  # Limiting to the top 100 features
    about_tfidf = tfidf_vectorizer.fit_transform(df['About'].fillna('')).toarray()
    about_tfidf_df = pd.DataFrame(about_tfidf, columns=[f"tfidf_{i}" for i in range(about_tfidf.shape[1])])

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    columns_to_keep = ['Username'] + numeric_columns
    combined_df = pd.concat([df[columns_to_keep], one_hot_df], axis=1)
    
    return combined_df

def prepare_graph_tensors(combined_df, df):

    # Convert 'Follower' and 'Groups' columns from string to list
    df['Follower'] = df['Follower'].apply(ast.literal_eval)
    df['Groups'] = df['Groups'].apply(ast.literal_eval)

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
        groups = row['Groups']
        for group in groups:
            if group in user_to_index:
                edges.append((user_to_index[user], user_to_index[group]))
                edges.append((user_to_index[group], user_to_index[user]))

        # Add edges for followers
        followed = row['Follower']
        for follow in followed:
            if follow in user_to_index:
                edges.append((user_to_index[user], user_to_index[follow]))
                edges.append((user_to_index[follow], user_to_index[user]))

    # Converting edges list to a tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return node_features, edge_index, user_to_index

def generate_masks(y, split=(2, 1, 1)):
    total_size = y.shape[0]  # Ensure this reflects the total dataset size
    train_size, val_size, test_size = split

    # Generate indices for training, validation, and testing
    indices = torch.randperm(total_size)
    train_end = int(train_size / sum(split) * total_size)
    val_end = train_end + int(val_size / sum(split) * total_size)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Create masks based on indices
    train_mask = index_to_mask(train_indices, size=total_size)
    val_mask = index_to_mask(val_indices, size=total_size)
    test_mask = index_to_mask(test_indices, size=total_size)
    # train_mask = index_to_mask(torch.as_tensor(train_index, dtype=torch.long), size=y.shape[0])
    # val_mask = index_to_mask(torch.as_tensor(val_index, dtype=torch.long), size=y.shape[0])
    # test_mask = index_to_mask(torch.as_tensor(test_index, dtype=torch.long), size=y.shape[0])

    return train_mask, val_mask, test_mask

def process():
    df = load_onehot_data()
    y_follow_label = preprocess_data(df)
    combined_df = one_hot_features(df)
    node_features, edge_index, user_to_index = prepare_graph_tensors(combined_df, df)
    data = Data(x=node_features, edge_index=edge_index)
    y = y_follow_label

    data.y = y.float()
    train_mask, val_mask, test_mask = generate_masks(y)
    data.edge_index = edge_index
    data.node_features = node_features
    data.user_to_index = user_to_index
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.groups = df['Groups'].tolist()
        
    with open('baseline_data2.pkl', 'wb') as f:
        pickle.dump(data, f)
        
    return data


if __name__ == "__main__":
    process()