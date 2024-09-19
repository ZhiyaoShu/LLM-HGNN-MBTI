import numpy as np
import pandas as pd
from torch_geometric.utils import index_to_mask
import torch
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import pickle
import logging

def load_onehot_data():
    df = pd.read_csv('dataset/updated_merge_new_df.csv', encoding='utf-8')

    return df

def preprocess_data(df):
    def encode_mbti_number(mbti):
        mbti_to_number = {
            "INTJ": 0,
            "ENTJ": 1,
            "INTP": 2,
            "ENTP": 3,
            "INFJ": 4,
            "INFP": 5,
            "ENFJ": 6,
            "ENFP": 7,
            "ISTJ": 8,
            "ESTJ": 9,
            "ISFJ": 10,
            "ESFJ": 11,
            "ISTP": 12,
            "ESTP": 13,
            "ISFP": 14,
            "ESFP": 15,
        }
        return mbti_to_number[mbti]
    # Apply encoding
    df.loc[:, 'Label'] = df['MBTI'].apply(
        encode_mbti_number)

    # Prepare class and label methods
    y_follow_label = torch.tensor(
        df.loc[:, 'Label'].values, dtype=torch.long).unsqueeze(1)

    return y_follow_label

def one_hot_features(df):
    df.drop(columns=['MBTI'], inplace=True)
    df.fillna({'Gender': 'Unknown', 'Sexual': 'Unknown', 'Location': 'Unknown', 'Followers':'Unknown'}, inplace=True)

    # One-hot encode the categorical features
    encoder = OneHotEncoder(handle_unknown='ignore')
    one_hot_encoded = encoder.fit_transform(df[['Gender', 'Sexual', 'Location']]).toarray()
    
    # Use all generated feature names for columns
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out()).reset_index(drop=True)
    
    tfidf_vectorizer = TfidfVectorizer(max_features=100)  # Limiting to the top 100 features
    about_tfidf = tfidf_vectorizer.fit_transform(df['About'].fillna('')).toarray()
    about_tfidf_df = pd.DataFrame(about_tfidf, columns=[f"tfidf_{i}" for i in range(about_tfidf.shape[1])]).reset_index(drop=True)

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    columns_to_keep = ['Username'] + numeric_columns
    combined_df = pd.concat([df[columns_to_keep].reset_index(drop=True), one_hot_df, about_tfidf_df], axis=1)
    
    return combined_df

def prepare_graph_tensors(combined_df, df):
    df['Follower'].fillna('[]', inplace=True)
    df['Groups'].fillna('[]', inplace=True)

    # Convert 'Follower' and 'Groups' columns from string to list safely
    def safe_literal_eval(s):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return [] 

    # Convert 'Follower' and 'Groups' columns from string to list
    df['Follower'] = df['Follower'].apply(safe_literal_eval)
    df['Groups'] = df['Groups'].apply(safe_literal_eval)

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
    total_size = y.shape[0] 
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
    
    with open('baseline_data.pkl', 'wb') as f:
        pickle.dump(data, f)
        
    return data


if __name__ == "__main__":
    process()