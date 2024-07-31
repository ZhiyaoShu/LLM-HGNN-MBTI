import torch
import torch.utils.data
import numpy as np
import pandas as pd
from torch_geometric.utils import index_to_mask
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import pickle
import parse

from personality_loader import y_enngram, y_mbti

args = parse.parse_arguments()

def fill_na_with_mean(df):
    for column in df.select_dtypes(include=[np.number]):
        df[column].fillna(df[column].mean(), inplace=True)
    return df

def load_data():
    # Load merged data
    df = pd.read_csv("data/user_data_cleaned.csv")
    # Load embeddings
    embeddings_df = pd.read_json("data/embeddings2.json")
    df = fill_na_with_mean(df)
    return df, embeddings_df

def one_hot_features(df, embeddings_df, use_llm):
    df.fillna(
        {"Gender": "Unknown", "Sexual": "Unknown", "Location": "Unknown"},
        inplace=True,
    )
    df = fill_na_with_mean(df)

    # One-hot encode the categorical features
    encoder = OneHotEncoder(handle_unknown="ignore")
    one_hot_encoded = encoder.fit_transform(
        df[["Gender", "Sexual", "Location"]]
    ).toarray()

    # Use all generated feature names for columns
    one_hot_df = pd.DataFrame(
        one_hot_encoded, columns=encoder.get_feature_names_out())

    tfidf_vectorizer = TfidfVectorizer(
        max_features=100
    )  # Limiting to the top 100 features
    about_tfidf = tfidf_vectorizer.fit_transform(
        df["About"].fillna("")).toarray()
    about_tfidf_df = pd.DataFrame(
        about_tfidf, columns=[
            f"tfidf_{i}" for i in range(about_tfidf.shape[1])]
    )

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    columns_to_keep = ["Username"] + numeric_columns

    if use_llm:
        combined_df = pd.concat(
            [df[columns_to_keep], one_hot_df, about_tfidf_df, embeddings_df], axis=1
        )
    else:
        combined_df = pd.concat(
            [df[columns_to_keep], one_hot_df, about_tfidf_df], axis=1
        )

    return combined_df


def prepare_graph_tensors(combined_df, df):
    df['Follower'] = df['Follower'].fillna('[]').apply(ast.literal_eval)
    if combined_df.isnull().any().any():
        combined_df.fillna(0, inplace=True)

    # Node Features
    node_features = torch.tensor(
        combined_df.iloc[:, 1:].values, dtype=torch.float)

    # User to Index mapping
    user_to_index = {username: i for i,
                     username in enumerate(combined_df["Username"])}

    # Constructing Edges
    edges = []

    for _, row in df.iterrows():
        user = row["Username"]

        # Add edges for followers
        followed = row["Follower"]
        for follow in followed:
            if follow in user_to_index:
                edges.append((user_to_index[user], user_to_index[follow]))
                edges.append((user_to_index[follow], user_to_index[user]))

    # Converting edges list to a tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return node_features, edge_index, user_to_index


# Create train, test, validate masks
def generate_masks(y, split=(2, 1, 1)):
    total_size = y.shape[0]
    train_size, val_size, test_size = split

    # Generate indices for training, validation, and testing
    indices = torch.randperm(total_size)
    train_end = int(train_size / sum(split) * total_size)
    adjust_train_end = int(train_end - train_end * 0.0)

    val_end = train_end + int(val_size / sum(split) * total_size)

    train_indices = indices[:adjust_train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Create masks based on indices
    train_mask = index_to_mask(train_indices, size=total_size)
    val_mask = index_to_mask(val_indices, size=total_size)
    test_mask = index_to_mask(test_indices, size=total_size)

    return train_mask, val_mask, test_mask


def process(args):
    df, embeddings_df = load_data()

    # Determine the type of personality labels to use
    if args.mbti:
        y = y_mbti(df)
    else:
        y = y_enngram(df)

    combined_df = one_hot_features(df, embeddings_df)

    node_features, edge_index, user_to_index = prepare_graph_tensors(
        combined_df, df)

    def check_for_nans(tensor, name="Tensor"):
        if torch.isnan(tensor).any():
            raise ValueError(f"{name} contains NaN values")
    check_for_nans(node_features, "Node Features")
    check_for_nans(edge_index, "Edge Index")

    data = Data(x=node_features, edge_index=edge_index)

    data.y = y.float()

    train_mask, val_mask, test_mask = generate_masks(
        y.squeeze()
    )

    data.edge_index = edge_index
    data.node_features = node_features
    data.user_to_index = user_to_index

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.groups = df["Groups"].tolist()

    with open('data_features.pkl', 'wb') as f:
        pickle.dump(data, f)
    return data


if __name__ == "__main__":
    process()
