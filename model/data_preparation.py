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

def fill_na_with_mean(df):
    for column in df.select_dtypes(include=[np.number]):
        df[column].fillna(df[column].mean(), inplace=True)
    return df

def load_data():
    # Load merged data
    df = pd.read_csv("data/user_data_cleaned.csv")
    # df_mbti = pd.read_csv("data/updated_merge_new_df_2.csv")
    # Load embeddings
    embeddings_df = pd.read_json("data/embeddings2.json")
    df = fill_na_with_mean(df)
    return df, embeddings_df

def preprocess_data(df_mbti):
    # Encoding MBTI to binary and numerical representation
    # def encode_mbti(mbti):
    #     encoding = {
    #         'I': '0', 'E': '1',
    #         'N': '0', 'S': '1',
    #         'T': '0', 'F': '1',
    #         'J': '0', 'P': '1',
    #     }
    #     default_encoding = [-1, -1, -1, -1]
    #     # Check if mbti is unknown or NaN
    #     if mbti in ['Unknown', 'nan', None] or pd.isna(mbti):
    #         return default_encoding
    #     encoded = [int(encoding.get(char, -1)) for char in mbti]

    #     return encoded if len(encoded) == 4 else default_encoding

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
    df_mbti.loc[:, "Label"] = df_mbti["MBTI"].apply(encode_mbti_number)
    # Prepare class and label methods
    y_follow_label = torch.tensor(
        df_mbti.loc[:, "Label"].values, dtype=torch.long
    ).unsqueeze(1)
    # Get the enneagram types from existed dataset
    # enneagram = df["EnneagramType"].unique()
    # print(f"Enneagram types: {enneagram}")
    # def enneagramType(enneagram):
    #     enneagram_to_number = {
    #         "Type 1": 0,
    #         "Type 2": 1,
    #         "Type 3": 2,
    #         "Type 4": 3,
    #         "Type 5": 4,
    #         "Type 6": 5,
    #         "Type 7": 6,
    #         "Type 8": 7,
    #         "Type 9": 8,
    #     }
    #     if enneagram in ["Unknown", "nan", None] or pd.isna(enneagram):
    #         return 9
    #     enneagram = enneagram.split("w")[0].strip()
    #     return enneagram_to_number.get(enneagram)

    # df.loc[:, "Label"] = df["EnneagramType"].apply(enneagramType)
    # print(df["Label"])
    # y_follow_label = torch.tensor(
    #     df.loc[:, "Label"].values, dtype=torch.long
    # ).unsqueeze(1)
    return y_follow_label


def one_hot_features(df, embeddings_df):
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
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out())

    tfidf_vectorizer = TfidfVectorizer(
        max_features=100
    )  # Limiting to the top 100 features
    about_tfidf = tfidf_vectorizer.fit_transform(df["About"].fillna("")).toarray()
    about_tfidf_df = pd.DataFrame(
        about_tfidf, columns=[f"tfidf_{i}" for i in range(about_tfidf.shape[1])]
    )

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    columns_to_keep = ["Username"] + numeric_columns

    combined_df = pd.concat(
        [df[columns_to_keep], one_hot_df, about_tfidf_df, embeddings_df], axis=1
    )

    return combined_df


def prepare_graph_tensors(combined_df, df):
    # Convert 'Follower' and 'Groups' columns from string to list
    df['Follower'] = df['Follower'].fillna('[]').apply(ast.literal_eval)
    if combined_df.isnull().any().any():
        combined_df.fillna(0, inplace=True)

    # Node Features
    node_features = torch.tensor(combined_df.iloc[:, 1:].values, dtype=torch.float)

    # User to Index mapping
    user_to_index = {username: i for i, username in enumerate(combined_df["Username"])}

    # Constructing Edges
    edges = []

    for _, row in df.iterrows():
        user = row["Username"]
        # groups = row['Groups']

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
    total_size = y.shape[0]  # Ensure this reflects the total dataset size
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


def process():
    df, embeddings_df, df_mbti = load_data()
    # Unpack the tuple returned by preprocess_data
    y_follow_label = preprocess_data(df_mbti)

    combined_df = one_hot_features(df, embeddings_df)

    node_features, edge_index, user_to_index = prepare_graph_tensors(combined_df, df)
    def check_for_nans(tensor, name="Tensor"):
        if torch.isnan(tensor).any():
            raise ValueError(f"{name} contains NaN values")
    check_for_nans(node_features, "Node Features")
    check_for_nans(edge_index, "Edge Index")
    
    data = Data(x=node_features, edge_index=edge_index)
    
    # data.y = y.float()
    # data.y = torch.argmax(y, dim=1)

    # Correctly use y_follow_label tensor, ensuring it's a float tensor
    data.y = y_follow_label.float()


    train_mask, val_mask, test_mask = generate_masks(
        y_follow_label.squeeze()
    )  # Remove unnecessary dimension

    data.edge_index = edge_index
    data.node_features = node_features
    data.user_to_index = user_to_index

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.groups = df["Groups"].tolist()
    # print(len(data))
    print("Total number of classes:", len(data.y.unique()))
    # print("node features:", node_features.shape)
    # print("edge index:", edge_index)
    # print("y_follow_label:", y_follow_label)
    # print(node_features.shape)
    # print(edge_index.shape)
    print(f"Train mask number: {torch.sum(data.train_mask)}")
    print(f"Val mask: {torch.sum(data.val_mask)}")
    print(f"Test mask: {torch.sum(data.test_mask)}")

    with open('graph_with_embedding2.pkl', 'wb') as f:
    # with open ('test_train_change1.pkl', 'wb') as f:
    # with open("Enneagram_embedding.pkl", "wb") as f:
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
