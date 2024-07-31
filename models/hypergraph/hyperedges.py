import os
import pickle
import torch
import dhg
import ast

# Define the self-loop removal function
def remove_self_loops(edge_index: torch.Tensor) -> torch.Tensor:
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]


device = torch.device("cuda" if torch.cuda else "cpu")


def get_dhg_hyperedges(data, df, cache_dir="cache", hyperedges_file="hyperedges.pkl"):
    print("Checking for non-finite values in feature matrix...")
    if not torch.isfinite(data.x).all():
        print("Non-finite values found in data.x, applying fill strategy...")
        data.x[~torch.isfinite(data.x)] = 0

    # Define the hyperedges based on data
    edge_index = data.edge_index
    edge_index_no_self_loops = remove_self_loops(edge_index)

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
        for idx, group in enumerate(
            set(
                sum(
                    [
                        ast.literal_eval(row["Groups"])
                        for _, row in df.iterrows()
                        if row["Groups"]
                    ],
                    [],
                )
            )
        )
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

    # Ensure the cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Save the hyperedges to a file
    hyperedges_path = os.path.join(cache_dir, hyperedges_file)
    with open(hyperedges_path, "wb") as f:
        pickle.dump(hg, f)

    print(f"Hyperedges saved to {hyperedges_path}")

    return data

