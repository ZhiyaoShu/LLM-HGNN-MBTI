import pickle
import csv
import torch
import random

# Load the pickle file
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_edges_to_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def delete_random_edges(data):
    total_edges = data.edge_index.size(1)
    print(f"Total edges: {total_edges}")
    
    num_edges_to_keep = total_edges - int(total_edges * 0.2)
    # Shuffle indices
    indices = list(range(total_edges))
    # Select indices to keep
    indices_to_keep = random.sample(indices, num_edges_to_keep)
    # Select edges based on these indices
    remaining_edges = data.edge_index[:, indices_to_keep]
    print(f"Remaining edges: {remaining_edges.shape}")
    
    data.edge_index = remaining_edges
    
    return data


if __name__ == "__main__":
    data = load_pickle('baseline_data2.pkl')
    
    data = delete_random_edges(data)

    # Save the modified edge data back to the pickle file
    save_edges_to_pickle(data, 'baseline_delete_edge_file.pkl')