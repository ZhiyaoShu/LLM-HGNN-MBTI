import pickle
import csv
import torch
import random

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def delete_train_data(data):
    data = load_pickle('test_train_change8.pkl')
    train_mask = data.train_mask
    rest_train = data.x[train_mask] - data.x[train_mask]* 0.1
    data.train_mask = rest_train
    print(f"Remaining train data: {torch.sum(data.train_mask)}")
    return rest_train

if __name__ == "__main__":
    data = load_pickle('test_train_change8.pkl')
    data = delete_train_data(data)
