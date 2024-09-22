from models.hypergraph.hgnn import HGNN
from models.hypergraph.hgnnp import HGNNP
from models.hypergraph.hyperedges import get_dhg_hyperedges
from models.network_utils import normalize_features
from dataloader import data_preparation
import pickle
import os
import logging
import parse_arg

args = parse_arg.parse_arguments()

if args.use_llm:
    data_path = "data_features.pkl"
else:
    data_path = "baseline_data.pkl"

hyperedge_path = "hyperedges.pkl"

class DataObject:
    def __init__(self, data_dict):
        self.__dict__.update(data_dict)

def HGNFrame():
    if not os.path.exists(data_path):
        logging.info("Data file not found. Preparing data...")
        data_preparation()
    else:
        logging.info("Data file found. Loading data...")
        data = pickle.load(open(data_path, "rb"))
    df, _ = data_preparation.load_data()
    train_mask = data.train_mask
    print(f"Train mask before hyperedge processing: {train_mask}")

    if not os.path.exists(hyperedge_path):
        logging.info(f"Hyperedges not found. Preparing hyperedges...")
        data = get_dhg_hyperedges(data, df)
        train_mask = data.train_mask
        print(f"Train mask: {train_mask}")
    else:
        logging.info(f"Loading hyperedges from {hyperedge_path}")
        data = pickle.load(open(hyperedge_path, "rb"))
    data = DataObject(data)
    print(f"Data object properties: {list(vars(data).keys())}")

    data = normalize_features(data)
    logging.info(f"Node features shape: {data.x.shape}")
    model = HGNN(
        in_channels=data.x.shape[1],
        hid_channels=300,
        num_classes=16,
        use_bn=True,
    )
    return model, data


def HGNPFrame():
    if not os.path.exists(data_path):
        logging.info("Data file not found. Preparing data...")
        data_preparation()
    else:
        logging.info("Data file found. Loading data...")
        data = pickle.load(open("data_features.pkl", "rb"))
    df, _ = data_preparation.load_data()

    if not os.path.exists(hyperedge_path):
        logging.info(f"Hyperedges not found. Preparing hyperedges...")
        data = get_dhg_hyperedges(data, df)
    else:
        logging.info(f"Loading hyperedges from {hyperedge_path}")
        data = HyperedgeData.load_hyperedges(hyperedge_path)
    data = normalize_features(data)
    node_features = data.x
    model = HGNNP(
        in_channels=node_features.shape[1],
        hid_channels=300,
        num_classes=16,
        use_bn=True,
    )
    return model, data
