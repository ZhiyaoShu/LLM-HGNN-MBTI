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

data_path = "data_features.pkl"
hyperedge_path = "cache/hyperedges.pkl"


def HGNFrame():
    if not os.path.exists(data_path):
        data_preparation()
    else:
        data = pickle.load(open("data_features.pkl", "rb"))
    df, _ = data_preparation.load_data()

    if not os.path.exists(hyperedge_path):
        data = get_dhg_hyperedges(data, df)
    else:
        data = pickle.load(open(hyperedge_path, "rb"))
    logging.debug(f"Input data type: {type(data)}")
    logging.debug(f"Attributes of data: {dir(data)}")

    data = normalize_features(data)
    node_features = data.x
    model = HGNN(
        in_channels=node_features.shape[1],
        hid_channels=300,
        num_classes=16,
        use_bn=True,
    )
    return model, data


def HGNPFrame():
    if not os.path.exists(data_path):
        logging.debug("Data file not found. Preparing data...")
        data_preparation()
    else:
        logging.debug("Data file found. Loading data...")
        data = pickle.load(open("data_features.pkl", "rb"))
    df, _ = data_preparation.load_data()
    if not os.path.exists(hyperedge_path):
        logging.debug("Hyperedge file not found. Preparing hyperedges...")
        data = get_dhg_hyperedges(data, df)
    else:
        logging.debug("Hyperedge file found. Loading hyperedges...")
        data = pickle.load(open(hyperedge_path, "rb"))
    logging.debug(f"Data type: {type(data)}")
    logging.debug(f"Attributes of data: {dir(data)}")
    data = normalize_features(data)
    node_features = data.x
    model = HGNNP(
        in_channels=node_features.shape[1],
        hid_channels=300,
        num_classes=16,
        use_bn=True,
    )
    return model, data
