from models.hypergraph.hgnn import HGNN
from models.hypergraph.hgnnp import HGNNP
from models.hypergraph.hyperedges import HyperedgeData, get_hyperedges
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


def HGNFrame():
    if not os.path.exists(data_path):
        logging.info("Data file not found. Preparing data...")
        data_preparation()
    else:
        logging.info("Data file found. Loading data...")
        data = pickle.load(open(data_path, "rb"))
    df, _ = data_preparation.load_data()

    if not os.path.exists(hyperedge_path):
        logging.info(f"Hyperedges not found. Preparing hyperedges...")
        data = get_hyperedges(data, df)
        data = HyperedgeData(data, df)
        data.save_hyperedges(hyperedge_path)
    else:
        logging.info(f"Loading hyperedges from {hyperedge_path}")
        data = HyperedgeData.load_hyperedges(hyperedge_path)
    logging.info(f"Input data type: {type(data)}")
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
        logging.info("Data file not found. Preparing data...")
        data_preparation()
    else:
        logging.info("Data file found. Loading data...")
        data = pickle.load(open("data_features.pkl", "rb"))
    df, _ = data_preparation.load_data()

    if not os.path.exists(hyperedge_path):
        logging.info(f"Hyperedges not found. Preparing hyperedges...")
        data = get_hyperedges(data, df)
        data = HyperedgeData(data, df)
        data.save_hyperedges(hyperedge_path)
    else:
        logging.info(f"Loading hyperedges from {hyperedge_path}")
        data = HyperedgeData.load_hyperedges(hyperedge_path)
    logging.info(f"Input data type: {type(data)}")
    data = normalize_features(data)
    node_features = data.x
    model = HGNNP(
        in_channels=node_features.shape[1],
        hid_channels=300,
        num_classes=16,
        use_bn=True,
    )
    return model, data
