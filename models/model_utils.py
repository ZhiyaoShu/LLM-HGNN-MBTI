from models.graph.GCN import GCN
from models.graph.GAT import GAT
from models.graph.G_transformer import GTRANS
from models.hypergraph.networks import HGNFrame, HGNPFrame
from dataloader.data_preparation import load_data
import pickle
import os
import parse_arg
import logging

args = parse_arg.parse_arguments()
model_type = args.model


def get_models(model_type):
    """
    Get the model based on the model name.
    """
    cache_file = f"cache/{model_type}_model_data.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            model, data = pickle.load(f)
        logging.debug(f"Loaded {model_type} model and data from cache.")
        return model, data
    else:
        # Generate the model and data
        if model_type in ["hgnn", "hgnnp"]:
            if model_type == "hgnn":
                model, data = HGNFrame()
            else:
                model, data = HGNPFrame()

            # Save the model and data to cache
            with open(cache_file, "wb") as f:
                pickle.dump((model, data), f)
            print(f"Saved {model_type} model and data to cache.")
            return model, data
        else:
            if model_type == "gcn":
                model, data = GCN()
            elif model_type == "gat":
                model, data = GAT()
            elif model_type == "gtrans":
                model, data = GTRANS()
            logging.debug(f"Model shape: {model}")
            # Save the model and data to cache
            with open(cache_file, "wb") as f:
                pickle.dump((model, data), f)
            logging.debug(f"Saved {model_type} model and data to cache.")
            return model, data
