from hgnn import HGNN
from hgnnp import HGNNP
from hyperedges import get_dhg_hyperedges
from network_utils import normalize_features
from dataloader.data_preparation import load_data
import pickle

def HGN(model, data):
    data = pickle.load(open("graph_with_embedding.pkl", "rb"))
    df, _ = load_data()
    data = get_dhg_hyperedges(data, df)
    data = normalize_features(data)
    node_features = data.x
    model = HGNN(
        in_channels=node_features.shape[1],
        hid_channels=300,
        num_classes=16,
        use_bn=True,
    )
    return model, data


def HGNP(model, data):
    data = pickle.load(open("graph_with_embedding.pkl", "rb"))
    df, _ = load_data()
    data = get_dhg_hyperedges(data, df)
    data = normalize_features(data)
    node_features = data.x
    model = HGNNP(
        in_channels=node_features.shape[1],
        hid_channels=300,
        num_classes=16,
        use_bn=True,
    )
    return model, data
