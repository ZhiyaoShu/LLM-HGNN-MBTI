from models.hypergraph.hgnn import HGNN
from models.hypergraph.hgnnp import HGNNP
from models.hypergraph.hyperedges import get_dhg_hyperedges
from models.network_utils import normalize_features
from dataloader.data_preparation import load_data
import pickle


def HGNFrame():
    data = pickle.load(open("hgm.pkl", "rb"))
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


def HGNPFrame():
    data = pickle.load(open("hgm.pkl", "rb"))
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
