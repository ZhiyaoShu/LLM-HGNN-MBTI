import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from graph.GCN import GCN
from graph.GAT import GAT
from graph.G_transformer import GCNCT
from hypergraph.networks import HGN, HGNP
from dataloader.data_preparation import load_data

# Feature normalization
def check_data_distribution(data):
    """
    Check the distribution of the features in the data.
    """
    from scipy.stats import shapiro

    sample = data.x[np.random.choice(
        data.x.size(0), 1000, replace=False)].numpy()

    stat, p = shapiro(sample)
    return "normal" if p > 0.05 else "non-normal"


def normalize_features(data):
    """
    Normalize the node features based on their distribution.
    """
    distribution = check_data_distribution(data)
    scaler = StandardScaler() if distribution == "normal" else MinMaxScaler()
    data.x = torch.tensor(scaler.fit_transform(
        data.x.numpy()), dtype=torch.float)
    return data


def get_models(model_type):
    """
    Get the model based on the model name.
    """
    if model_type in ['hgnn', 'hgnnp']:
        if model_type == 'hgnn':
            model, data = HGN()
        else:
            model, data = HGNP()
        out = model(data.node_features, data.hg)
        out_logits = out[data.train_mask]
        logits_shifted = out_logits - \
            out_logits.max(dim=1, keepdim=True).values
        return out, out_logits, logits_shifted
    else:
        if model_type == 'gcn':
            model = GCN()
        elif model_type == 'gat':
            model = GAT()
        elif model_type == 'gcnct':
            model = GCNCT()
        data = load_data()
        out = model(data)
        return out
