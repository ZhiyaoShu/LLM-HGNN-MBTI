from models.graph.GCN import GCN
from models.graph.GAT import GAT
from models.graph.G_transformer import GCNCT
from models.hypergraph.networks import HGNFrame, HGNPFrame
from dataloader.data_preparation import load_data

def get_models(model_type):
    """
    Get the model based on the model name.
    """
    if model_type in ["hgnn", "hgnnp"]:
        if model_type == "hgnn":
            model, data = HGNFrame()
        else:
            model, data = HGNPFrame()
        out = model(data.node_features, data.hg)
        out_logits = out[data.train_mask]
        logits_shifted = out_logits - out_logits.max(dim=1, keepdim=True).values
        return out, out_logits, logits_shifted
    else:
        if model_type == "gcn":
            model = GCN()
        elif model_type == "gat":
            model = GAT()
        elif model_type == "gcnct":
            model = GCNCT()
        data = load_data()
        out = model(data)
        return out
