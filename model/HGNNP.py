import torch
import torch.nn as nn
import dhg
from dhg.nn import HGNNConv
import pickle
from dhg.nn import HGNNPConv
from torch.nn import Linear, LayerNorm, ReLU, Sequential
from utils.model_config import normalize_features
from model.data_preparation import load_data
from model.Hyperedges import get_dhg_hyperedges, custom_hyperedges

# from torch_geometric.nn import DeepGCNLayer


class HGNNP(nn.Module):
    r"""The HGNN :sup:`+` model proposed in `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        # use_skip_connections=False,
    ) -> None:
        super().__init__()
        # self.use_skip_connections = use_skip_connections
        self.layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        self.layers.append(
            HGNNPConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        # # add a linear layer to match the dimensions
        # if self.use_skip_connections:
        #     self.skip_layers.append(nn.Linear(in_channels, hid_channels))

        self.layers.append(
            HGNNPConv(hid_channels, hid_channels, use_bn=use_bn, is_last=True)
        )
        # add MLP to match the dimensions
        self.mlp = Linear(hid_channels, num_classes)
        
    def get_embedding(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        for layer in self.layers:
            X = layer(X, hg)
        
        return X
    
        # # add a linear layer to match the dimensions
        # if self.use_skip_connections and in_channels != num_classes:
        #     self.skip_layers.append(nn.Linear(hid_channels, num_classes))

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        X = self.get_embedding(X, hg)
        X = self.mlp(X)
        return X


def HGNP():
    # data = pickle.load(open("baseline_data2.pkl", "rb"))
    # data = pickle.load(open("graph_with_embedding2.pkl", "rb"))
    data = pickle.load(open("Enneagram_embedding.pkl", "rb"))
    df, _ = load_data()
    # data = custom_hyperedges(data, df)
    data = get_dhg_hyperedges(data, df)
    data = normalize_features(data)
    node_features = data.x
    model = HGNNP(
        in_channels=node_features.shape[1],
        hid_channels=300,
        num_classes=17,
        use_bn=True,
    )

    return model, data


if __name__ == "__main__":
    HGNP()
