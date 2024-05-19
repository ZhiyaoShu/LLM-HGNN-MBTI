import torch
import torch.nn as nn
import dhg
from dhg.nn import HGNNConv
import pickle
from dhg.nn import HGNNPConv
from torch.nn import Linear, LayerNorm, ReLU, Sequential
from util.model_config import normalize_features
from data_preparation import load_data
from Hyperedges import get_dhg_hyperedges, custom_hyperedges

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
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        self.layers.append(
            HGNNPConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )

        self.layers.append(
            HGNNPConv(hid_channels, hid_channels, use_bn=use_bn, is_last=True)
        )
        # add MLP to match the dimensions
        self.mlp = Linear(hid_channels, num_classes)
        
    def get_embedding(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        for layer in self.layers:
            X = layer(X, hg)
        
        return X

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
    data = pickle.load(open("graph_with_embedding.pkl", "rb"))
    df, _ = load_data()
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
