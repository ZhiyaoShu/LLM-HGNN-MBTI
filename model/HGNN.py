import torch
import torch.nn as nn
import dhg
from dhg.nn import HGNNConv
import pickle

from model_config import normalize_features
from data_preparation import load_data
from Hyperedges import get_dhg_hyperedges

class HGNN(nn.Module):
    r"""The HGNN model proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.5.
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
        self.layers.append(
            HGNNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            HGNNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
            
        return X

def HGN():
    # data = pickle.load(open("baseline_data2.pkl", "rb"))
    # data = pickle.load(open("test_train_change1.pkl", "rb"))
    data = pickle.load(open("graph_with_embedding2.pkl", "rb"))
    # data = pickle.load(open("Enneagram_embedding.pkl", "rb"))
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
    visual = dhg.Hypergraph.draw(data.hg)
    return model, data, visual

if __name__ == "__main__":
    HGN()