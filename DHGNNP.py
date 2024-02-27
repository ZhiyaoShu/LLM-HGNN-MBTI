import torch
import torch.nn as nn
import dhg
import pickle
from dhg.nn import HGNNPConv
from torch.nn import Linear, LayerNorm, ReLU, Sequential
from utils import normalize_features
from data_preparation import load_data
from Hypergraph import get_dhg_hyperedges
from torch_geometric.nn import DeepGCNLayer
import pickle


class DeepHGNNP(nn.Module):
    r"""
    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
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
        super(DeepHGNNP, self).__init__()
        self.layers = nn.ModuleList()

        # Initial hypergraph convolution layer
        self.layers.append(
            HGNNPConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            DeepGCNLayer(
                HGNNPConv(
                    hid_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate
                ),
                block="res+",
                dropout=drop_rate,
                ckpt_grad=False,
            )
        )

        self.layers.append(
            HGNNPConv(hid_channels, num_classes, use_bn=use_bn, drop_rate=drop_rate)
        )

        # # add a linear layer to match the dimensions
        # if self.use_skip_connections and in_channels != num_classes:
        #     self.skip_layers.append(nn.Linear(hid_channels, num_classes))

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return X

def save_hyperedges(hypergraph, filename='hypergraph.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(hypergraph, f)

def save_model(model, filename='model.pth'):
    torch.save(model.state_dict(), filename)
        
def DHGNNP():
    # data = pickle.load(open("edges_delete_file.pkl", "rb"))
    # data = get_dhg_hyperedges(data)
    # data = normalize_features(data)
    model = DeepHGNNP(
        in_channels=384,
        hid_channels=384,
        num_classes=17,
        use_bn=True,
    )
    
    # save_hyperedges(data.hg, 'hypergraph.pkl')
    # save_model(model, 'deep_hgnnp_model.pth')
    
    return model


if __name__ == "__main__":
    DHGNNP()
