import torch
import torch.nn as nn
import dhg
import pickle
from dhg.nn import HGNNPConv
from torch.nn import Linear, LayerNorm, ReLU, Sequential
from utils import normalize_features
from data_preparation import load_data
from Hypergraph import get_dhg_hyperedges, custom_hyperedges
from archieve.DHNNs import DeeperHNN
from torch_geometric.nn import DeepGCNLayer

# from torch_geometric.nn import DeepGCNLayer
import pickle
import torch.nn.functional as F
# device = torch.device("cuda" if torch.cuda else "cpu")


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
        out_channels: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        # use_skip_connections=False,
    ) -> None:
        super(DeepHGNNP, self).__init__()
        
        self.node_encoder = Linear(in_channels, hid_channels)
        self.layers = nn.ModuleList()
        # Initial hypergraph convolution layer
        for i in range(1, out_channels + 1):
            conv = HGNNPConv(
                hid_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate
            )

            self.norm = LayerNorm(hid_channels, elementwise_affine=True)

            self.act = ReLU(inplace=True)

            deepgcn_layer = DeepGCNLayer(conv=conv, act=self.act, block="res+", norm=self.norm, dropout=drop_rate, ckpt_grad=i % 3)
            
            self.layers.append(deepgcn_layer)

        self.lin = Linear(hid_channels, out_channels)

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        # print(f"Initial shape: {X.shape}")
        X = self.node_encoder(X)
        # print(f"After node_encoder: {X.shape}")
        for layer in self.layers:
            X = layer(X, hg)
            # print(f"After layer: {X.shape}")
        return F.log_softmax(self.lin(X), dim=1)


# def save_hyperedges(hypergraph, filename="hypergraph.pkl"):
#     with open(filename, "wb") as f:
#         pickle.dump(hypergraph, f)


# def save_model(model, filename="model.pth"):
#     torch.save(model.state_dict(), filename)


def DHGNNP():
    data = pickle.load(open("graph_with_embedding2.pkl", "rb"))
    # data = pickle.load(open('baseline_delete_edge_file.pkl', 'rb'))
    df, _ = load_data()
    data = get_dhg_hyperedges(data, df)
    data = normalize_features(data)

    model = DeepHGNNP(
        in_channels=data.x.shape[1],
        hid_channels=300,
        out_channels=17,
        use_bn=True,
    )

    return model, data


if __name__ == "__main__":
    DHGNNP()
