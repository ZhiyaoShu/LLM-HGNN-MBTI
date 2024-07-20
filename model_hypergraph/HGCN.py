import torch
import torch.nn as nn

import dhg
from dhg.nn import HGNNConv
import pickle
from dhg.nn import HGNNPConv
from torch.nn import Linear, LayerNorm, ReLU, Sequential
from model_hypergraph.utils import normalize_features
from dataloader.data_preparation import load_data
from Hyperedges import get_dhg_hyperedges, custom_hyperedges
from torch_geometric.nn import DeepGCNLayer
from dhg.nn import HyperGCNConv
from dhg.structure.graphs import Graph

class HyperGCN(nn.Module):
    r"""The HyperGCN model proposed in `HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs <https://papers.nips.cc/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf>`_ paper (NeurIPS 2019).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_mediator`` (``str``): Whether to use mediator to transform the hyperedges to edges in the graph. Defaults to ``False``.
        ``fast`` (``bool``): If set to ``True``, the transformed graph structure will be computed once from the input hypergraph and vertex features, and cached for future use. Defaults to ``True``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.5.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_mediator: bool = False,
        use_bn: bool = False,
        fast: bool = True,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.fast = fast
        self.cached_g = None
        self.with_mediator = use_mediator
        self.layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        self.layers.append(
            HyperGCNConv(
                in_channels,
                hid_channels,
                use_mediator,
                use_bn=use_bn,
                drop_rate=drop_rate,
            )
        )
        self.layers.append(
            HyperGCNConv(
                hid_channels, num_classes, use_mediator, use_bn=use_bn, is_last=True
            )
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        if self.fast:
            if self.cached_g is None:
                self.cached_g = Graph.from_hypergraph_hypergcn(
                    hg, X, self.with_mediator
                )
            for layer in self.layers:
                X = layer(X, hg, self.cached_g)
        else:
            for layer in self.layers:
                X = layer(X, hg)
                
def HGCN():
    data = pickle.load(open("graph_with_embedding.pkl", "rb"))
    df, _ = load_data()
    data = get_dhg_hyperedges(data, df)
    data = normalize_features(data)
    node_features = data.x
    model = HyperGCN(
        in_channels=node_features.shape[1],
        hid_channels=300,
        num_classes=16,
        use_bn=True,
    )

    return model, data

if __name__ == "__main__":
    HGCN()