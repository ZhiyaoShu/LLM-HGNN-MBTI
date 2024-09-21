import torch
import torch.nn as nn
import dhg
from dhg.nn import HGNNPConv
from torch.nn import Linear

# Reference from Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>_ paper (AAAI 2019).
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
        for i, layer in enumerate(self.layers):
            skip_X = X  
            X = layer(X, hg)  
            if i < len(self.skip_layers):  
                skip_X = self.skip_layers[i](skip_X)  
                X = X + skip_X  

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
