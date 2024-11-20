from functools import reduce
from operator import mul
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, DenseGCNConv, GATv2Conv, GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.typing import Adj, OptTensor
from typing import List

class GraphWaveNet(nn.Module):
    r"""GraphWaveNet implementation from the paper `"Graph WaveNet for Deep Spatial-Temporal
    Graph Modeling" <https://arxiv.org/abs/1906.00121>`.

    Args:
        num_nodes (int): Number of nodes in the input graph
        in_channels (int): Size of each hidden sample.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        out_timesteps (int): the number of time steps we are making predictions for
        dilations (list(int), optional):
        adaptive_embeddings (int, optional):
        dropout (float, optional): Dropout probability. (default: :obj:`0.3`)
        residual_channels (int, optional): number of residual channels
        dilation_channels (int, optional): number of dilation channels
        skip_channels (int, optional): size of skip layers
        end_channels (int, optional): the size of the final linear layers
    """
    def __init__(self,
                 dynamic_channels: int,
                 static_channels: int,
                 out_channels: int,
                 out_timesteps: int,
                 dilations: List[int]=[1, 2, 1, 2, 1, 2, 1, 2],
                 dropout: float=0.3,
                 residual_channels: int=32,
                 dilation_channels: int=32,
                 skip_channels: int=256,
                 end_channels: int=512):

        super(GraphWaveNet, self).__init__()

        self.total_dilation = sum(dilations)
        self.dilations = dilations
        self.dropout = dropout
        self.out_timesteps = out_timesteps
        self.out_channels = out_channels
        self.residual_channels = residual_channels
        self.input = nn.Conv2d(in_channels=dynamic_channels,
                               out_channels=residual_channels,
                               kernel_size=(1, 1))
        self.static_emb = nn.Sequential(
                nn.Linear(static_channels, residual_channels*2),
                nn.Linear(residual_channels*2, residual_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )

        self.tcn_a = nn.ModuleList()
        self.tcn_b = nn.ModuleList()
        self.gat = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.skip = nn.ModuleList()
        self.ma = nn.MultiheadAttention(embed_dim=residual_channels, num_heads=residual_channels // 4, dropout=0.4, batch_first=True)

        for d in dilations:
            self.tcn_a.append(
                nn.Conv2d(in_channels=residual_channels,
                          out_channels=dilation_channels,
                          kernel_size=(1, 2),
                          dilation=d))

            self.tcn_b.append(
                nn.Conv2d(in_channels=residual_channels,
                          out_channels=dilation_channels,
                          kernel_size=(1, 2),
                          dilation=d))

            self.skip.append(
                nn.Conv2d(in_channels=residual_channels,
                          out_channels=skip_channels,
                          kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(residual_channels))
            self.gat.append(
                GATConv(in_channels=dilation_channels,
                out_channels=residual_channels,
                heads=4,
                concat=False,
                add_self_loops=True,
                dropout=self.dropout))

        self.end1 = nn.Conv2d(in_channels=skip_channels,
                              out_channels=end_channels,
                              kernel_size=(1, 1))
        self.end2 = nn.Conv2d(in_channels=end_channels,
                              out_channels=out_channels * out_timesteps,
                              kernel_size=(1, 1))

    def forward(self,
        dynamic: Tensor,
        static: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        batch_size = dynamic.size()[:-3]
        input_timesteps = dynamic.size(-3)

        dynamic = dynamic.transpose(-1, -3)
        if self.total_dilation + 1 > input_timesteps:
            dynamic = F.pad(dynamic, (self.total_dilation - input_timesteps + 1, 0))
        # print(dynamic.shape)
        x = self.input(dynamic)
        static = self.static_emb(static)

        skip_out = None
        # print(x.shape, static.shape)
        # torch.Size([256, 16, 8, 21]) torch.Size([8, 16])
        static = static.T
        static = static.unsqueeze(0)
        static = static.repeat(x.size(0), 1, 1)
        static = static.unsqueeze(-1)
        static = static.repeat(1, 1, 1, x.size(-1))
        # print(static.shape)
        # torch.Size([256, 16, 8, 21])
        # s = static.repeat(x.size(0), x.size(1), )
        # print(static.shape)
        x = x + static



        for k, dil in enumerate(self.dilations):
            residual = x

            g1 = torch.tanh(self.tcn_a[k](x))
            g2 = torch.sigmoid(self.tcn_b[k](x))
            g = g1 * g2

            skip_cur = self.skip[k](g)
            skip_out = skip_cur if skip_out is None else skip_out[..., -skip_cur.size(-1):] + skip_cur

            g = g.transpose(-1, -3)
            timesteps = g.size(-3)
            g = g.reshape(reduce(mul, g.size()[:-2]), *g.size()[-2:])

            data = self.batch_timesteps(g, edge_index, edge_weight).to(g.device)
            # print(data.x.shape)
            g_x = self.gat[k](data.x, data.edge_index, data.edge_attr)
            g_x = g_x.reshape(*batch_size, timesteps, -1, self.gat[k].out_channels)

            x = F.dropout(g_x, p=self.dropout)
            x = x.transpose(-3, -1)
            x = x + residual[..., -x.size(-1):]
            x = self.bn[k](x)

        skip_out = skip_out[..., -1:]

        x = torch.relu(skip_out)
        x = torch.relu(self.end1(x))
        x = self.end2(x)
        x = x.reshape(*batch_size, self.out_timesteps, self.out_channels, -1).transpose(-1, -2)

        return x

    def batch_timesteps(
        self, 
        x,
        edge_index, 
        edge_weight=None
    ) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
        """
        edge_index = edge_index.expand(x.size(0), *edge_index.shape)

        if edge_weight != None:
            edge_weight = edge_weight.expand(x.size(0), *edge_weight.shape)

        dataset = [
            Data(x=_x, edge_index=e_i, edge_attr=e_w)
            for _x, e_i, e_w in zip(x, edge_index, edge_weight)
        ]
        loader = DataLoader(dataset=dataset, batch_size=x.size(0))

        return next(iter(loader))

"""
example usage:

model = GraphWaveNet(dynamic_channels = 17, static_channels=5, out_channels=1, out_timesteps=1)
BATCH_SIZE = 64
DC = 17
SC = 5
TIMESTEP = 21
NN = 150
dynamic = torch.randn(BATCH_SIZE, TIMESTEP, NN, DC)
static = torch.randn(BATCH_SIZE, NN, SC)
edge_index = torch.randint(0, NN, (2, NN*2))
edge_attr = torch.randn(NN*2, 3)
out = model(dynamic, static, edge_index, edge_attr)

"""

# if __name__ == "__main__":
#     model = GraphWaveNet(dynamic_channels = 17, static_channels=5, out_channels=1, out_timesteps=1)
#     BATCH_SIZE = 64
#     DC = 17
#     SC = 5
#     TIMESTEP = 21
#     NN = 150 # number of nodes
#     dynamic = torch.randn(BATCH_SIZE, TIMESTEP, NN, DC)
#     static = torch.randn(BATCH_SIZE, NN, SC)
#     edge_index = torch.randint(0, NN, (2, NN*2))
#     edge_attr = torch.randn(NN*2, 3)
#     out = model(dynamic, static, edge_index, edge_attr)
#     print(out.shape)
#     print(out)

