import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
      x = self.conv1(x, edge_index)
      x = F.relu(x)
      x = F.dropout(x, training=self.training)
      x = self.conv2(x, edge_index)
      return F.log_softmax(x, dim=1)

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# Cora データセットの読み込み
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)