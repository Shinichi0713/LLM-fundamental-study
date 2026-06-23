import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import roc_auc_score


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        adj_tilde = adj + torch.eye(adj.size(0))
        degree = torch.sum(adj_tilde, dim=1)
        D_inv_sqrt = torch.diag(torch.pow(degree, -0.5))
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
        adj_norm = torch.mm(torch.mm(D_inv_sqrt, adj_tilde), D_inv_sqrt)
        return torch.mm(adj_norm, torch.mm(x, self.weight))



class LinkPredictionGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim) # 最終ノードベクトルを出力

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        return self.gcn2(x, adj)

    def compute_link_score(self, z, edge_list):
        # 2つのノードベクトルの内積（類似度）を計算し、リンクの存在確率（0〜1）にする
        u_embed = z[edge_list[:, 0]]
        v_embed = z[edge_list[:, 1]]
        scores = torch.sum(u_embed * v_embed, dim=1)
        return torch.sigmoid(scores)


