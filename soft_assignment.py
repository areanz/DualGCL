import torch
import torch.nn as nn
from GCNlayer import GraphConv, GINConv
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU


class GIN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_gc_layers=2):
        super(GIN, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_gc_layers - 1):
            if i:
                linear = Sequential(Linear(hid_dim, hid_dim), ReLU(), Linear(hid_dim, hid_dim))
            else:
                linear = Sequential(Linear(in_dim, hid_dim), ReLU(), Linear(hid_dim, hid_dim))
            conv = GINConv(linear)
            bn = nn.BatchNorm1d(hid_dim)
            self.convs.append(conv)
            self.bns.append(bn)

        linear = Sequential(Linear(hid_dim, hid_dim), ReLU(), Linear(hid_dim, out_dim))
        conv = GINConv(linear)
        bn = nn.BatchNorm1d(out_dim)
        self.convs.append(conv)
        self.bns.append(bn)

    def forward(self, x, adj):
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, adj))
            x = x.permute(0, 2, 1)
            x = self.bns[i](x)
            x = x.permute(0, 2, 1)

        return x

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_gc_layers=2):
        super().__init__()
        self.num_gc_layers = num_gc_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        if num_gc_layers == 1:
            self.convs.append(GraphConv(in_dim, out_dim, True, False))
            bn = nn.BatchNorm1d(out_dim)
            self.bns.append(bn)
        else:
            for i in range(num_gc_layers - 1):
                if i:
                    self.convs.append(GraphConv(hid_dim, hid_dim, True, False))
                else:
                    self.convs.append(GraphConv(in_dim, hid_dim, True, False))
                bn = nn.BatchNorm1d(hid_dim)
                self.bns.append(bn)

            self.convs.append(GraphConv(hid_dim, out_dim, True, False))
            bn = nn.BatchNorm1d(out_dim)
            self.bns.append(bn)

    def forward(self, x, adj):
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, adj))
            x = x.permute(0, 2, 1)
            x = self.bns[i](x)
            x = x.permute(0, 2, 1)

        return x


class SoftAssignment(nn.Module):
    def __init__(self, num_gcn_layer, in_dim, hid_dim, assign_dim, cat):
        super(SoftAssignment, self).__init__()
        self.emb_block = GCN(in_dim, hid_dim, hid_dim, num_gcn_layer)
        self.assign_block = GCN(in_dim, hid_dim, assign_dim, num_gcn_layer)

    def forward(self, x, adj):
        embedding_tensor = self.emb_block(x, adj)
        assign_h = self.assign_block(x, adj)
        assign_h = nn.Softmax(dim=-1)(assign_h)
        adj = torch.transpose(assign_h, 1, 2) @ adj @ assign_h
        assign_tensor = torch.matmul(torch.transpose(assign_h, 1, 2), embedding_tensor)
        x = embedding_tensor

        return x, assign_tensor, adj





