import torch
from torch import nn


class NodeGraphContrastiveLoss(nn.Module):
    def __init__(self):
        super(NodeGraphContrastiveLoss, self).__init__()

    def forward(self, l_enc, g_enc):
        T = 0.2
        num_assign = l_enc.shape[1]
        l_enc = l_enc.reshape(-1, l_enc.shape[-1])
        x_abs = l_enc.norm(dim=1)
        g_abs = g_enc.norm(dim=1)

        g_expand = g_enc.unsqueeze(1)
        g_expand = g_expand.expand(-1, num_assign, -1)
        g_expand = g_expand.reshape(-1, g_expand.shape[-1])
        g_expand_abs = g_expand.norm(dim=1)
        l_pos = torch.einsum('nc,nc->n', [l_enc, g_expand]) / torch.einsum('n,n->n', [x_abs, g_expand_abs])
        l_pos = torch.exp(l_pos / T)
        sim_matrix = torch.einsum('nc,kc->nk', [l_enc, g_enc]) / torch.einsum('i,j->ij', x_abs, g_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        loss = l_pos / (sim_matrix.sum(dim=1) - l_pos)
        loss = - torch.log(loss).mean()
        return loss


class GraphGraphContrastiveLoss(nn.Module):
    def __init__(self):
        super(GraphGraphContrastiveLoss, self).__init__()

    def forward(self, x, x_aug):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss