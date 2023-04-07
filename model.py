import torch
import torch.nn as nn
import numpy as np
from soft_assignment import SoftAssignment
from transformer import *
from einops import rearrange, repeat

class Model(nn.Module):
    def __init__(self, num_gcn_layer, in_dim, hid_dim, dim_feedforward, assign_dim, num_classes, cat, depth, heads,
                emb_dp, transformer_dp):
        super(Model, self).__init__()
        self.soft_assignment = SoftAssignment(num_gcn_layer, in_dim, hid_dim, assign_dim, cat)
        self.pos_embedding = nn.Parameter(torch.randn(1, assign_dim + 1, hid_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hid_dim))
        self.dropout = nn.Dropout(emb_dp)
        self.to_cls_token = nn.Identity()
        self.encoder_layers = TransformerEncoderLayer(hid_dim, heads, dim_feedforward, transformer_dp)
        self.encoder = TransformerEncoder(self.encoder_layers, depth)
        self.decoder = nn.Sequential(
            nn.LayerNorm(hid_dim),
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(emb_dp),
            nn.Linear(hid_dim, hid_dim)
        )

    def forward(self, x, adj):
        embedding_tensor, assign_tensor, new_adj = self.soft_assignment(x, adj)
        x = assign_tensor
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)
        x = self.encoder(x, new_adj)
        average_pooling = torch.mean(x[:, 1:], dim=1)
        x = self.to_cls_token(x[:, 0])

        return embedding_tensor, assign_tensor, self.decoder(x), average_pooling

    def get_embedding(self, dataloader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            output = []
            labels = []
            for data in dataloader:
                adj = data['adj'].float().to(device)
                h0 = data['feats'].float().to(device)
                label = data['label'].long()
                labels.append(label)
                _, _, cls, _ = self.forward(h0, adj)
                output.append(cls.detach().cpu().numpy())
        output = np.concatenate(output, 0)
        labels = np.concatenate(labels, 0)
        return output, labels






