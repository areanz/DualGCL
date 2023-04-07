import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import time
import csv
from load_data import read_graphfile
from graph_sampler import *
from util import AverageMeter
from model import Model
from evaluate_embedding import evaluate_embedding
from losses import GraphGraphContrastiveLoss, NodeGraphContrastiveLoss


def train(dataset, model, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [350, 450], gamma=0.1)
    ng_criterion = NodeGraphContrastiveLoss().to(args.device)
    gg_criterion = GraphGraphContrastiveLoss().to(args.device)
    iter = 0
    train_loss = []
    train_epochs = []
    best_loss = 1e9
    log_interval = 2
    best_acc = 0

    for epoch in range(args.epochs):
        start_time = time.time()
        total_time = 0
        model.train()
        grad_norm = AverageMeter()
        loss_avg = AverageMeter()
        print('Epoch: ', epoch)
        for batch_idx, data in enumerate(dataset):
            begin_time = time.time()
            optimizer.zero_grad()
            adj = data['adj'].float().to(args.device)
            h0 = data['feats'].float().to(args.device)
            node_x, assign_x, cls, avg = model(h0, adj)

            loss = gg_criterion(cls, avg)
            loss_ng = ng_criterion(assign_x, cls)
            loss = args.coef * loss + loss_ng
            loss.backward()
            g_norm = clip_grad_norm_(model.parameters(), args.grad_clip)
            grad_norm.update(g_norm)
            optimizer.step()
            iter += 1
            loss_avg.update(loss.item())
            elapsed = time.time() - begin_time
            total_time += elapsed

        if epoch % log_interval == 0 and epoch > 380:
            model.eval()
            graph_embedding, labels = model.get_embedding(dataset)
            res = evaluate_embedding(graph_embedding, labels)
       

        scheduler.step()
        train_loss.append(loss_avg.avg)
        train_epochs.append(epoch)
        end_time = time.time()
        print("loss: {}, epoch_time: {}".format(loss_avg.avg, end_time-start_time))



def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MUTAG')
    parser.add_argument('--feat', type=str, default='node-label', choices=['node-label', 'node-feat'],
                        help='node feature type')
    parser.add_argument('--depth', type=int, default=2, help="the depth of transformer encoder layer")
    parser.add_argument('--heads', type=int, default=2, help="the number of attention heads")
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--dim_feedforward', type=int, default=64, help='feedforward dimension of transformer encoder')
    parser.add_argument('--pool_ratio', type=float, default=0.5)
    parser.add_argument('--lr', dest='lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--feature', dest='feature_type', default="default",
                        help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--epochs', type=int, default=400, help='num-of-epoch')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0,
                        help='number of workers when dataloading')
    parser.add_argument('--bias', dest='bias', action='store_const',
                        const=True, default=True, help='switch for bias')
    parser.add_argument('--result_file', default='contrastive_results.csv', help='result csv file saving dir')
    parser.add_argument('--datadir', dest='datadir', default='data', help='Directory where benchmark is located')
    parser.add_argument('--cat', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--emb_dp', type=float, default=0.)
    parser.add_argument('--transformer_dp', type=float, default=0.)
    parser.add_argument('--num_gcn_layer', type=int, default=3)
    parser.add_argument('--coef', type=float, default=1.0)
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args


def main(args):
    # deterministic
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    graphs, num_graph_labels, max_num_nodes = read_graphfile(args.datadir, args.dataset, max_nodes=1000)
    example_node = graphs[0].nodes[0]
    if args.feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif args.feat == 'node-label' and 'label' in example_node:
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                util.node_dict(G)[u]['feat'] = np.array(util.node_dict(G)[u]['label'])
    else:
        args.feature_type = 'deg'

    dataset_sampler = GraphSampler(graphs, max_num_nodes, normalize=True, features=args.feature_type)
    avg_num_nodes = np.mean([G.number_of_nodes() for G in graphs])
    print('num_graphs:{}, max_node_num:{}, avg_node_num:{}'.format(len(graphs), max_num_nodes, avg_num_nodes))
    args.assign_dim = int(avg_num_nodes * args.pool_ratio)
    input_dim = dataset_sampler.feat_dim
    dataset_loader = torch.utils.data.DataLoader(dataset_sampler, batch_size=args.batch_size, shuffle=True,
                                                 drop_last=False, num_workers=args.num_workers, pin_memory=True)
    model = Model(args.num_gcn_layer, input_dim, args.hidden_dim, args.dim_feedforward, int(avg_num_nodes * args.pool_ratio),
                  num_graph_labels,
                  args.cat, args.depth, args.heads, args.emb_dp, args.transformer_dp).to(args.device)

    train(dataset_loader, model, args)


if __name__ == '__main__':
    args = arg_parse()
    main(args)
