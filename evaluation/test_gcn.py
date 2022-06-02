import argparse
import math
from operator import mod
import sys
import numpy as np
import time
import torch
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, RedditDataset
from dgl.utils import expand_as_pair
from dgl.ops import edge_softmax
import torch.nn as nn
import dgl.function as fn
from dgl.nn import GATConv, GraphConv
from util import *

sys.path.append('../')
from cuda.util import graph_preprocess
from cuda.QLinear import QLinear, Qlinear_func, Qlinear_func_reduce
from cuda.SPMM import *
from cuda.SDDMM import *


class QGraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False):
        super(QGraphConv, self).__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise Exception('Invalid norm value. Must be either "none", "both", "right" or "left".'
                            ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

        if weight:
            self.weight = nn.Parameter(torch.Tensor(out_feats, in_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None, edge_weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise Exception('There are 0-in-degree nodes in the graph, '
                                    'output for those nodes will be invalid. '
                                    'This is harmful for some applications, '
                                    'causing silent performance regression. '
                                    'Adding self-loop on the input graph by '
                                    'calling `g = dgl.add_self_loop(g)` will resolve '
                                    'the issue. Setting ``allow_zero_in_degree`` '
                                    'to be `True` when constructing this module will '
                                    'suppress the check and let the code run.')
            aggregate_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                raise NotImplementedError

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise Exception('External weight is provided while at the same time the'
                                    ' module has defined its own weight parameter. Please'
                                    ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src, scale_feat = Qlinear_func_reduce.apply(feat_src, weight)
                    # feat_src = torch.matmul(feat_src, weight)
                # graph.srcdata['h'] = feat_src
                # graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                # rst = graph.dstdata['h']
                rst = spmm_copy_u.apply(graph, feat_src, scale_feat)
            else:
                # aggregate first then mult W
                # graph.srcdata['h'] = feat_src
                # graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                # rst = graph.dstdata['h']
                rst = spmm_copy_u.apply(graph, feat_src)
                if weight is not None:
                    rst = Qlinear_func.apply(rst, weight)
                    # rst = torch.matmul(rst, weight)

            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 quant):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        layer = QGraphConv if quant else GraphConv
        # input layer
        self.layers.append(layer(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                layer(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(layer(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):
    # load and preprocess dataset
    g = load_graph_train(args.dataset)
    graph_preprocess(g)
    cuda = True
    print(g)
    features = g.ndata['feat']
    # trim features to fit in k%4
    size = features.shape[1] - (features.shape[1] % 4)
    # features = features[:, :size]
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(g,
                in_feats,
                args.n_hidden,
                args.n_hidden,
                args.n_layers,
                F.relu,
                args.dropout,
                args.quant)
    print(model)
    if cuda:
        model.cuda()

    proj_layer = nn.Linear(args.n_hidden, g.n_classes).cuda()  # proj to predict

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    epsilon = 1 - math.log(2)
    # initialize graph
    dur = []
    f = open(args.filename, 'w+')
    for epoch in range(args.n_epochs):
        model.train()
        # forward
        start = torch.cuda.Event(enable_timing=True)
        forward = torch.cuda.Event(enable_timing=True)
        backward = torch.cuda.Event(enable_timing=True)
        start.record()
        logits = model(features)
        forward.record()
        logits = proj_layer(logits)

        def cross_entropy(x, labels):
            y = F.cross_entropy(x, labels[:, 0], reduction="none")
            y = torch.log(epsilon + y) - math.log(epsilon)
            return torch.mean(y)

        if args.dataset == 'ogbn-arxiv':
            loss = cross_entropy(logits[train_mask], labels[train_mask])
        else:
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        backward.record()
        loss.backward()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(forward) + forward.elapsed_time(backward)
        optimizer.step()

        if epoch >= 3:
            dur.append(elapsed)
        f.write(elapsed.__str__() + '\n')
        acc = evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))
    f.close()
    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="reddit",
                        help="Dataset")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=1000,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=256,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=4,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument('--quant', action="store_true", default=False,
                        help="use QGCN")
    parser.add_argument('--filename', type=str, default='time.txt')
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
