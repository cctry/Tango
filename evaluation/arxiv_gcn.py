import argparse
import math
import time
import sys
from turtle import forward

import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import torch.nn as nn
from dgl import function as fn
import dgl.nn.pytorch as dglnn
import dgl
sys.path.append('../')
from cuda.util import graph_preprocess
from cuda.QLinear import QLinear, Qlinear_func
from cuda.SPMM import *


class QGraphConv(dglnn.GraphConv):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False):
        super(QGraphConv, self).__init__(in_feats, out_feats,
                                         norm, weight, bias, activation, allow_zero_in_degree)
        if weight:
            self.weight = nn.Parameter(th.Tensor(out_feats, in_feats))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def forward(self, graph, feat, weight=None, edge_weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise Exception('0-in-degree nodes found.')
            aggregate_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                raise Exception('edge_weight is not supported.')

            feat_src, feat_dst = dgl.utils.expand_as_pair(feat, graph)
            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
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
                    # feat_src = Qlinear_func.apply(feat_src, weight)
                    feat_src = F.linear(feat_src, weight)

                # graph.srcdata['h'] = feat_src
                # graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                # rst = graph.dstdata['h']
                rst = spmm_copy_u.apply(graph, feat_src)
            else:
                # aggregate first then mult W
                # graph.srcdata['h'] = feat_src
                # graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                # rst = graph.dstdata['h']
                rst = spmm_copy_u.apply(graph, feat_src)
                if weight is not None:
                    # rst = Qlinear_func.apply(rst, weight)
                    rst = F.linear(rst, weight)

            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, use_linear):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.use_linear = use_linear

        self.convs = nn.ModuleList()
        if use_linear:
            self.linear = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            bias = i == n_layers - 1

            self.convs.append(dglnn.GraphConv(
                in_hidden, out_hidden, "both", bias=bias))
            if use_linear:
                self.linear.append(
                    nn.Linear(in_hidden, out_hidden, bias=False))
            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_hidden))

        self.input_drop = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)

            if self.use_linear:
                linear = self.linear[i](h)
                h = conv + linear
            else:
                h = conv

            if i < self.n_layers - 1:
                h = self.norms[i](h)
                h = self.activation(h)
                h = self.dropout(h)

        return h



class QGCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, use_linear):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.use_linear = use_linear

        self.convs = nn.ModuleList()
        if use_linear:
            self.linear = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            bias = i == n_layers - 1

            self.convs.append(QGraphConv(
                in_hidden, out_hidden, "both", bias=bias))
            if use_linear:
                self.linear.append(
                    QLinear(in_hidden, out_hidden, bias=False))
            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_hidden))

        self.input_drop = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)

            if self.use_linear:
                linear = self.linear[i](h)
                h = conv + linear
            else:
                h = conv

            if i < self.n_layers - 1:
                h = self.norms[i](h)
                h = self.activation(h)
                h = self.dropout(h)

        return h



device = None
in_feats, n_classes = None, None
epsilon = 1 - math.log(2)


def gen_model(args):
    if args.use_labels:
        model = QGCN(
            in_feats + n_classes, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.use_linear
        )
    else:
        model = QGCN(in_feats, args.n_hidden, n_classes,
                    args.n_layers, F.relu, args.dropout, args.use_linear)
    return model


def cross_entropy(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = th.log(epsilon + y) - math.log(epsilon)
    return th.mean(y)


def compute_acc(pred, labels, evaluator):
    return evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]


def add_labels(feat, labels, idx):
    onehot = th.zeros([feat.shape[0], n_classes]).to(device)
    onehot[idx, labels[idx, 0]] = 1
    return th.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train(model, graph, labels, train_idx, optimizer, use_labels):
    global elapsed
    model.train()

    feat = graph.ndata["feat"]

    if use_labels:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_pred_idx = train_idx[mask]

    optimizer.zero_grad()
    start = torch.cuda.Event(enable_timing=True)
    forward = torch.cuda.Event(enable_timing=True)
    backward = torch.cuda.Event(enable_timing=True)
    start.record()
    pred = model(graph, feat)
    forward.record()
    loss = cross_entropy(pred[train_pred_idx], labels[train_pred_idx])
    loss.backward()
    backward.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(forward) + forward.elapsed_time(backward)
    optimizer.step()

    return loss, pred


@th.no_grad()
def evaluate(model, graph, labels, train_idx, val_idx, test_idx, use_labels, evaluator):
    model.eval()

    feat = graph.ndata["feat"]

    if use_labels:
        feat = add_labels(feat, labels, train_idx)

    pred = model(graph, feat)
    train_loss = cross_entropy(pred[train_idx], labels[train_idx])
    val_loss = cross_entropy(pred[val_idx], labels[val_idx])
    test_loss = cross_entropy(pred[test_idx], labels[test_idx])

    return (
        compute_acc(pred[train_idx], labels[train_idx], evaluator),
        compute_acc(pred[val_idx], labels[val_idx], evaluator),
        compute_acc(pred[test_idx], labels[test_idx], evaluator),
        train_loss,
        val_loss,
        test_loss,
    )


elapsed = 0
def run(args, model, graph, labels, train_idx, val_idx, test_idx, evaluator):
    global elapsed
    f = open("GCN.csv", "w+")
    f.write(f"epoch,acc,train_acc,val_acc,test_acc,train_loss,val_loss,test_loss\n")
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100, verbose=True, min_lr=1e-3
    )

    # training loop
    total_time = 0
    best_val_acc, final_test_acc, best_val_loss = 0, 0, float("inf")

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    for epoch in range(1, args.n_epochs + 1):

        adjust_learning_rate(optimizer, args.lr, epoch)
        loss, pred = train(model, graph, labels, train_idx,
                           optimizer, args.use_labels)

        lr_scheduler.step(loss)
        total_time += elapsed

        acc = compute_acc(pred[train_idx], labels[train_idx], evaluator)

        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss = evaluate(
            model, graph, labels, train_idx, val_idx, test_idx, args.use_labels, evaluator
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            final_test_acc = test_acc
        f.write(f"{epoch},{acc},{train_acc},{val_acc},{test_acc},{train_loss},{val_loss},{test_loss}\n")
        if args.log_every >= 0 and epoch % args.log_every == 0:
            print(
                f"Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}\n"
                f"Loss: {loss.item():.4f}, Acc: {acc:.4f}\n"
                f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                f"Train/Val/Test/Best val/Final test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{final_test_acc:.4f}"
            )

            for l, e in zip(
                [accs, train_accs, val_accs, test_accs, losses,
                    train_losses, val_losses, test_losses],
                [acc, train_acc, val_acc, test_acc, loss,
                    train_loss, val_loss, test_loss],
            ):
                l.append(e)

    print("*" * 50)
    print(f"Best val acc: {best_val_acc}, Final test acc: {final_test_acc}")
    print("*" * 50)
    f.close()
    return best_val_acc, final_test_acc


def main():
    global device, in_feats, n_classes

    argparser = argparse.ArgumentParser(
        "GCN on OGBN-Arxiv", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--n-epochs", type=int,
                           default=1000, help="number of epochs")
    argparser.add_argument(
        "--use-labels", action="store_true", help="Use labels in the training set as input features."
    ) # needs to use
    argparser.add_argument(
        "--use-linear", action="store_true", help="Use linear layer.") # needs to use
    argparser.add_argument(
        "--lr", type=float, default=0.005, help="learning rate")
    argparser.add_argument("--n-layers", type=int,
                           default=3, help="number of layers")
    argparser.add_argument("--n-hidden", type=int,
                           default=256, help="number of hidden units")
    argparser.add_argument("--dropout", type=float,
                           default=0.5, help="dropout rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument("--log-every", type=int,
                           default=50, help="log every LOG_EVERY epochs")
    args = argparser.parse_args()

    device = th.device("cuda:%d" % args.gpu)

    # load data
    data = DglNodePropPredDataset(name="ogbn-arxiv")
    evaluator = Evaluator(name="ogbn-arxiv")

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]

    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)

    # add self-loop
    graph = graph.remove_self_loop().add_self_loop()

    in_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()
    graph.create_formats_()

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    labels = labels.to(device)
    graph = graph.to(device)
    graph_preprocess(graph)
    # define model and optimizer
    model = gen_model(args)
    model = model.to(device)

    # run
    val_accs = []
    test_accs = []

    val_acc, test_acc = run(args, model, graph, labels,
                            train_idx, val_idx, test_idx, evaluator)
    print("Val Acc:", val_acc)
    print("Test Acc:", test_acc)


if __name__ == "__main__":
    main()
