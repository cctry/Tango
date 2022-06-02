import argparse
from logging import exception
import math
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
from dgl.nn import GraphConv, GATConv
from util import *
sys.path.append('../')
from cuda.util import graph_preprocess
from cuda.QLinear import QLinear, Qlinear_func
from cuda.SPMM import *
from cuda.SDDMM import *

epsilon = 1 - math.log(2)


class HGT(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads, is_quant, dropout=0.2, use_norm=False) -> None:
        super().__init__()
        self.is_quant = is_quant
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.d_k = out_dim // n_heads
        self.n_heads = n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        linear = QLinear if is_quant else nn.Linear
        self.q_linear = linear(in_dim, out_dim)
        self.k_linear = linear(in_dim, out_dim)
        self.v_linear = linear(in_dim, out_dim)
        self.msg_linear = linear(out_dim, out_dim)
        self.attn_linear = linear(out_dim, out_dim)
        self.a_linear = linear(out_dim, out_dim)
        if use_norm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, g, h):
        with g.local_scope():
            k = self.k_linear(h)
            k = self.attn_linear(k).view(-1, self.n_heads, self.d_k)
            v = self.msg_linear(self.v_linear(h)).view(-1,
                                                       self.n_heads, self.d_k)
            q = self.q_linear(h).view(-1, self.n_heads, self.d_k)
            if not self.is_quant:
                t = dgl.ops.u_dot_v(g, k, q)
            else:
                t = sddmm_u_dot_v.apply(g, k, q)
            t = t.div_(self.sqrt_dk)
            attn_score = edge_softmax(g, t, norm_by='dst')
            if not self.is_quant:
                t = dgl.ops.u_mul_e_sum(g, v, attn_score)
            else:
                t = spmm_u_mul_e.apply(g, v, attn_score)
            t = t.view(-1, self.out_dim)
            t = self.a_linear(t)
            return t


def main(args):
    # load and preprocess dataset
    g = load_graph_train(args.dataset)

    features = g.ndata['feat']
    # trim features to fit in k%4
    size = features.shape[1] - (features.shape[1] % 4)
    features = features[:, :size].contiguous()
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    num_feats = features.shape[1]
    n_classes = g.n_classes

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    graph_preprocess(g)
    print(g)
    # create model
    model = HGT(
        num_feats,
        args.num_hidden,
        args.num_heads,
        args.quant)
    proj_layer = nn.Linear(
        args.num_hidden, n_classes).cuda()  # proj to predict
    print(model)
    model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # initialize graph
    elapsed = 0
    dur = []
    f = open(args.filename, 'w+')
    for epoch in range(args.epochs):
        model.train()
        # forward
        start = torch.cuda.Event(enable_timing=True)
        forward = torch.cuda.Event(enable_timing=True)
        backward = torch.cuda.Event(enable_timing=True)
        start.record()
        logits = model(g, features)
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
        print("Epoch {:05d} | Time(ms) {:.4f}".format(epoch, np.mean(dur)))
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HGT')
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv",
                        help="dataset")
    parser.add_argument("--num-hidden", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="number of attention heads")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="number of training epochs")
    parser.add_argument('--quant', action="store_true", default=False,
                        help="use QHGT")
    parser.add_argument('--filename', type=str, default='time.txt')
    args = parser.parse_args()
    main(args)
