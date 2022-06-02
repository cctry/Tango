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
from dgl.data import  PubmedGraphDataset, RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset

from dgl.utils import expand_as_pair
from dgl.ops import edge_softmax
import torch.nn as nn
import dgl.function as fn
from dgl.nn import GraphConv, GATConv

sys.path.append('../')
from cuda.util import graph_preprocess
from cuda.QLinear import QLinear, Qlinear_func
from cuda.SPMM import *
from cuda.SDDMM import *


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

    def forward(self, h, g, neg_graph):
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
                h = dgl.ops.u_mul_e_sum(g, v, attn_score)
            else:
                h = spmm_u_mul_e.apply(g, v, attn_score)
            h = h.view(-1, self.out_dim)
            h = self.a_linear(h)
        h = h.unsqueeze_(1)
        assert h.dim() == 3, "the output of gat layer should be 3-dim but now is {}".format(h.size())
        if self.is_quant:
            g.edata['score'] = sddmm_u_dot_v.apply(g, h, h)
            neg_graph.edata['score'] = sddmm_u_dot_v.apply(neg_graph, h, h)
        else:
            g.ndata['h'] = h
            neg_graph.ndata['h'] = h
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
        


def loss_fcn(pos_graph, neg_graph):
    pos_score = pos_graph.edata['score']
    neg_score = neg_graph.edata['score']
    score = torch.cat([pos_score, neg_score])
    label = torch.cat([torch.ones_like(pos_score),
                       torch.zeros_like(neg_score)])
    loss = F.binary_cross_entropy_with_logits(score, label)
    return loss


def construct_negative_graph(graph, k):
    src, _ = graph.edges()
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(),
                            (len(src) * k,), device=src.device)
    return dgl.graph((neg_src, neg_dst)).to('cuda')


def main(args):
    # load and preprocess dataset
    if args.dataset == 'reddit':
        data = RedditDataset()
        g = data[0]
        n_edges = g.num_edges()
        ratio = 0.08
        idx = torch.randperm(n_edges)[:int(n_edges * ratio)]
        g = dgl.edge_subgraph(g, idx)
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
        g = data[0]
    elif args.dataset == "dblp":
        out = np.loadtxt('dataset/com-dblp.ungraph.txt',
                         skiprows=4, unpack=True, dtype=np.int64)
        g = dgl.graph((out[0], out[1]))
        g = dgl.to_bidirected(g)
        g.ndata['feat'] = torch.rand(g.num_nodes(), 128)
    elif args.dataset == "amazon":
        out = np.loadtxt('dataset/amazon0505.txt', skiprows=4,
                         unpack=True, dtype=np.int64)
        g = dgl.graph((out[0], out[1]))
        g = dgl.to_bidirected(g)
        g.ndata['feat'] = torch.rand(g.num_nodes(), 128)
    elif args.dataset == "ogbn-arxiv":
        data = DglNodePropPredDataset(name="ogbn-arxiv")
        g, _ = data[0]
        srcs, dsts = g.all_edges()
        g.add_edges(dsts, srcs)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    print(g)
    g = g.to('cuda')
    features = g.ndata['feat']
    # trim features to fit in k%4
    size = features.shape[1] - (features.shape[1] % 4)
    features = features[:, :size].contiguous()
    in_feats = features.shape[1]

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    graph_preprocess(g)

    # create model
    model = HGT(in_feats,
                args.num_hidden,
                args.num_heads,
                args.quant)
    print(model)
    model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    f = open(f"{args.filename}.csv", 'w')
    dur = []
    for epoch in range(args.epochs):
        model.train()
        neg_graph = construct_negative_graph(g, args.k)
        graph_preprocess(neg_graph)
        # forward
        start = torch.cuda.Event(enable_timing=True)
        forward = torch.cuda.Event(enable_timing=True)
        backward = torch.cuda.Event(enable_timing=True)
        start.record()
        model(features, g, neg_graph)
        forward.record()
        loss = loss_fcn(g, neg_graph)

        optimizer.zero_grad()
        loss.backward()
        backward.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(forward) + forward.elapsed_time(backward)
        optimizer.step()

        if epoch >= 3:
            dur.append(elapsed)
        f.write(f"{np.mean(dur)}\n")

        # print("Epoch {:05d} | Time(ms) {:.4f} | Loss {:.4f} | "
        #       "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
        #                                      n_edges / np.mean(dur) / 1000))

    print()
    f.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--epochs", type=int, default=1000,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    parser.add_argument('--quant', action="store_true", default=False,
                        help="use QGAT")
    parser.add_argument('--k', type=int, default=8,
                        help='number of negative samples')
    parser.add_argument("--filename", type=str, default="results.csv")
    args = parser.parse_args()
    print(args)

    main(args)
