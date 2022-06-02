import argparse
from logging import exception
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


class QGATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(QGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = QLinear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = QLinear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = QLinear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(
                torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = QLinear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if self.res_fc is not None:
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise Exception(
                        'There are 0-in-degree nodes in the graph.')

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                h_src = h_src.contiguous()
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (
                        graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            # graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            
            graph.edata['e'] = sddmm_u_add_v.apply(
                graph, graph.srcdata["el"], graph.dstdata["er"])

            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            # graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
            #                  fn.sum('m', 'ft'))
            # rst = graph.dstdata['ft']
            rst = spmm_u_mul_e.apply(
                graph, graph.srcdata["ft"], graph.edata["a"])
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(
                    *dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)
            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 quant):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.quant = quant
        layer = QGATConv if quant else GATConv
        if num_layers > 1:
            # input projection (no residual)
            self.gat_layers.append(layer(
                in_dim, num_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, False, self.activation))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(layer(
                    num_hidden * heads[l - 1], num_hidden, heads[l],
                    feat_drop, attn_drop, negative_slope, residual, self.activation))
            # output projection
            self.gat_layers.append(layer(
                num_hidden * heads[-2], num_hidden, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))
        else:
            self.gat_layers.append(layer(
                in_dim, num_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs, pos_graph, neg_graph):
        h = inputs
        for l in range(self.num_layers - 1):
            h = self.gat_layers[l](pos_graph, h).flatten(1)
        # output projection
        h = self.gat_layers[-1](pos_graph, h)
        assert h.dim() == 3, "the output of gat layer should be 3-dim but now is {}".format(h.size())
        if self.quant:
            pos_graph.edata['score'] = sddmm_u_dot_v.apply(pos_graph, h, h)
            neg_graph.edata['score'] = sddmm_u_dot_v.apply(neg_graph, h, h)
        else:
            pos_graph.ndata['h'] = h
            neg_graph.ndata['h'] = h
            pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
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
    features = features[:, :size]
    in_feats = features.shape[1]

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    graph_preprocess(g)

    # create model
    heads = ([args.num_heads] * (args.num_layers - 1)) + [args.num_out_heads]
    model = GAT(args.num_layers,
                in_feats,
                args.num_hidden,
                args.num_hidden,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual,
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
