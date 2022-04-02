import argparse
from logging import exception
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
                 g,
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
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
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

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers - 1):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


def main(args):
    # load and preprocess dataset
    if args.dataset == 'reddit':
        data = RedditDataset()
        g = data[0].to('cuda')
        n_classes = data.num_labels
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
        g = data[0]
        # g = dgl.to_bidirected(g)
        g = g.to('cuda')
        n_classes = data.num_labels
    elif args.dataset == "dblp":
        out = np.loadtxt('dataset/com-dblp.ungraph.txt', skiprows=4, unpack=True, dtype=np.int64)
        g = dgl.graph((out[0], out[1]))
        g = dgl.to_bidirected(g)
        g.ndata['feat'] = torch.rand(g.num_nodes(), 128)
        n_classes = 32
        g.ndata['label'] = torch.randint(n_classes, (g.num_nodes(),))
        shuffle = np.random.permutation(g.num_nodes())
        train_idx = shuffle[:int(g.num_nodes() * 0.8)]
        val_idx = shuffle[int(g.num_nodes() * 0.8):]
        test_idx = shuffle[int(g.num_nodes() * 0.8):]
        train_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
        train_mask[train_idx] = True  
        val_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
        val_mask[val_idx] = True
        test_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
        test_mask[test_idx] = True
        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask
        g = g.to('cuda')
    elif args.dataset == "amazon":
        out = np.loadtxt('dataset/amazon0505.txt', skiprows=4, unpack=True, dtype=np.int64)
        g = dgl.graph((out[0], out[1]))
        g = dgl.to_bidirected(g)
        g.ndata['feat'] = torch.rand(g.num_nodes(), 128)
        n_classes = 32
        g.ndata['label'] = torch.randint(n_classes, (g.num_nodes(),))
        shuffle = np.random.permutation(g.num_nodes())
        train_idx = shuffle[:int(g.num_nodes() * 0.8)]
        val_idx = shuffle[int(g.num_nodes() * 0.8):]
        test_idx = shuffle[int(g.num_nodes() * 0.8):]
        train_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
        train_mask[train_idx] = True  
        val_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
        val_mask[val_idx] = True
        test_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
        test_mask[test_idx] = True
        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask
        g = g.to('cuda')
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    cuda = True

    features = g.ndata['feat']
    # trim features to fit in k%4
    size = features.shape[1] - (features.shape[1] % 4)
    features = features[:, :size]
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    num_feats = features.shape[1]

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    graph_preprocess(g)
    print(g)
    # create model
    heads = ([args.num_heads] * (args.num_layers - 1)) + [args.num_out_heads]
    model = GAT(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual,
                args.quant)
    proj_layer = nn.Linear(
        args.num_hidden, n_classes).cuda()  # proj to predict
    print(model)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    elapsed = 0
    dur = []
    for epoch in range(args.epochs):
        model.train()
        # forward
        start = torch.cuda.Event(enable_timing=True)
        forward = torch.cuda.Event(enable_timing=True)
        backward = torch.cuda.Event(enable_timing=True)
        start.record()
        logits = model(features)
        forward.record()
        logits = proj_layer(logits)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        backward.record()
        loss.backward()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(forward) + forward.elapsed_time(backward)
        optimizer.step()

        if epoch >= 3:
            dur.append(elapsed)

        train_acc = accuracy(logits[train_mask], labels[train_mask])

        if args.fastmode:
            val_acc = accuracy(logits[val_mask], labels[val_mask])
        else:
            val_acc = evaluate(model, features, labels, val_mask)

        print("Epoch {:05d} | Time(ms) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--epochs", type=int, default=200,
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
    args = parser.parse_args()
    print(args)

    main(args)
