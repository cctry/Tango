import sys
import time
import torch
import dgl
sys.path.append('../')
from cuda.kernels import *
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import PubmedGraphDataset
from dgl.data import AMDataset
from cuda.util import *
from dgl import function as fn


def test(g, N, E):
    N_, scaleN = quantize(N)
    scaleE = cuda_kernels.get_scale(E)
    E_T_ = transpose_quant(E, scaleE)
    scale = scaleN * scaleE
    return multi_cusparse_SPMM(g, N_, E_T_, scale)


dataset = DglNodePropPredDataset(name='ogbn-arxiv')
g, _ = dataset[0]
g = dgl.to_bidirected(g)
g = g.remove_self_loop().add_self_loop()
g = g.to('cuda')
graph_preprocess(g)


feat, a, rst, rst_ours = torch.load("rst_ref")
# print(rst)
g.srcdata['ft'] = feat
g.edata['a'] = a

print(dgl.ops.u_mul_e_sum(g, g.srcdata['ft'], g.edata['a'])[0])
# g.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "test"))
# print(g.dstdata["test"])
print(test(g, g.srcdata['ft'], g.edata['a'])[0]) # the first node has big error


# SPMM, maunally aggregate nodes
inward_nodes = g.predecessors(0)
inward_edges = g.edge_ids(inward_nodes, 0)
edge_features = a[inward_edges]
node_features = feat[inward_nodes]
res = (edge_features * node_features).view(len(inward_nodes), -1).sum(0)
print(res.view(4,256))
# SPMM with quantized a and feat
feat_, scaleN = quantize(feat)
a_, scaleE = quantize(a)
edge_features_quant = a_[inward_edges].to(torch.float32)
node_features_quant = feat_[inward_nodes].to(torch.float32)
res = (edge_features_quant * node_features_quant *
       scaleN * scaleE).view(len(inward_nodes), -1).sum(0)
print(res.view(4,256))


# test(g, g.srcdata['ft'], g.edata['a'])
# print(rst_ours)
# print(g.srcdata['ft'].shape)
# print(g.edata['a'].shape)
