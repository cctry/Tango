import sys
import time
import torch
import torch.nn as nn
from torch.autograd.function import Function, once_differentiable
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import PubmedGraphDataset, RedditDataset
from dgl.data import AMDataset
sys.path.append('../')
from cuda.util import *
from cuda.QLinear import QLinear, Qlinear_func, QgemmNT_func
from cuda.kernels import *
from cuda.SPMM import *
import os

def _need_reduce_last_dim(ufeat, efeat):
    """Indicates whether to reduce the last dimension on edges
    in the backward pass of spmm,
    if so, use dot instead of mul."""
    if ufeat is None or efeat is None:
        return False
    ushp = ufeat.shape
    eshp = efeat.shape
    return ushp[1:-1] == eshp[1:-1] and eshp[-1] == 1 and ushp[-1] > 1

class spmm_u_mul_e_nocache(Function):
    @staticmethod
    def forward(ctx, g, N, E):
        N_, scaleN = quantize(N)
        scaleE = cuda_kernels.get_scale(E)  
        E_T_ = transpose_quant(E, scaleE)
        scale = scaleN * scaleE
        out = multi_cusparse_SPMM(g, N_, E_T_, scale)
        reduce_last = _need_reduce_last_dim(N, E)
        ctx.backward_cache = g, N, E, reduce_last
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dZ):
        g, N, E, reduce_last = ctx.backward_cache
        dZ_, scaledZ = quantize(dZ)
        dN = dE = None
        N_, scaleN = quantize(N)
        if ctx.needs_input_grad[1]:
            scaleE = cuda_kernels.get_scale(E)  
            E_T_ = transpose_quant(E, scaleE)
            scale = scaleN * scaleE
            dN = multi_cusparse_SPMM(g, dZ_, E_T_, scale, True)
        if ctx.needs_input_grad[2]:
            if reduce_last:
                dE, _ = mySDDMM_int8(g, 'dot', N_, dZ_, scaleN, scaledZ, False)
            else:
                dE, _ = mySDDMM_int8(g, 'mul', N_, dZ_, scaleN, scaledZ, False)
        return None, dN, dE

times = 5

def test_noncached(g, h, n):
    E = torch.rand((g.num_edges(), h, 1), device=g.device, requires_grad=True)
    N = torch.rand((g.num_nodes(), h, n), device=g.device, requires_grad=True)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    forward = torch.cuda.Event(enable_timing=True)
    backward = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(times):
        N = spmm_u_mul_e_nocache.apply(g, N, E)
    forward.record()
    loss = torch.mean(N)
    loss.backward()
    backward.record()
    torch.cuda.synchronize()
    return start.elapsed_time(forward) / times + forward.elapsed_time(backward) / times

def test_cached(g, h, n):
    E = torch.rand((g.num_edges(), h, 1), device=g.device, requires_grad=True)
    N = torch.rand((g.num_nodes(), h, n), device=g.device, requires_grad=True)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    forward = torch.cuda.Event(enable_timing=True)
    backward = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(times):
        N = spmm_u_mul_e.apply(g, N, E)
    forward.record()
    loss = torch.mean(N)
    loss.backward()
    backward.record()
    torch.cuda.synchronize()
    return start.elapsed_time(forward) / times + forward.elapsed_time(backward) / times

if __name__ == "__main__":
    d_name = sys.argv[1]
    if d_name == 'AM':
        graph = AMDataset()[0]
        graph = dgl.to_homogeneous(graph)
    elif d_name == 'Pubmed':
        graph = PubmedGraphDataset()[0]
        graph = dgl.to_bidirected(graph)
    elif d_name == 'Reddit':
        graph = RedditDataset()[0]
        # graph = dgl.to_bidirected(graph)
    else:
        dataset = DglNodePropPredDataset(name=d_name)
        graph, _ = dataset[0]
        graph = dgl.to_bidirected(graph)

    graph = graph.remove_self_loop().add_self_loop()
    graph = graph.to('cuda')
    graph_preprocess(graph)
    print(graph)
    test_cached(graph, 4, 128) # dry run
    print(f"{test_cached(graph, 4, 128)}, {test_noncached(graph, 4, 128)},{test_cached(graph, 2, 256)}, {test_noncached(graph, 2, 256)}")
