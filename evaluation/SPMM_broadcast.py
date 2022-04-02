import sys
import time
import torch
import dgl
from util import *
sys.path.append('../')
from cuda.kernels import *
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import PubmedGraphDataset, RedditDataset
from dgl.data import AMDataset
from cuda.util import *

def test(g, N, E):
    N_, scaleN = quantize(N)
    scaleE = cuda_kernels.get_scale(E)  
    E_T_ = transpose_quant(E, scaleE)
    scale = scaleN * scaleE
    return multi_cusparse_SPMM(g, N_, E_T_, scale)


def test_correctness(g, h, n, p = False):
    E = torch.rand((g.num_edges(), h, 1), device=g.device)
    N = torch.rand((g.num_nodes(), h, n), device=g.device)
    s = dgl.ops.u_mul_e_sum(g, N, E)
    if p: 
        print(s)
    s = test(g, N, E)
    if p: 
        print(s)


times = 20


def test_dgl(g, h, n):
    E = torch.rand((g.num_edges(), h, 1), device=g.device)
    N = torch.rand((g.num_nodes(), h, n), device=g.device)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(times):
        s = dgl.ops.u_mul_e_sum(g, N, E)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / times


def test_ours(g, h, n):
    E = torch.rand((g.num_edges(), h, 1), device=g.device)
    N = torch.rand((g.num_nodes(), h, n), device=g.device)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(times):
        s = test(g, N, E)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / times


if __name__ == '__main__':
    d_name = sys.argv[1]
    g = load_graph(d_name)
    print(g)
    test_correctness(g, 4, 256, False)
    print(f"{test_dgl(g, 2, 128)}")
    print(f"{test_ours(g, 2, 128)}")
    print(f"{test_dgl(g, 4, 128)}")
    print(f"{test_ours(g, 4, 128)}")
    print(f"{test_dgl(g, 2, 256)}")
    print(f"{test_ours(g, 2, 256)}")
    print(f"{test_dgl(g, 4, 256)}")
    print(f"{test_ours(g, 4, 256)}")
