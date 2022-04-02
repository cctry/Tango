import sys
import time
import torch
import dgl
from util import *
sys.path.append('../')
from cuda.kernels import *
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import PubmedGraphDataset, AMDataset, RedditDataset
from cuda.util import *

# def test(g, N, E):
#     scaleE = cuda_kernels.get_scale(E)  
#     scaleN = cuda_kernels.get_scale(N) 
#     E_T_ = transpose_quant(E, scaleE)
#     N_T_ = transpose_quant(N, scaleN)
#     scale = scaleN * scaleE
#     return multi_cusparse_SPMV(g, N_T_, E_T_, scale)

def test(g, N, E):
    scaleE = cuda_kernels.get_scale(E)  
    N_, scaleN = quantize(N)
    E_T_ = transpose_quant(E, scaleE)
    scale = scaleN * scaleE
    return multi_cusparse_SPMM(g, N_, E_T_, scale)

def test_correctness(g, n):
    E = torch.rand((g.num_edges(), n, 1), device=g.device)
    N = torch.rand((g.num_nodes(), n, 1), device=g.device)
    s = dgl.ops.u_mul_e_sum(g, N, E)
    print(s)
    s = test(g, N, E)
    print(s)


times = 5

def test_dgl(g, n):
    E = torch.rand((g.num_edges(), n, 1), device=g.device)
    N = torch.rand((g.num_nodes(), n, 1), device=g.device)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(times):
        s = dgl.ops.u_mul_e_sum(g, N, E)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / times


def test_ours(g, n):
    E = torch.rand((g.num_edges(), n, 1), device=g.device)
    N = torch.rand((g.num_nodes(), n, 1), device=g.device)
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
    test_correctness(g, 2)
    # for i in range(2, 13, 1):
    #     print(f"{test_dgl(g, i)},{test_ours(g, i)}")
