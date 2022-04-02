import sys
import time
import torch
import torch.nn as nn
import dgl
from util import *
sys.path.append('../')
from cuda.QLinear import QLinear, Qlinear_func, QgemmNT_func, Qlinear_func_reduce
from cuda.SPMM import spmm_copy_u
from cuda.kernels import *
from cuda.util import *

times = 20



def fused(g, X, W, X_, W_, scaleX, scaleW):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for i in range(20):
        feat, scaleO = QGEMM_forward(X, X_, scaleX, W, W_, scaleW, None, True)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def baseline(g, X, W, X_, W_, scaleX, scaleW):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for i in range(20):
        feat = QGEMM_forward(X, X_, scaleX, W, W_, scaleW, None, False)
        scale = cuda_kernels.get_scale(feat) 
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


if __name__ == "__main__":
    d_name = sys.argv[1]
    g = load_graph(d_name)
    dim = int(sys.argv[2])
    X = torch.rand((g.number_of_nodes(), dim), device='cuda')
    W = torch.rand((dim, dim), device='cuda')
    scaleX = cuda_kernels.get_scale(X)
    scaleW = cuda_kernels.get_scale(W)
    X_ = torch.empty(X.shape, dtype=torch.int8, device=X.device)
    W_ = torch.empty(W.shape, dtype=torch.int8, device=X.device)

    baseline(g, X, W, X_, W_, scaleX, scaleW) # dry run
    print(f"{baseline(g, X, W, X_, W_, scaleX, scaleW)}\n{fused(g, X, W, X_, W_, scaleX, scaleW)}")
