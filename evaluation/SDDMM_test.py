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

times = 20

def test_dgl(op, g, h, n):
    U = torch.rand((g.num_nodes(), h, n), device=g.device)
    V = torch.rand((g.num_nodes(), h, n), device=g.device)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(times):
        if op == 'add':
            s = dgl.ops.u_add_v(g, U, V)
        elif op == 'dot':
            s = dgl.ops.u_dot_v(g, U, V)
        elif op == 'mul':
            s = dgl.ops.u_mul_v(g, U, V)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / times

def test_ours(op, g, h, n):
    U = torch.rand((g.num_nodes(), h, n), device=g.device)
    V = torch.rand((g.num_nodes(), h, n), device=g.device)
    U_, scaleU = quantize(U)
    del U 
    V_, scaleV = quantize(V)
    del V
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(times):
        s = mySDDMM_int8(g, op, U_, V_, scaleU, scaleV, False)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / times

if __name__ == '__main__':
    d_name = sys.argv[1]
    g = load_graph(d_name)
    print(g)
    test_dgl('dot', g, 4, 4) # dry run
    # print(f"{test_dgl('add', g, 4, 64)}, {test_dgl('dot', g, 4, 64)}, {test_ours('add', g, 4, 64)}, {test_ours('dot', g, 4, 64)}")
    # Reddit graph
    print(f"{test_dgl('add', g, 1, 16)}, {test_dgl('dot', g, 4, 64)}, {test_ours('add', g, 1, 16)}, {test_ours('dot', g, 4, 64)}")