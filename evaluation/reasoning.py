import sys
import time
import torch
import dgl
from util import *
sys.path.append('../')
from cuda.kernels import *
from cuda.util import *

def test_correctness(g, n):
    E = torch.rand((g.num_edges(), n, 1), device=g.device)
    s = dgl.ops.copy_e_sum(g, E)
    # print(s)
    E_, scaleE = quantize(E)
    s = incidence_SPMM(g, E_, scaleE)
    # print(s)

g = load_graph('ogbn-arxiv')
test_correctness(g, 16)
g = load_graph('dblp')
test_correctness(g, 16)
g = load_graph('amazon')
test_correctness(g, 16)
g = load_graph('reddit')
test_correctness(g, 16)
g = load_graph('pubmed')
test_correctness(g, 16)