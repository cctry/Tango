import sys
import time
import torch
import dgl
from util import *
import scipy.sparse as sp
sys.path.append('../')
from cuda.kernels import *
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import PubmedGraphDataset, RedditDataset
from dgl.data import AMDataset
from cuda.util import *

def inc_csr_spmm(g, N, scaleN):
    indptr, indices = g.inc_in
    feat_shape = N.shape[1:]
    out_shape = (g.num_nodes(), ) + feat_shape
    dim = N.numel() // N.size(0)
    out = torch.empty((g.num_nodes(), dim),
                      dtype=torch.float32, device=N.device)
    cuda_kernels.csr_SPMM(out, indptr, indices, N, scaleN)
    return out.view(out_shape)



def test_correctness(g, n):
    E = torch.rand((g.num_edges(), n, 1), device=g.device)
    s = dgl.ops.copy_e_sum(g, E)
    # print(s)
    E_, scaleE = quantize(E)
    # s = incidence_SPMM(g, E_, scaleE)
    s = inc_csr_spmm(g, E_, scaleE)
    # print(s)

times = 20

def test_dgl(g, n):
    E = torch.rand((g.num_edges(), n), device=g.device)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(times):
        s = dgl.ops.copy_e_sum(g, E)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / times


def test_ours(g, n):
    E = torch.rand((g.num_edges(), n), device=g.device)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    E_, scaleE = quantize(E)
    start.record()
    for i in range(times):
        # s = inc_csr_spmm(g, E_, scaleE)
        s = incidence_SPMM(g, E_, scaleE)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / times

def prepare(g):
    inc_in = g.inc('in', ctx = g.device).coalesce().indices().cpu().numpy()
    coo = (inc_in[0], inc_in[1])
    row = g.num_nodes()
    col = g.num_edges()
    nnz = g.num_edges()
    coo = sp.coo_matrix((np.ones(nnz) , coo))
    print(coo.shape)
    csr = coo.tocsr()
    indptr = torch.tensor(csr.indptr).to(g.device).to(torch.int64)
    indices = torch.tensor(csr.indices).to(g.device).to(torch.int64)
    g.inc_in = indptr, indices
    assert g.inc_in[0].is_cuda, "graph must be on GPU"


if __name__ == "__main__":
    d_name = sys.argv[1]
    g = load_graph(d_name)
    print(g)
    prepare(g)
    test_correctness(g, 16)
    for i in range(4, 33, 4):
        print(f"{test_dgl(g, i)}, {test_ours(g, i)}")
