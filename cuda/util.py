import torch
import dgl


def graph_preprocess(graph):
    graph.inc_in = graph.inc('in', ctx = graph.device).coalesce().indices()
    graph.inc_out = graph.inc('out', ctx = graph.device).coalesce().indices()
    assert graph.inc_in.is_cuda, "graph must be on GPU"
    graph.adj_csr = graph.adj_sparse('csr')
    graph.adj_csc = graph.adj_sparse('csc')
    graph.coo = graph.edges()

