import sys
import dgl
from dgl.data import  PubmedGraphDataset, RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np
sys.path.append('../')
from cuda.util import graph_preprocess


def load_graph(name):
    if name == 'reddit':
        g = RedditDataset()[0]
    elif name == 'pubmed':
        g = PubmedGraphDataset()[0]
    elif name == "dblp":
        out = np.loadtxt('dataset/com-dblp.ungraph.txt', skiprows=4, unpack=True, dtype=np.int64)
        g = dgl.graph((out[0], out[1]))
        g = dgl.to_bidirected(g)
    elif name == "amazon":
        out = np.loadtxt('dataset/amazon0505.txt', skiprows=4, unpack=True, dtype=np.int64)
        g = dgl.graph((out[0], out[1]))
        g = dgl.to_bidirected(g)
    elif name == "ogbn-arxiv":
        g, _ = DglNodePropPredDataset(name='ogbn-arxiv')[0]
        g = dgl.to_bidirected(g)
    else:
        raise ValueError('Unknown dataset: {}'.format(name))
    g = g.remove_self_loop().add_self_loop()
    g = g.to('cuda')
    graph_preprocess(g)
    return g