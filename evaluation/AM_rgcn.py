import argparse
import sys
import torch as th
import torch.nn.functional as F
sys.path.append('../')
from cuda.util import graph_preprocess
from cuda.QLinear import QLinear, Qlinear_func
from cuda.SPMM import *
import dgl

from torchmetrics.functional import accuracy

from model import RGCN

from dgl.data.rdf import AMDataset

def load_data(get_norm=False, inv_target=False):
    dataset = AMDataset()

    # Load hetero-graph
    hg = dataset[0]

    num_rels = len(hg.canonical_etypes)
    category = dataset.predict_category
    num_classes = dataset.num_classes
    labels = hg.nodes[category].data.pop('labels')
    train_mask = hg.nodes[category].data.pop('train_mask')
    test_mask = hg.nodes[category].data.pop('test_mask')
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    if get_norm:
        # Calculate normalization weight for each edge,
        # 1. / d, d is the degree of the destination node
        for cetype in hg.canonical_etypes:
            hg.edges[cetype].data['norm'] = dgl.norm_by_dst(hg, cetype).unsqueeze(1)
        edata = ['norm']
    else:
        edata = None

    # get target category id
    category_id = hg.ntypes.index(category)

    g = dgl.to_homogeneous(hg, edata=edata)
    # Rename the fields as they can be changed by for example NodeDataLoader
    g.ndata['ntype'] = g.ndata.pop(dgl.NTYPE)
    g.ndata['type_id'] = g.ndata.pop(dgl.NID)
    node_ids = th.arange(g.num_nodes())

    # find out the target node ids in g
    loc = (g.ndata['ntype'] == category_id)
    target_idx = node_ids[loc]

    if inv_target:
        # Map global node IDs to type-specific node IDs. This is required for
        # looking up type-specific labels in a minibatch
        inv_target = th.empty((g.num_nodes(),), dtype=th.int64)
        inv_target[target_idx] = th.arange(0, target_idx.shape[0],
                                           dtype=inv_target.dtype)
        return g, num_rels, num_classes, labels, train_idx, test_idx, target_idx, inv_target
    else:
        return g, num_rels, num_classes, labels, train_idx, test_idx, target_idx

def main(args):
    g, num_rels, num_classes, labels, train_idx, test_idx, target_idx = load_data(
        args.dataset, get_norm=True)

    model = RGCN(g.num_nodes(),
                 args.n_hidden,
                 num_classes,
                 num_rels,
                 num_bases=args.n_bases)

    device = th.device(0)
    labels = labels.to(device)
    model = model.to(device)
    g = g.to(device)
    graph_preprocess(g)

    optimizer = th.optim.Adam(
        model.parameters(), lr=1e-2, weight_decay=args.wd)

    model.train()
    for epoch in range(100):
        logits = model(g)
        logits = logits[target_idx]
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits[train_idx].argmax(
            dim=1), labels[train_idx]).item()
        print("Epoch {:05d} | Train Accuracy: {:.4f} | Train Loss: {:.4f}".format(
            epoch, train_acc, loss.item()))
    print()

    model.eval()
    with th.no_grad():
        logits = model(g)
    logits = logits[target_idx]
    test_acc = accuracy(logits[test_idx].argmax(
        dim=1), labels[test_idx]).item()
    print("Test Accuracy: {:.4f}".format(test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RGCN for entity classification')
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden units")
    parser.add_argument("--n-bases", type=int, default=-1,
                        help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=['aifb', 'mutag', 'bgs', 'am'],
                        help="dataset to use")
    parser.add_argument("--wd", type=float, default=5e-4,
                        help="weight decay")

    args = parser.parse_args()
    print(args)
    main(args)
