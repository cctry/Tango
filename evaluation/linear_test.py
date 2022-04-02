import sys
import time
import torch
import torch.nn as nn
from util import *
sys.path.append('../')
from cuda.QLinear import QLinear, Qlinear_func, QgemmNT_func
import os


def test_performance(N, nfeat):
    X = torch.randn((N, nfeat), requires_grad=True, device='cuda')
    nlayer = 10
    layers = nn.ModuleList([nn.Linear(nfeat, nfeat, bias=False) for i in range(nlayer)]).cuda()
    ref = measure(layers, X)
    del X
    X = torch.randn((N, nfeat), requires_grad=True, device='cuda')
    layers = nn.ModuleList([QLinear(nfeat, nfeat, bias=False) for i in range(nlayer)]).cuda()
    ours = measure(layers, X)
    return f"{ref}, {ours}"

def measure(layers, X):
    model = nn.Sequential(*layers).cuda()
    start = torch.cuda.Event(enable_timing=True)
    forward = torch.cuda.Event(enable_timing=True)
    backward = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    out = model(X)
    forward.record()
    loss = out[0][0]
    loss.backward()
    backward.record()
    torch.cuda.synchronize()
    forward_time = start.elapsed_time(forward) / len(layers)
    backward_time = forward.elapsed_time(backward) / len(layers)
    return forward_time + backward_time


def test_correctness(N, nfeat):
    X = torch.randn(N, nfeat, device='cuda')
    W = torch.randn(nfeat, nfeat, device='cuda')
    b = torch.randn(nfeat, device='cuda')
    out = X @ W.T + b
    # print(out)
    # print(X.shape)
    out = Qlinear_func.apply(X, W, b)
    # print(out)



if __name__ == "__main__":
    d_name = sys.argv[1]
    g = load_graph(d_name)
    print(g)
    N = g.number_of_nodes()
    test_correctness(N, 128) # dry run
    print(f"{test_performance(N, 256)}, {test_performance(N, 512)}")
