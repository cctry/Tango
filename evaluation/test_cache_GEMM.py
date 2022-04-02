import sys
import time
import torch
import torch.nn as nn
from torch.autograd.function import Function, once_differentiable
from util import *
sys.path.append('../')
from cuda.QLinear import QLinear, Qlinear_func, QgemmNT_func
from cuda.kernels import *
import os


class Qlinear_func_nocache(Function):
    @staticmethod
    def forward(ctx, X, W, b=None):
        scaleX = cuda_kernels.get_scale(X)
        scaleW = cuda_kernels.get_scale(W)
        X_ = torch.empty(X.shape, dtype=torch.int8, device=X.device)
        W_ = torch.empty(W.shape, dtype=torch.int8, device=X.device)
        out = QGEMM_forward(X, X_, scaleX, W, W_, scaleW, b)
        ctx.backward_cache = X, W, b
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        X, W, b = ctx.backward_cache
        dX = dW = db = None
        dY_, scaledY = quantize(grad_output)
        if ctx.needs_input_grad[0]:
            W_, scaleW = quantize(W)
            dX = torch.empty(X.shape, dtype=torch.float32, device=X.device)
            cuda_kernels.QGEMM_backward_gradX(dX, dY_, W_, scaledY, scaleW)
        if ctx.needs_input_grad[1]:
            X_, scaleX = quantize(X)
            dW = torch.empty(W.shape, dtype=torch.float32, device=X.device)
            cuda_kernels.QGEMM_backward_gradW(dW, dY_, X_, scaledY, scaleX)
        if b is not None and ctx.needs_input_grad[2]:
            db = grad_output.sum(0)
        return dX, dW, db


class RefLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(RefLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        t = input.view(-1, input.size(-1))
        assert t.size(-1) % 4 == 0, "feature size % 4 != 0"
        assert t.stride(0) % 4 == 0, "stride % 4 != 0"
        out_shape = t.shape[:-1] + (self.weight.size(0), )
        out = Qlinear_func_nocache.apply(t, self.weight, self.bias)
        return out.reshape(out_shape)


def test_performance(X, nfeat):
    nlayer = 10
    layers = nn.ModuleList([RefLinear(nfeat, nfeat, bias=False)
                            for i in range(nlayer)]).cuda()
    forward_ref, backward_ref = measure(layers, X)
    layers = nn.ModuleList([QLinear(nfeat, nfeat, bias=False)
                            for i in range(nlayer)]).cuda()
    forward, backward = measure(layers, X)
    print(f"{forward_ref + backward_ref}\n{forward + backward}")


def measure(layers, X):
    X_ = X.clone().cuda()
    model = nn.Sequential(*layers).cuda()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    forward = torch.cuda.Event(enable_timing=True)
    backward = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda.nvtx.range_push("forward")
    out = model(X_)
    torch.cuda.nvtx.range_pop()
    forward.record()
    loss = out[0][0]
    torch.cuda.nvtx.range_push("backward")
    loss.backward()
    torch.cuda.nvtx.range_pop()
    backward.record()
    torch.cuda.synchronize()
    forward_time = start.elapsed_time(forward) / len(layers)
    backward_time = forward.elapsed_time(backward) / len(layers)
    return forward_time, backward_time

if __name__ == "__main__":
    d_name = sys.argv[1]
    g = load_graph(d_name)
    print(g)
    N = g.number_of_nodes()
    in_feat = int(sys.argv[2])
    out_feat = in_feat
    X = torch.randn(N, in_feat, requires_grad=True, device='cuda')
    X_clone = X.clone().cuda()
    W = torch.randn(out_feat, in_feat, requires_grad=True, device='cuda')
    Qlinear_func_nocache.apply(X, W) # dry run
    torch.cuda.empty_cache()
    test_performance(X, in_feat)
