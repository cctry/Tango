import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function, once_differentiable
from . import kernels


class QgemmNT_func(Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return kernels.cuda_kernels.QgemmNT(A, B)

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = grad_B = None
        if ctx.needs_input_grad[0]:
            grad_A = kernels.cuda_kernels.QgemmNN(grad_output, B)
        if ctx.needs_input_grad[1]:
            grad_B = kernels.cuda_kernels.QgemmTN(grad_output, A)
        return grad_A, grad_B


class Qlinear_func(Function):
    @staticmethod
    def forward(ctx, X, W, b=None):
        scaleX = kernels.cuda_kernels.get_scale(X)
        scaleW = kernels.cuda_kernels.get_scale(W)
        X_ = torch.empty(X.shape, dtype=torch.int8, device=X.device)
        W_ = torch.empty(W.shape, dtype=torch.int8, device=X.device)
        out = kernels.QGEMM_forward(X, X_, scaleX, W, W_, scaleW, b)
        ctx.backward_cache = X_, scaleX, W_, scaleW, b
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        X_, scaleX, W_, scaleW, b = ctx.backward_cache
        grad_input = grad_weight = grad_bias = None
        dY_, scaledY = kernels.quantize(grad_output)
        grad_input, grad_weight = kernels.QGEMM_backward(
            dY_, X_, W_, scaledY, scaleX, scaleW, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        if b is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias


class Qlinear_func_reduce(Function):
    @staticmethod
    def forward(ctx, X, W, b=None):
        scaleX = kernels.cuda_kernels.get_scale(X)
        scaleW = kernels.cuda_kernels.get_scale(W)
        X_ = torch.empty(X.shape, dtype=torch.int8, device=X.device)
        W_ = torch.empty(W.shape, dtype=torch.int8, device=X.device)
        out, scaleO = kernels.QGEMM_forward(
            X, X_, scaleX, W, W_, scaleW, b, True)
        ctx.backward_cache = X_, scaleX, W_, scaleW, b
        return out, scaleO

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_dummy):
        X_, scaleX, W_, scaleW, b = ctx.backward_cache
        grad_input = grad_weight = grad_bias = None
        dY_, scaledY = kernels.quantize(grad_output)
        grad_input, grad_weight = kernels.QGEMM_backward(
            dY_, X_, W_, scaledY, scaleX, scaleW, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        if b is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias


class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.isReduce = False

    def enable_reduce(self):
        self.isReduce = True

    def forward(self, input):
        t = input.view(-1, input.size(-1))
        assert t.size(-1) % 4 == 0, "feature size % 4 != 0"
        assert t.stride(0) % 4 == 0, "stride % 4 != 0"
        out_shape = t.shape[:-1] + (self.weight.size(0), )
        if self.isReduce:
            out, scale = Qlinear_func_reduce.apply(t, self.weight, self.bias)
            return out.reshape(out_shape), scale
        else:
            out = Qlinear_func.apply(t, self.weight, self.bias)
            return out.reshape(out_shape)
