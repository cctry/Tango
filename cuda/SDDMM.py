import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function, once_differentiable
from .kernels import *
import dgl
import copy


class sddmm_u_add_v(Function):
    @staticmethod
    def forward(ctx, g, U, V, scaleU_=None, scaleV_=None):
        U_, scaleU = quantize(U, scaleU_)
        if (id(U) == id(V)):
            V_, scaleV = U_, scaleU
        else:
            V_, scaleV = quantize(V, scaleV_)
        out, _ = mySDDMM_int8(g, 'add', U_, V_, scaleU, scaleV, False)
        ctx.backward_cache = g
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dZ):
        g = ctx.backward_cache
        dZ_, scaledZ = quantize(dZ)
        dU = dV = None
        if ctx.needs_input_grad[2]:
            dV = incidence_SPMM(g, dZ_, scaledZ)
        if ctx.needs_input_grad[1]:
            dU = incidence_SPMM(g, dZ_, scaledZ, True)
        return None, dU, dV, None, None
