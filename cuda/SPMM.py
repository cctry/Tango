import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function, once_differentiable
from .kernels import *


class spmm_copy_u(Function):
    @staticmethod
    def forward(ctx, g, N, scaleN=None):
        N_, scaleN = quantize(N, scaleN)
        out = csr_SPMM(g, N_, scaleN)
        ctx.backward_cache = g
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dZ):
        g = ctx.backward_cache
        dZ_, scaledZ = quantize(dZ)
        dN = None
        if ctx.needs_input_grad[1]:
            dN = csr_SPMM(g, dZ_, scaledZ, True)
        return None, dN, None

def _need_reduce_last_dim(ufeat, efeat):
    """Indicates whether to reduce the last dimension on edges
    in the backward pass of spmm,
    if so, use dot instead of mul."""
    if ufeat is None or efeat is None:
        return False
    ushp = ufeat.shape
    eshp = efeat.shape
    return ushp[1:-1] == eshp[1:-1] and eshp[-1] == 1 and ushp[-1] > 1

class spmm_u_mul_e(Function):
    @staticmethod
    def forward(ctx, g, N, E):
        scaleN = cuda_kernels.get_scale(N) 
        scaleE = cuda_kernels.get_scale(E)  
        E_T_ = transpose_quant(E, scaleE)
        scale = scaleN * scaleE
        N_, scaleN = quantize(N)
        out = multi_cusparse_SPMM(g, N_, E_T_, scale)
        reduce_last = _need_reduce_last_dim(N, E)
        ctx.backward_cache = g, N_, scaleN, E_T_, scale, reduce_last
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dZ):
        g, N_, scaleN, E_T_, scale, reduce_last = ctx.backward_cache
        dZ_, scaledZ = quantize(dZ)
        dN = dE = None
        if ctx.needs_input_grad[1]:
            dN = multi_cusparse_SPMM(g, dZ_, E_T_, scale, True)
        if ctx.needs_input_grad[2]:
            if reduce_last:
                dE, _ = mySDDMM_int8(g, 'dot', N_, dZ_, scaleN, scaledZ, False)
            else:
                dE, _ = mySDDMM_int8(g, 'mul', N_, dZ_, scaleN, scaledZ, False)
        return None, dN, dE