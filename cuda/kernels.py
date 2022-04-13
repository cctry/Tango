import torch
from torch.utils.cpp_extension import load
from dgl.sparse import infer_broadcast_shape
import os

CUTLASS_PATH = '/workspace/cutlass'

path = __file__.replace('kernels.py', 'src/')

sources = ["cuda.cpp", "SDDMM.cu",
           "QGEMM_forward.cu", "QGEMM_backward.cu", "quantization.cu", "SPMM.cu", "Qgemm_impl.cu"]
cuda_kernels = load(name="cuda_kernels", extra_cflags=["-O3"], extra_cuda_cflags=[
                    "-O3", "--use_fast_math", "--generate-line-info", "-lnvrtc", "-lcusparse"],
                    extra_include_paths=[
                        CUTLASS_PATH + '/include', CUTLASS_PATH + '/tools/util/include'],
                    sources=[path + s for s in sources], verbose=False)


def quantize(t: torch.Tensor, scale: torch.Tensor = None):
    """
    Quantize the input tensor to 8-bit.
    :param t: input tensor
    :return: quantized tensor and the scale
    """
    assert t.is_cuda, "tensor must be on GPU"
    if scale is None:
        scale = cuda_kernels.get_scale(t)
    out = torch.empty(t.shape, dtype=torch.int8, device=t.device)
    cuda_kernels.quantize(out, t, scale)
    return out, scale


def mySDDMM_int8(g, op, U, V, scaleU, scaleV, isReduce, reverse=False):
    if reverse:
        dst, src = g.coo
    else:
        src, dst = g.coo
    assert src.is_cuda, "src must be on GPU"
    assert dst.is_cuda, "dst must be on GPU"
    assert U.is_cuda, "U must be on GPU"
    assert V.is_cuda, "V must be on GPU"
    assert scaleU.is_cuda, "scaleU must be on GPU"
    assert scaleV.is_cuda, "scaleV must be on GPU"
    UV_unsqueezed = False
    if len(U.shape) == 2 and len(V.shape) == 2:
        U = U.unsqueeze(1)
        V = V.unsqueeze(1)
        UV_unsqueezed = True
    out_shape = (src.size(0), ) + infer_broadcast_shape(op,
                                                        U.shape[1:], V.shape[1:])
    out = torch.empty(out_shape, dtype=torch.float32, device=src.device)
    if isReduce:
        scale = torch.zeros((1,), dtype=torch.float32, device=src.device)
    else:
        scale = torch.Tensor()
    cuda_kernels.mySDDMM_int8(op, out, scale, src, dst, U, V, scaleU, scaleV)
    if UV_unsqueezed:
        out = out.squeeze(1)
    return out, scale


# def mySPMM_int8_reduce(g, U, E, scaleU, scaleE, reverse=False, out=None):
#     scale = torch.zeros((1,), dtype=torch.float32, device=U.device)
#     if reverse:
#         indptr, indices, edge_ids = g.adj_sparse('csr')
#     else:
#         indptr, indices, edge_ids = g.adj_sparse('csc')
#     if edge_ids is None:
#         edge_ids = torch.arange(indptr.size(0) - 1, device=indptr.device)
#     if out == None:
#         out_shape = (indptr.size(0) - 1, ) + infer_broadcast_shape("mul",
#                                                                    U.shape[1:], E.shape[1:])
#         out = torch.zeros(out_shape, dtype=torch.float32, device=U.device)
#     cuda_kernels.mySPMM_int8_reduce(out, scale, indptr, indices, edge_ids,
#                                     U, E, scaleU, scaleE)
#     return out, scale


def QGEMM_forward(x, x_, scaleX, w, w_, scaleW, bias=None, isReduce=False):
    assert x.is_cuda, "x must be on GPU"
    assert w.is_cuda, "w must be on GPU"
    assert bias is None or bias.is_cuda, "bias must be on GPU"
    assert x.size(1) == w.size(
        1), "x and w must have the same number of columns"
    out_fp = torch.empty((x.size(0), w.size(0)),
                         dtype=torch.float32, device=x.device)
    scale_out = torch.zeros_like(scaleX) if isReduce else torch.Tensor()
    bias = torch.Tensor() if bias is None else bias
    cuda_kernels.QGEMM_forward(
        out_fp, x, x_, w, w_, bias, scale_out, scaleX, scaleW)
    if isReduce:
        return out_fp, scale_out
    else:
        return out_fp


def QGEMM_backward(dY, X, W, scaledY, scaleX, scaleW, X_need=True, W_need=True):
    assert scaledY.is_cuda, "scaledY must be on GPU"
    assert dY.is_cuda, "dY must be on GPU"
    dX, dW = None, None
    if X_need:
        assert W.is_cuda, "X must be on GPU"
        assert scaleW.is_cuda, "scaleW must be on GPU"
        dX = torch.empty(X.shape, dtype=torch.float32, device=X.device)
        cuda_kernels.QGEMM_backward_gradX(dX, dY, W, scaledY, scaleW)
    if W_need:
        assert X.is_cuda, "X must be on GPU"
        assert scaleX.is_cuda, "scaleW must be on GPU"
        dW = torch.empty(W.shape, dtype=torch.float32, device=X.device)
        cuda_kernels.QGEMM_backward_gradW(dW, dY, X, scaledY, scaleX)
    return dX, dW


def transpose_quant(data, scale):
    assert data.ndim == 3
    nrow = data.size(0)
    ncol = data.numel() // nrow
    data_ = cuda_kernels.transpose_quant(data, scale, nrow, ncol)
    return data_.reshape((data.size(1), data.size(2), nrow))


def multi_cusparse_SPMM(g, N, E_T, scale, reverse=False):
    """
    E_T should be in shape (H, D, NNZ)
    """
    assert E_T.ndim == 3 and N.ndim == 3, "E and N must be 3D tensors"
    assert E_T.size(1) == 1, "Cannot broadcast E"
    if reverse:
        indptr, indices, edge_ids = g.adj_csr
    else:
        indptr, indices, edge_ids = g.adj_csc
    if edge_ids is None:
        E_T_new = E_T
    else:
        E_T_new = torch.index_select(E_T, -1, edge_ids)
    return cuda_kernels.multi_cusparse_SPMM(indptr, indices, N, E_T_new, scale)


def incidence_SPMM(g, E, scaleE, reverse=False):
    assert E.is_cuda and scaleE.is_cuda
    if reverse:
        coo = g.inc_out
    else:
        coo = g.inc_in
    src = coo[0]
    dst = coo[1]
    out_shape = (g.num_nodes(), ) + E.shape[1:]
    dim = E.numel() // E.size(0)
    out = torch.empty((g.num_nodes(), dim),
                      dtype=torch.float32, device=E.device)
    cuda_kernels.incidence_SPMM(out, src, dst, E, scaleE)
    return out.view(out_shape)

def csr_SPMM(g, N, scaleN, reverse=False):
    if reverse:
        indptr, indices, _ = g.adj_csr
    else:
        indptr, indices, _ = g.adj_csc
    feat_shape = N.shape[1:]
    out_shape = (g.num_nodes(), ) + feat_shape
    dim = N.numel() // N.size(0)
    out = torch.empty((g.num_nodes(), dim),
                      dtype=torch.float32, device=N.device)
    cuda_kernels.csr_SPMM(out, indptr, indices, N, scaleN)
    return out.view(out_shape)


def multi_cusparse_SPMV(g, N_T, E_T, scale, reverse=False):
    assert E_T.ndim == 3 and N_T.ndim == 3, "E and N must be 3D tensors"
    if reverse:
        indptr, indices, edge_ids = g.adj_csr
    else:
        indptr, indices, edge_ids = g.adj_csc
    if edge_ids is None:
        E_T_new = E_T
    else:
        E_T_new = torch.index_select(E_T, -1, edge_ids)
    return cuda_kernels.multi_cusparse_SPMM(indptr, indices, N_T, E_T_new, scale)
