#include <assert.h>
#include <cstdint>
#include <string>
#include <torch/extension.h>

void QGEMM_forward_impl(torch::Tensor &Y, torch::Tensor &X, torch::Tensor &X_q,
                        torch::Tensor &W, torch::Tensor &W_q,
                        torch::Tensor &bias, torch::Tensor &scaleY,
                        torch::Tensor &scaleX, torch::Tensor &scaleW);

void QGEMM_forward(torch::Tensor &Y, torch::Tensor &X, torch::Tensor &X_q,
                   torch::Tensor &W, torch::Tensor &W_q, torch::Tensor &bias,
                   torch::Tensor &scaleY, torch::Tensor &scaleX,
                   torch::Tensor &scaleW) {
    QGEMM_forward_impl(Y, X, X_q, W, W_q, bias, scaleY, scaleX, scaleW);
}

void quantize_impl(torch::Tensor &out, torch::Tensor &in, torch::Tensor &scale);
void quantize(torch::Tensor &out, torch::Tensor &in, torch::Tensor &scale) {
    quantize_impl(out, in, scale);
}
torch::Tensor get_scale_impl(torch::Tensor &in);
torch::Tensor get_scale(torch::Tensor &in) { return get_scale_impl(in); }

torch::Tensor transpose_quant_impl(torch::Tensor &in, torch::Tensor &scale,
                                   int64_t nrow, int64_t ncol);
torch::Tensor transpose_quant(torch::Tensor &in, torch::Tensor &scale,
                              int64_t nrow, int64_t ncol) {
    auto res = transpose_quant_impl(in, scale, nrow, ncol);
    return res;
}

void mySDDMM_int8_impl(std::string &op, torch::Tensor &out,
                       torch::Tensor &scaleOut, torch::Tensor &src,
                       torch::Tensor &dst, torch::Tensor &U, torch::Tensor &V,
                       torch::Tensor &scaleU, torch::Tensor &scaleV);

void mySDDMM_int8(std::string &op, torch::Tensor &out, torch::Tensor &scaleOut,
                  torch::Tensor &src, torch::Tensor &dst, torch::Tensor &U,
                  torch::Tensor &V, torch::Tensor &scaleU,
                  torch::Tensor &scaleV) {
    mySDDMM_int8_impl(op, out, scaleOut, src, dst, U, V, scaleU, scaleV);
}

// void mySPMM_int8_impl(torch::Tensor &out, torch::Tensor &scaleOut,
//                       const torch::Tensor &indptr, const torch::Tensor
//                       &indices, const torch::Tensor &edge_ids, const
//                       torch::Tensor &U, const torch::Tensor &E, const
//                       torch::Tensor &scaleU, const torch::Tensor &scaleE);

// void mySPMM_int8(torch::Tensor &out, torch::Tensor &scaleOut,
//                  const torch::Tensor &indptr, const torch::Tensor &indices,
//                  const torch::Tensor &edge_ids, const torch::Tensor &U,
//                  const torch::Tensor &E, const torch::Tensor &scaleU,
//                  const torch::Tensor &scaleV) {
//     mySPMM_int8_impl(out, scaleOut, indptr, indices, edge_ids, U, E, scaleU,
//                      scaleV);
// }

// void mySPMM_int8_copy_impl(torch::Tensor &out, torch::Tensor &scaleOut,
//                            const torch::Tensor &indptr,
//                            const torch::Tensor &indices,
//                            const torch::Tensor &edge_ids,
//                            const torch::Tensor &E, const torch::Tensor
//                            &scaleE);

// void mySPMM_int8_copy(torch::Tensor &out, torch::Tensor &scaleOut,
//                       const torch::Tensor &indptr, const torch::Tensor
//                       &indices, const torch::Tensor &edge_ids, const
//                       torch::Tensor &E, const torch::Tensor &scaleE) {
//     mySPMM_int8_copy_impl(out, scaleOut, indptr, indices, edge_ids, E,
//     scaleE);
// }

torch::Tensor Qgemm_impl(const torch::Tensor &A, const torch::Tensor &B,
                         bool trans_A, bool trans_B);

torch::Tensor QgemmNT(const torch::Tensor &A, const torch::Tensor &B) {
    return Qgemm_impl(A, B, false, true);
}

torch::Tensor QgemmTN(const torch::Tensor &A, const torch::Tensor &B) {
    return Qgemm_impl(A, B, true, false);
}

torch::Tensor QgemmNN(const torch::Tensor &A, const torch::Tensor &B) {
    return Qgemm_impl(A, B, false, false);
}

void QGEMM_backward_gradX_impl(torch::Tensor &dX, torch::Tensor &dY,
                               torch::Tensor &W, torch::Tensor &scaledY,
                               torch::Tensor &scaleW);

void QGEMM_backward_gradX(torch::Tensor &dX, torch::Tensor &dY,
                          torch::Tensor &W, torch::Tensor &scaledY,
                          torch::Tensor &scaleW) {
    QGEMM_backward_gradX_impl(dX, dY, W, scaledY, scaleW);
}

void QGEMM_backward_gradW_impl(torch::Tensor &dW, torch::Tensor &dY,
                               torch::Tensor &X, torch::Tensor &scaledY,
                               torch::Tensor &scaleX);

void QGEMM_backward_gradW(torch::Tensor &dW, torch::Tensor &dY,
                          torch::Tensor &X, torch::Tensor &scaledY,
                          torch::Tensor &scaleX) {
    QGEMM_backward_gradW_impl(dW, dY, X, scaledY, scaleX);
}

void cusp_SPMM_csr_impl(torch::Tensor &out, const torch::Tensor &indptr,
                        const torch::Tensor &indices, const torch::Tensor &U,
                        const torch::Tensor &scaleU);

void csr_SPMM(torch::Tensor &out, const torch::Tensor &indptr,
              const torch::Tensor &indices, const torch::Tensor &U,
              const torch::Tensor &scaleU) {
    cusp_SPMM_csr_impl(out, indptr, indices, U, scaleU);
}

torch::Tensor multi_cusparse_SPMM_broadcastE_impl(const torch::Tensor &indptr,
                                                  const torch::Tensor &indices,
                                                  const torch::Tensor &N,
                                                  const torch::Tensor &E_T,
                                                  const torch::Tensor &scale);

torch::Tensor multi_cusparse_SPMM_broadcastE(const torch::Tensor &indptr,
                                             const torch::Tensor &indices,
                                             const torch::Tensor &N,
                                             const torch::Tensor &E_T,
                                             const torch::Tensor &scale) {
    return multi_cusparse_SPMM_broadcastE_impl(indptr, indices, N, E_T, scale);
}

torch::Tensor multi_cusparse_SPMM_elementwise_impl(const torch::Tensor &indptr,
                                              const torch::Tensor &indices,
                                              const torch::Tensor &N,
                                              const torch::Tensor &E_T,
                                              const torch::Tensor &scale);
torch::Tensor multi_cusparse_SPMM_elementwise(const torch::Tensor &indptr,
                                              const torch::Tensor &indices,
                                              const torch::Tensor &N,
                                              const torch::Tensor &E_T,
                                              const torch::Tensor &scale) {
    return multi_cusparse_SPMM_elementwise_impl(indptr, indices, N, E_T, scale);
}

torch::Tensor multi_cusparse_SPMM(const torch::Tensor &indptr,
                                  const torch::Tensor &indices,
                                  const torch::Tensor &N,
                                  const torch::Tensor &E_T,
                                  const torch::Tensor &scale) {
    if (E_T.size(1) == 1)
        return multi_cusparse_SPMM_broadcastE(indptr, indices, N, E_T, scale);
    else {
        return multi_cusparse_SPMM_elementwise(indptr, indices, N, E_T, scale);
    }
}

void incidence_SPMM_impl(torch::Tensor &out, const torch::Tensor &src,
                         const torch::Tensor &dst, const torch::Tensor &E,
                         const torch::Tensor &scale);

void incidence_SPMM(torch::Tensor &out, const torch::Tensor &src,
                    const torch::Tensor &dst, const torch::Tensor &E,
                    const torch::Tensor &scale) {
    incidence_SPMM_impl(out, src, dst, E, scale);
}

torch::Tensor multi_cusparse_SPMV_impl(const torch::Tensor &indptr,
                                       const torch::Tensor &indices,
                                       const torch::Tensor &N_T,
                                       const torch::Tensor &E_T,
                                       const torch::Tensor &scale);
torch::Tensor multi_cusparse_SPMV(const torch::Tensor &indptr,
                                  const torch::Tensor &indices,
                                  const torch::Tensor &N_T,
                                  const torch::Tensor &E_T,
                                  const torch::Tensor &scale) {
    return multi_cusparse_SPMV_impl(indptr, indices, N_T, E_T, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("QGEMM_forward", &QGEMM_forward, "QGEMM_forward");
    m.def("quantize", &quantize, "quantize");
    m.def("get_scale", &get_scale, "get_scale");
    m.def("transpose_quant", &transpose_quant, "transpose_quant");
    m.def("QgemmNT", &QgemmNT, "QgemmNT");
    m.def("QgemmTN", &QgemmTN, "QgemmTN");
    m.def("QgemmNN", &QgemmNN, "QgemmNN");
    m.def("QGEMM_backward_gradX", &QGEMM_backward_gradX,
          "QGEMM_backward_gradX");
    m.def("QGEMM_backward_gradW", &QGEMM_backward_gradW,
          "QGEMM_backward_gradW");
    // cusparse
    m.def("csr_SPMM", &csr_SPMM, "csr_SPMM");
    m.def("multi_cusparse_SPMM", &multi_cusparse_SPMM, "multi_cusparse_SPMM");
    m.def("incidence_SPMM", &incidence_SPMM, "incidence_SPMM");
    m.def("multi_cusparse_SPMV", &multi_cusparse_SPMV, "multi_cusparse_SPMV");

    // m.def("mySPMM_int8", &mySPMM_int8, "mySPMM_int8");
    // m.def("mySPMM_int8_copy", &mySPMM_int8_copy, "mySPMM_int8_copy");

    m.def("mySDDMM_int8", &mySDDMM_int8, "mySDDMM_int8");
}