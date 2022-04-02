#include "utils.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <nvrtc.h>
#include <stdlib.h>
#include <torch/extension.h>

#define CHECK_CUSPARSE(func)                                                   \
    {                                                                          \
        cusparseStatus_t status = (func);                                      \
        if (status != CUSPARSE_STATUS_SUCCESS) {                               \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n",     \
                   __LINE__, cusparseGetErrorString(status), status);          \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

const auto transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
const auto transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
const auto ALG = CUSPARSE_SPMM_CSR_ALG2;

void cusp_SPMM_csr_impl(torch::Tensor &out, const torch::Tensor &indptr,
                        const torch::Tensor &indices, const torch::Tensor &N,
                        const torch::Tensor &scale) {
    auto handle = at::cuda::getCurrentCUDASparseHandle();
    cusparsePointerMode_t ptr_mode;
    cusparseGetPointerMode(handle, &ptr_mode);
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);
    int64_t m = indptr.size(0) - 1;
    int64_t n = out.size(-1);
    int64_t k = N.size(0);
    int64_t nnz = indices.size(0);
    float *alpha = scale.data_ptr<float>();
    auto beta_tensor = torch::zeros_like(scale);
    float *beta = beta_tensor.data_ptr<float>();
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    // sparse matrix A is all one
    auto A = torch::empty({nnz}, torch::dtype(torch::kInt8).device(N.device()));
    int8_t *A_ptr = A.data_ptr<int8_t>();
    cudaErrchk(cudaMemset(A_ptr, 1, nnz * sizeof(int8_t)));

    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, m, k, nnz, indptr.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(), A_ptr, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_8I));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, k, n, n, N.data_ptr<int8_t>(),
                                       CUDA_R_8I, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, m, n, n, out.data_ptr<float>(),
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // workspace
    size_t workspace_size;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, transA, transB, alpha, matA,
                                           matB, beta, matC, CUDA_R_32F, ALG,
                                           &workspace_size));
    auto workspace = torch::empty(
        {(long)workspace_size}, torch::dtype(torch::kInt8).device(N.device()));
    int8_t *workspace_ptr = workspace.data_ptr<int8_t>();
    // call SPMM
    CHECK_CUSPARSE(cusparseSpMM(handle, transA, transB, alpha, matA, matB, beta,
                                matC, CUDA_R_32F, ALG, workspace_ptr));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    cusparseSetPointerMode(handle, ptr_mode);
}

torch::Tensor multi_cusparse_SPMM_broadcastE_impl(const torch::Tensor &indptr,
                                                  const torch::Tensor &indices,
                                                  const torch::Tensor &N,
                                                  const torch::Tensor &E_T,
                                                  const torch::Tensor &scale) {
    const auto &device = N.device();
    float *alpha = scale.data_ptr<float>();
    auto beta_tensor = torch::zeros_like(scale);
    float *beta = beta_tensor.data_ptr<float>();
    auto handle = at::cuda::getCurrentCUDASparseHandle();
    cusparsePointerMode_t ptr_mode;
    cusparseGetPointerMode(handle, &ptr_mode);
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);
    const int64_t m = indptr.size(0) - 1;
    const int64_t n = N.size(-1);
    const int64_t k = N.size(0);
    const int64_t nnz = indices.size(0);
    const int64_t b = N.size(1);
    // Sparse matrix
    cusparseSpMatDescr_t matA;
    // E is transposed to (b, nnz, 1) making contiguous memory are edges
    auto A_ptr = E_T.data_ptr<int8_t>();
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, m, k, nnz, indptr.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(), A_ptr, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_8I));
    // Dense matrix
    cusparseDnMatDescr_t matB, matC;
    // Output
    auto out =
        torch::empty({m, b, n}, torch::dtype(torch::kFloat32).device(device));
    auto out_ptr = out.data_ptr<float>();
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, m, n, b * n, out_ptr, CUDA_R_32F,
                                       CUSPARSE_ORDER_ROW));
    // Node features
    auto N_ptr = N.data_ptr<int8_t>();
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, k, n, b * n, N_ptr, CUDA_R_8I,
                                       CUSPARSE_ORDER_ROW));
    // workspace
    size_t workspace_size;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, transA, transB, alpha, matA,
                                           matB, beta, matC, CUDA_R_32F, ALG,
                                           &workspace_size));
    auto workspace = torch::empty({(long)workspace_size},
                                  torch::dtype(torch::kInt8).device(device));
    auto *ws_ptr = workspace.data_ptr<int8_t>();
    for (int i = 1; i <= b; i++) {
        // call SPMM
        CHECK_CUSPARSE(cusparseSpMM(handle, transA, transB, alpha, matA, matB,
                                    beta, matC, CUDA_R_32F, ALG, ws_ptr));
        // move to next block
        CHECK_CUSPARSE(cusparseSpMatSetValues(matA, A_ptr + i * nnz));
        CHECK_CUSPARSE(cusparseDnMatSetValues(matB, N_ptr + i * n));
        CHECK_CUSPARSE(cusparseDnMatSetValues(matC, out_ptr + i * n));
    }
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    cusparseSetPointerMode(handle, ptr_mode);
    return out;
}

torch::Tensor multi_cusparse_SPMM_elementwise_impl(const torch::Tensor &indptr,
                                                   const torch::Tensor &indices,
                                                   const torch::Tensor &N,
                                                   const torch::Tensor &E_T,
                                                   const torch::Tensor &scale) {
    const auto &device = N.device();
    auto handle = at::cuda::getCurrentCUDASparseHandle();
    cusparsePointerMode_t ptr_mode;
    cusparseGetPointerMode(handle, &ptr_mode);
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);
    float *alpha = scale.data_ptr<float>();
    auto beta_tensor = torch::zeros_like(scale);
    float *beta = beta_tensor.data_ptr<float>();

    const int64_t m = indptr.size(0) - 1;
    const int64_t n = 1;
    const int64_t k = N.size(0);
    const int64_t nnz = indices.size(0);
    const int64_t b = N.size(1) * N.size(2);
    // Sparse matrix
    cusparseSpMatDescr_t matA;
    // E is transposed making contiguous memory are edges
    auto A_ptr = E_T.data_ptr<int8_t>();
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, m, k, nnz, indptr.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(), A_ptr, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_8I));
    // Dense matrix
    cusparseDnMatDescr_t matB, matC;
    // Output
    auto out = torch::empty({m, N.size(1), N.size(2)},
                            torch::dtype(torch::kFloat32).device(device));
    auto out_ptr = out.data_ptr<float>();
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, m, n, b * n, out_ptr, CUDA_R_32F,
                                       CUSPARSE_ORDER_ROW));
    // Node features
    auto N_ptr = N.data_ptr<int8_t>();
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, k, n, b * n, N_ptr, CUDA_R_8I,
                                       CUSPARSE_ORDER_ROW));
    // workspace
    size_t workspace_size;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, transA, transB, alpha, matA,
                                           matB, beta, matC, CUDA_R_32F, ALG,
                                           &workspace_size));
    auto workspace = torch::empty({(long)workspace_size},
                                  torch::dtype(torch::kInt8).device(device));
    auto *ws_ptr = workspace.data_ptr<int8_t>();
    for (int i = 1; i <= b; i++) {
        // call SPMM
        CHECK_CUSPARSE(cusparseSpMM(handle, transA, transB, alpha, matA, matB,
                                    beta, matC, CUDA_R_32F, ALG, ws_ptr));
        // move to next block
        CHECK_CUSPARSE(cusparseSpMatSetValues(matA, A_ptr + i * nnz));
        CHECK_CUSPARSE(cusparseDnMatSetValues(matB, N_ptr + i * n));
        CHECK_CUSPARSE(cusparseDnMatSetValues(matC, out_ptr + i * n));
        kernelErrchk();
    }
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    cusparseSetPointerMode(handle, ptr_mode);
    return out;
}

torch::Tensor multi_cusparse_SPMV_impl(const torch::Tensor &indptr,
                                       const torch::Tensor &indices,
                                       const torch::Tensor &N_T,
                                       const torch::Tensor &E_T,
                                       const torch::Tensor &scale) {
    const auto &device = N_T.device();
    auto handle = at::cuda::getCurrentCUDASparseHandle();
    cusparsePointerMode_t ptr_mode;
    cusparseGetPointerMode(handle, &ptr_mode);
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);
    float *alpha = scale.data_ptr<float>();
    auto beta_tensor = torch::zeros_like(scale);
    float *beta = beta_tensor.data_ptr<float>();

    const int64_t m = indptr.size(0) - 1;
    const int64_t n = 1;
    const int64_t k = N_T.size(-1);
    const int64_t nnz = indices.size(0);
    const int64_t b = N_T.size(1) * N_T.size(0);
    // Sparse matrix
    cusparseSpMatDescr_t matA;
    // E is transposed making contiguous memory are edges
    auto A_ptr = E_T.data_ptr<int8_t>();
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, m, k, nnz, indptr.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(), A_ptr, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_8I));
    // Dense vector
    cusparseDnVecDescr_t vecB, vecC;
    // Output
    auto out =
        torch::empty({m, b}, torch::dtype(torch::kFloat32).device(device));
    // auto out_ptr = out.data_ptr<float>();
    // CHECK_CUSPARSE(cusparseCreateDnVec(&vecC, m, out_ptr, CUDA_R_32F));
    // // Node features
    // auto N_ptr = N_T.data_ptr<int8_t>();
    // CHECK_CUSPARSE(cusparseCreateDnVec(&vecB, k, N_ptr, CUDA_R_8I));
    // // workspace
    // size_t workspace_size;
    // CHECK_CUSPARSE(cusparseSpMV_bufferSize(
    //     handle, transA, alpha, matA, vecB, beta, vecC, CUDA_R_32F,
    //     CUSPARSE_SPMV_ALG_DEFAULT, &workspace_size));
    // auto workspace = torch::empty({(long)workspace_size},
    //                               torch::dtype(torch::kInt8).device(device));
    // auto *ws_ptr = (size_t*)workspace.data_ptr<int8_t>();
    // for (int i = 1; i <= b; i++) {
    //     // call SPMM
    //     CHECK_CUSPARSE(cusparseSpMV_bufferSize(
    //         handle, transA, alpha, matA, vecB, beta, vecC, CUDA_R_32F,
    //         CUSPARSE_SPMV_ALG_DEFAULT, ws_ptr));
    //     // move to next block
    //     CHECK_CUSPARSE(cusparseSpMatSetValues(matA, A_ptr + i * nnz));
    //     CHECK_CUSPARSE(cusparseDnVecSetValues(vecB, N_ptr + i * k));
    //     CHECK_CUSPARSE(cusparseDnVecSetValues(vecC, out_ptr + i * k));
    //     kernelErrchk();
    // }
    // CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    // CHECK_CUSPARSE(cusparseDestroyDnVec(vecB));
    // CHECK_CUSPARSE(cusparseDestroyDnVec(vecC));
    // cusparseSetPointerMode(handle, ptr_mode);
    return out;
}

void incidence_SPMM_impl(torch::Tensor &out, const torch::Tensor &src,
                         const torch::Tensor &dst, const torch::Tensor &E,
                         const torch::Tensor &scale) {
    auto handle = at::cuda::getCurrentCUDASparseHandle();
    cusparsePointerMode_t ptr_mode;
    cusparseGetPointerMode(handle, &ptr_mode);

    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);
    float *alpha = scale.data_ptr<float>();
    auto beta_tensor = torch::zeros_like(scale);
    float *beta = beta_tensor.data_ptr<float>();

    const int64_t m = out.size(0);
    const int64_t n = out.size(1);
    const int64_t k = E.size(0);
    const int64_t nnz = src.size(0);
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    // sparse matrix A is all one
    auto A = torch::empty({nnz}, torch::dtype(torch::kInt8).device(E.device()));
    int8_t *A_ptr = A.data_ptr<int8_t>();
    cudaErrchk(cudaMemset(A_ptr, 1, nnz * sizeof(int8_t)));
    CHECK_CUSPARSE(cusparseCreateCoo(
        &matA, m, nnz, nnz, src.data_ptr<int64_t>(), dst.data_ptr<int64_t>(),
        A_ptr, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_8I));
    // Dense matrix
    auto E_ptr = E.data_ptr<int8_t>();
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, nnz, n, n, E_ptr, CUDA_R_8I,
                                       CUSPARSE_ORDER_ROW));
    // Output
    auto out_ptr = out.data_ptr<float>();
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, m, n, n, out_ptr, CUDA_R_32F,
                                       CUSPARSE_ORDER_ROW));
    // call SPMM
    // std::cout << alpha << " " << A_ptr << " " << src.data_ptr<int64_t>() << "
    // " << dst.data_ptr<int64_t>() << " " << out_ptr << "\n";
    CHECK_CUSPARSE(cusparseSpMM(handle, transA, transB, alpha, matA, matB, beta,
                                matC, CUDA_R_32F, CUSPARSE_SPMM_COO_ALG4,
                                nullptr));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    cusparseSetPointerMode(handle, ptr_mode);
}
