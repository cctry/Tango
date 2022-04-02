#define CUTLASS_DEBUG_TRACE_LEVEL 0
#include "QuantizedGemmKernel.h"
#include "epilogue.h"
#include "iterator.h"
#include <cstdint>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_base.h>
#ifndef TEST
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#endif

template <typename ThreadblockShape_, typename WarpShape_,
          typename EpilogueOutputOp, bool isReduce = false>
struct GEMMKernel {
    using ThreadblockShape = ThreadblockShape_;
    using WarpShape = WarpShape_;

    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 4>;
    using OperatorClass = cutlass::arch::OpClassSimt;
    using Operator = cutlass::arch::OpMultiplyAdd;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementAccumulator = int32_t;

    using ThreadblockSwizzle =
        typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    // Define the MmaCore components
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
        ElementB, LayoutB, ElementAccumulator, cutlass::layout::RowMajor,
        OperatorClass, 2, Operator>;

    // Define iterators over tiles from the A operand
    using IteratorA = QuantizeTileIterator2dThreadTile<
        cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>, LayoutA,
        1, typename MmaCore::IteratorThreadMapA>;

    // Define iterators over tiles from the B operand
    using IteratorB = QuantizeTileIterator2dThreadTile<
        cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>, LayoutB,
        0, typename MmaCore::IteratorThreadMapB>;

    // Define the threadblock-scoped pipelined matrix multiply
    using Mma = cutlass::gemm::threadblock::MmaPipelined<
        typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
        IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
        cutlass::layout::RowMajor, typename MmaCore::MmaPolicy>;

    using Epilogue = typename cutlass::platform::conditional<
        isReduce,
        typename EpilogueHelper<ThreadblockShape, typename Mma::Operator,
                                EpilogueOutputOp,
                                EpilogueOutputOp::kCount>::Epilogue,
        typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
            ThreadblockShape, typename Mma::Operator, EpilogueOutputOp,
            EpilogueOutputOp::kCount>::Epilogue>::type;

    using Kernel = QuantizeGemm<Mma, Epilogue, ThreadblockSwizzle>;
};

constexpr auto kMode = cutlass::gemm::GemmUniversalMode::kGemm;
#define run_gemm(isReduce, isBias)                                             \
    do {                                                                       \
        using Gemm = Gemm_<isReduce, isBias>;                                  \
        typename Gemm::Arguments arguments(                                    \
            kMode, size, 1, {scaleX, scaleW, scaleY}, X, X_q, scaleX, W, W_q,  \
            scaleW, bias, Y, M *K, K *N, N, M *N, ldA, ldB, 0, ldC);           \
        Gemm op;                                                               \
        check_cutlass_err(op.initialize(arguments, nullptr));                  \
        check_cutlass_err(op());                                               \
    } while (0)

template <template <bool, bool> class Gemm_>
void QGEMM_forward(float *Y, float *X, int8_t *X_q, float *W, int8_t *W_q,
                   int M, int N, int K, int ldA, int ldB, int ldC,
                   float *scaleX, float *scaleW, float *scaleY, float *bias) {
    cutlass::gemm::GemmCoord size(M, N, K);
    kernelErrchk();
    if (scaleY && bias) {
        run_gemm(true, true);
    } else if (scaleY && !bias) {
        run_gemm(true, false);
    } else if (!scaleY && bias) {
        run_gemm(false, true);
    } else {
        run_gemm(false, false);
    }
    kernelErrchk();
}

template <bool isReduce, bool hasBias>
using GemmN256 = cutlass::gemm::device::GemmUniversalBase<
    typename GEMMKernel<cutlass::gemm::GemmShape<128, 256, 32>,
                        cutlass::gemm::GemmShape<64, 64, 32>,
                        Dequant<float, int32_t, hasBias>, isReduce>::Kernel>;

template <bool isReduce, bool hasBias>
using GemmN64 = cutlass::gemm::device::GemmUniversalBase<typename GEMMKernel<
    cutlass::gemm::GemmShape<128, 64, 32>, cutlass::gemm::GemmShape<32, 64, 32>,
    Dequant<float, int32_t, hasBias>, isReduce>::Kernel>;

template <bool isReduce, bool hasBias>
using GemmN128Large = cutlass::gemm::device::GemmUniversalBase<
    typename GEMMKernel<cutlass::gemm::GemmShape<256, 128, 32>,
                        cutlass::gemm::GemmShape<64, 64, 32>,
                        Dequant<float, int32_t, hasBias>, isReduce>::Kernel>;

template <bool isReduce, bool hasBias>
using GemmN128Small = cutlass::gemm::device::GemmUniversalBase<
    typename GEMMKernel<cutlass::gemm::GemmShape<128, 128, 32>,
                        cutlass::gemm::GemmShape<32, 64, 32>,
                        Dequant<float, int32_t, hasBias>, isReduce>::Kernel>;

#define call_gemm(kernel)                                                      \
    QGEMM_forward<kernel>(Y_ptr, X_ptr, X_q_ptr, W_ptr, W_q_ptr, M, N, K, ldA, \
                          ldB, ldC, scaleX_ptr, scaleW_ptr, scaleY_ptr,        \
                          bias_ptr)

#ifndef TEST
void QGEMM_forward_impl(torch::Tensor &Y, torch::Tensor &X, torch::Tensor &X_q,
                        torch::Tensor &W, torch::Tensor &W_q,
                        torch::Tensor &bias, torch::Tensor &scaleY,
                        torch::Tensor &scaleX, torch::Tensor &scaleW) {
    int M = X.size(0);
    int N = W.size(0);
    int K = W.size(1);
    int ldA = X.stride(0);
    int ldB = W.stride(0);
    int ldC = Y.stride(0);
    auto Y_ptr = Y.data_ptr<float>();
    auto X_ptr = X.data_ptr<float>();
    auto X_q_ptr = X_q.data_ptr<int8_t>();
    auto W_ptr = W.data_ptr<float>();
    auto W_q_ptr = W_q.data_ptr<int8_t>();
    auto scaleX_ptr = scaleX.data_ptr<float>();
    auto scaleW_ptr = scaleW.data_ptr<float>();
    float *bias_ptr = bias.numel() ? bias.data_ptr<float>() : nullptr;
    float *scaleY_ptr = scaleY.numel() ? scaleY.data_ptr<float>() : nullptr;
    kernelErrchk();
    if (N >= 256) {
        call_gemm(GemmN256);
    } else if (N <= 64) {
        call_gemm(GemmN64);
    } else {
        if (M < 16384) {
            call_gemm(GemmN128Small);
        } else {
            call_gemm(GemmN128Large);
        }
    }
    kernelErrchk();
}
#else
#include <bits/stdc++.h>
#include <cublas_v2.h>

template <typename T>
__global__ void InitializeMatrix_kernel(T *matrix, int rows, int columns,
                                        int seed = 0) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < rows && j < columns) {
        int offset = i + j * rows;

        // Generate arbitrary elements.
        int const k = 16807;
        int const m = 16;
        T value = T(1);
        if (seed != 0)
            value = T(((offset + seed) * k % m) - m / 2);

        matrix[offset] = value;
    }
}

template <typename T>
/// Simple function to initialize a matrix to arbitrary small integers.
void InitializeMatrix(T *matrix, int rows, int columns, int seed = 0) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x,
              (columns + block.y - 1) / block.y);
    InitializeMatrix_kernel<<<grid, block>>>(matrix, rows, columns, seed);
}

int main(int ac, char **av) {
    int M = atoi(av[1]);
    int N = atoi(av[2]);
    int K = atoi(av[3]);
    int8_t *A, *B;
    float *C, *C_ref, *scales;
    cudaMallocManaged(&A, M * K * sizeof(int8_t));
    cudaMallocManaged(&B, K * N * sizeof(int8_t));
    InitializeMatrix(A, M, K, 41);
    InitializeMatrix(B, N, K, 17);
    cudaMallocManaged(&C, M * N * sizeof(float));
    cudaMallocManaged(&C_ref, M * N * sizeof(float));
    cudaMemset(C, 0, M * N * sizeof(float));
    cudaMemset(C_ref, 0, M * N * sizeof(float));
    cudaMallocManaged(&scales, 3 * sizeof(float));
    scales[0] = 0.5f;
    scales[1] = 0.5f;
    QGEMM_forward<GemmLarge>(C, A, B, M, N, K, K, K, N, scales, scales + 1,
                             scales + 2);
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 0.25f;
    float beta = 0.0f;
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B,
                 CUDA_R_8I, K, A, CUDA_R_8I, K, &beta, C_ref, CUDA_R_32F, N,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < N; c++) {
            if (std::abs(C_ref[r * N + c] - C[r * N + c]) > 1e-3) {
                std::cout << "Error at " << r << "," << c << ": "
                          << C_ref[r * N + c] << " vs " << C[r * N + c] << "\n";
                return -1;
            }
        }
    }
    std::cout << "Success\n";
}
#endif