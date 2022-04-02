#include "utils.h"
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <torch/extension.h>

torch::Tensor gemm(const torch::Tensor &A, const torch::Tensor &B, bool trans_A,
                   bool trans_B) {
    const int m = trans_A ? A.size(1) : A.size(0);
    const int n = trans_B ? B.size(0) : B.size(1);
    const int k = trans_A ? A.size(0) : A.size(1);
    const int8_t *A_ptr = A.data_ptr<int8_t>();
    const int8_t *B_ptr = B.data_ptr<int8_t>();
    const int lda = A.stride(0);
    const int ldb = B.stride(0);
    // Allocate output tensor
    auto options = torch::TensorOptions()
                       .dtype(torch::kInt32)
                       .device(A.device().type(), A.device().index())
                       .requires_grad(A.requires_grad() || B.requires_grad());
    auto C = torch::empty({m, n}, options);
    int32_t *C_ptr = C.data_ptr<int32_t>();
    const int ldc = C.stride(0);
    // cublas
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    const auto OP_A = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto OP_B = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;
    const static int32_t beta = 0;
    const static int32_t alpha = 1;
    check_cublas_err(cublasGemmEx(handle, OP_B, OP_A, n, m, k, &alpha, B_ptr,
                                  CUDA_R_8I, ldb, A_ptr, CUDA_R_8I, lda, &beta,
                                  C_ptr, CUDA_R_32I, ldc, CUDA_R_32I,
                                  CUBLAS_GEMM_DEFAULT));
    return C;
}

inline __device__ char clamp(int x) { return (char)max(-127, min(x, 127)); }
inline __device__ char4 quantize(float4 x, float scale) {
    char4 res;
    res.x = clamp(int(rintf(x.x * scale)));
    res.y = clamp(int(rintf(x.y * scale)));
    res.z = clamp(int(rintf(x.z * scale)));
    res.w = clamp(int(rintf(x.w * scale)));
    return res;
}
__global__ void __kernel_quantize(char *__restrict__ out,
                                  float *__restrict__ _scale,
                                  const float *__restrict__ in,
                                  const float *__restrict__ _min_val,
                                  const float *__restrict__ _max_val,
                                  int64_t size) {

    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto nthd = blockDim.x * gridDim.x;

    const float min_val = _min_val[0];
    const float max_val = _max_val[0];
    const float threshold = max(fabsf(min_val), fabsf(max_val));
    const float scale_val = threshold / 127.0f;
    float div_scale = 1.0f / scale_val;

    auto size4 = size / 4;
    for (auto i = tid; i < size4; i += nthd) {
        float4 val = __ldcs((float4 *)&in[i * 4]);
        char4 res = quantize(val, div_scale);
        __stcs((char4 *)&out[i * 4], res);
    }
    int remain = size % 4;
    if (remain > 0 && tid < remain) {
        float val = in[size4 * 4 + tid + 1];
        char res = clamp(int(rintf(val * div_scale)));
        out[size4 * 4 + tid + 1] = res;
    }

    if (tid == 0) {
        _scale[0] = scale_val;
    }
}

torch::Tensor quantize(const torch::Tensor &X, float *scale) {
    auto options_out = torch::TensorOptions()
                           .dtype(torch::kInt8)
                           .device(X.device().type(), X.device().index());
    torch::Tensor out = torch::empty_like(X, options_out);
    auto min_val = X.min();
    auto max_val = X.max();
    int nblk, nthd;
    cudaOccupancyMaxPotentialBlockSize(&nblk, &nthd, __kernel_quantize);
    auto out_ptr = (char *)out.data_ptr<int8_t>();
    __kernel_quantize<<<nblk, nthd>>>(out_ptr, scale, X.data_ptr<float>(),
                                      min_val.data_ptr<float>(),
                                      max_val.data_ptr<float>(), X.numel());
    return out;
}

__global__ void __kernel_dequantize(float *__restrict__ out,
                                    const float *__restrict__ _scale,
                                    const int32_t *__restrict__ in,
                                    int64_t size) {
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    const auto nthd = blockDim.x * gridDim.x;
    const float scale = _scale[0] * _scale[1];
    for (auto i = tid; i < size; i += nthd) {
        out[i] = static_cast<float>(in[i]) * scale;
    }
}

torch::Tensor Qgemm_impl(const torch::Tensor &A, const torch::Tensor &B,
                         bool trans_A, bool trans_B) {
    float *scales;
    cudaMalloc(&scales, sizeof(float) * 2);
    auto A_ = quantize(A, &scales[0]);
    auto B_ = quantize(B, &scales[1]);
    auto C_ = gemm(A_, B_, trans_A, trans_B);
    // Allocate output tensor
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(A.device().type(), A.device().index())
                       .requires_grad(A.requires_grad() || B.requires_grad());
    auto C = torch::empty_like(C_, options);
    int nblk, nthd;
    cudaOccupancyMaxPotentialBlockSize(&nblk, &nthd, __kernel_dequantize);
    __kernel_dequantize<<<nblk, nthd>>>(C.data_ptr<float>(), scales,
                                        C_.data_ptr<int32_t>(), C_.numel());
    return C;
}
