#include "utils.h"
#ifndef TEST
#include <torch/extension.h>
#else
#include <bits/stdc++.h>
#endif

template <bool isReduce = false>
__global__ void kernel_SDDMM_dot_int8(
    float *__restrict__ out, float *__restrict__ scaleOut,
    const int64_t *__restrict__ src, const int64_t *__restrict__ dst,
    const int8_t *__restrict__ U, const int8_t *__restrict__ V,
    const float *__restrict__ scaleU, const float *__restrict__ scaleV,
    int64_t nEdge, int64_t out_len, int64_t dim) {
    auto ty = blockIdx.x * blockDim.y + threadIdx.y;
    float scale = __ldg(scaleU) * __ldg(scaleV);
    float reduce_temp = 0;
    if (ty < nEdge) {
        const int64_t src_id = __ldg(src + ty);
        const int64_t dst_id = __ldg(dst + ty);
        const auto *lhsoff = U + src_id * dim * out_len;
        const auto *rhsoff = V + dst_id * dim * out_len;
        auto *outoff = out + ty * out_len;
        int tx = threadIdx.x;
        for (int i = blockIdx.y; i < out_len; i += gridDim.y) {
            int32_t val = 0;
            for (int j = tx; j < dim; j += 64) {
                val += lhsoff[i * dim + j] * rhsoff[i * dim + j];
                if (j + 32 < dim) {
                    val += lhsoff[i * dim + j + 32] * rhsoff[i * dim + j + 32];
                }
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
                val += __shfl_down_sync(0xFFFFFFFF, val, offset);
            if (tx == 0) {
                float temp = (float)val * scale;
                __stcs(outoff + i, temp);
                if (isReduce)
                    reduce_temp = max(reduce_temp, fabs(temp));
            }
        }
    }
    if (isReduce) {
        scale_reduce(reduce_temp / 127.0f, scaleOut);
    }
}

struct Add {
    __host__ __device__ float operator()(float a, float b) { return a + b; }
};

struct Sub {
    __host__ __device__ float operator()(float a, float b) { return a - b; }
};

struct Mul {
    __host__ __device__ float operator()(float a, float b) { return a * b; }
};

struct Div {
    __host__ __device__ float operator()(float a, float b) { return a / b; }
};

template <typename OP, bool isReduce = false>
__global__ __launch_bounds__(1024) void kernel_SDDMM_int8(
    float *__restrict__ out, float *__restrict__ scaleOut,
    const int64_t *__restrict__ src, const int64_t *__restrict__ dst,
    const int8_t *__restrict__ U, const int8_t *__restrict__ V,
    const float *__restrict__ scaleU, const float *__restrict__ scaleV,
    int64_t nEdge, int64_t dim) {
    float scaleU_ = __ldg(scaleU);
    float scaleV_ = __ldg(scaleV);
    float reduce_temp = 0;
    OP op;
    auto ty = blockIdx.y * blockDim.y + threadIdx.y;
    const auto stride_y = blockDim.y * gridDim.y;
    while (ty < nEdge) {
        const auto src_id = __ldg(src + ty);
        const auto dst_id = __ldg(dst + ty);
        const auto eid = ty;
        const auto *lhsoff = U + src_id * dim;
        const auto *rhsoff = V + dst_id * dim;
        auto *outoff = out + eid * dim;
        int tx = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride_x = blockDim.x * gridDim.x;
        while (tx < dim) {
            float val =
                op((float)lhsoff[tx] * scaleU_, (float)rhsoff[tx] * scaleV_);
            if (isReduce)
                reduce_temp = max(reduce_temp, fabs(val));
            outoff[tx] = val;
            tx += stride_x;
        }
        ty += stride_y;
    }
    if (isReduce) {
        scale_reduce(reduce_temp / 127.0f, scaleOut);
    }
}

#ifndef TEST

#define CALL_SDDMM(op, reduce, out, scaleOut, src, dst, U, V, scaleU, scaleV,  \
                   nEdge, dim)                                                 \
    do {                                                                       \
        if (reduce) {                                                          \
            kernel_SDDMM_int8<op, true><<<nblks, nthrs>>>(                     \
                out.data_ptr<float>(), scaleOut.data_ptr<float>(),             \
                src.data_ptr<int64_t>(), dst.data_ptr<int64_t>(),              \
                U.data_ptr<int8_t>(), V.data_ptr<int8_t>(),                    \
                scaleU.data_ptr<float>(), scaleV.data_ptr<float>(), nEdge,     \
                dim);                                                          \
        } else {                                                               \
            kernel_SDDMM_int8<op, false><<<nblks, nthrs>>>(                    \
                out.data_ptr<float>(), scaleOut.data_ptr<float>(),             \
                src.data_ptr<int64_t>(), dst.data_ptr<int64_t>(),              \
                U.data_ptr<int8_t>(), V.data_ptr<int8_t>(),                    \
                scaleU.data_ptr<float>(), scaleV.data_ptr<float>(), nEdge,     \
                dim);                                                          \
        }                                                                      \
    } while (0)

void mySDDMM_int8_impl(std::string &op, torch::Tensor &out,
                       torch::Tensor &scaleOut, torch::Tensor &src,
                       torch::Tensor &dst, torch::Tensor &U, torch::Tensor &V,
                       torch::Tensor &scaleU, torch::Tensor &scaleV) {
    bool isReduce = scaleOut.numel() != 0;
    int64_t out_len = out.size(1);
    if (op == "dot") {
        const int ntx = 32; // on feature dimension
        const int nty = 8;  // on out dimension
        const int nbx = (src.numel() + nty - 1) / nty;
        const int nby = 1; // out_len;
        const dim3 nblks(nbx, nby);
        const dim3 nthrs(ntx, nty);
        if (isReduce) {
            kernel_SDDMM_dot_int8<true><<<nblks, nthrs>>>(
                out.data_ptr<float>(), scaleOut.data_ptr<float>(),
                src.data_ptr<int64_t>(), dst.data_ptr<int64_t>(),
                U.data_ptr<int8_t>(), V.data_ptr<int8_t>(),
                scaleU.data_ptr<float>(), scaleV.data_ptr<float>(), src.numel(),
                out_len, U.size(-1));
        } else {
            kernel_SDDMM_dot_int8<false><<<nblks, nthrs>>>(
                out.data_ptr<float>(), nullptr, src.data_ptr<int64_t>(),
                dst.data_ptr<int64_t>(), U.data_ptr<int8_t>(),
                V.data_ptr<int8_t>(), scaleU.data_ptr<float>(),
                scaleV.data_ptr<float>(), src.numel(), out_len, U.size(-1));
        }
    } else {
        int dim = U.size(-1);
        if (U.dim() == 3)
            dim = U.size(-2) * U.size(-1);

        const int ntx = findThreadNum(dim);
        const int nty = 1024 / ntx;
        const int nbx = (dim + ntx - 1) / ntx;
        const int nby = std::min(65535L, (src.numel() + nty - 1) / nty);
        const dim3 nblks(nbx, nby);
        const dim3 nthrs(ntx, nty);
        if (op == "add") {
            CALL_SDDMM(Add, isReduce, out, scaleOut, src, dst, U, V, scaleU,
                       scaleV, src.numel(), dim);
        } else if (op == "sub") {
            CALL_SDDMM(Sub, isReduce, out, scaleOut, src, dst, U, V, scaleU,
                       scaleV, src.numel(), dim);
        } else if (op == "mul") {
            CALL_SDDMM(Mul, isReduce, out, scaleOut, src, dst, U, V, scaleU,
                       scaleV, src.numel(), dim);
        } else if (op == "div") {
            CALL_SDDMM(Div, isReduce, out, scaleOut, src, dst, U, V, scaleU,
                       scaleV, src.numel(), dim);
        } else
            printf("%s op not supported\n", op.c_str());
    }
    kernelErrchk();
}

#endif

#ifdef TEST
// int main(int argc, char **argv) {
//     int64_t *src, *dst;
//     int8_t *U, *V;
//     float *scaleU, *scaleV;
//     float *out;
//     const int dim = atoi(argv[1]);
//     const int out_len = atoi(argv[2]);
//     constexpr int nEdge = 4;
//     constexpr int nVertex = 3;
//     cudaMallocManaged(&src, sizeof(int64_t) * nEdge);
//     cudaMallocManaged(&dst, sizeof(int64_t) * nEdge);
//     cudaMallocManaged(&U, sizeof(int8_t) * nVertex * dim * out_len);
//     cudaMallocManaged(&V, sizeof(int8_t) * nVertex * dim * out_len);
//     cudaMallocManaged(&scaleU, sizeof(float) * 1);
//     cudaMallocManaged(&scaleV, sizeof(float) * 1);
//     cudaMallocManaged(&out, sizeof(float) * nEdge * out_len);
//     src[0] = 0;
//     dst[0] = 0;
//     src[1] = 0;
//     dst[1] = 1;
//     src[2] = 0;
//     dst[2] = 2;
//     src[3] = 1;
//     dst[3] = 2;
//     for (int i = 0; i < dim * nVertex * out_len; i++) {
//         U[i] = i;
//         V[i] = (dim * nVertex * out_len - i);
//     }
//     scaleU[0] = 1.0;
//     scaleV[0] = 1.0;
//     const int ntx = 32; // on feature dimension
//     const int nty = 8;  // on out dimension
//     const int nbx = (nEdge + nty - 1) / nty;
//     const int nby = out_len;
//     const dim3 nblks(nbx, nby);
//     const dim3 nthrs(ntx, nty);
//     kernel_SDDMM_dot_int8<><<<nblks, nthrs>>>(
//         out, nullptr, src, dst, U, V, scaleU, scaleV, nEdge, out_len, dim);
//     cudaDeviceSynchronize();
//     for (int i = 0; i < nEdge; i++) {
//         for (int j = 0; j < out_len; j++) {
//             printf("%.3f ", out[i * out_len + j]);
//         }
//         printf("\n");
//     }
// }

int main(int argc, char **argv) {
    int64_t *src, *dst;
    int8_t *U, *V;
    float *scaleU, *scaleV;
    float *out;
    const int dim = atoi(argv[1]);
    constexpr int nEdge = 4;
    constexpr int nVertex = 3;
    cudaMallocManaged(&src, sizeof(int64_t) * nEdge);
    cudaMallocManaged(&dst, sizeof(int64_t) * nEdge);
    cudaMallocManaged(&U, sizeof(int8_t) * nVertex * dim);
    cudaMallocManaged(&V, sizeof(int8_t) * nVertex * dim);
    cudaMallocManaged(&scaleU, sizeof(float) * 1);
    cudaMallocManaged(&scaleV, sizeof(float) * 1);
    cudaMallocManaged(&out, sizeof(float) * nEdge * dim);
    src[0] = 0;
    dst[0] = 0;
    src[1] = 0;
    dst[1] = 1;
    src[2] = 0;
    dst[2] = 2;
    src[3] = 1;
    dst[3] = 2;
    for (int i = 0; i < dim * nVertex; i++) {
        U[i] = i;
        V[i] = (dim * nVertex - i);
    }
    scaleU[0] = 1.0;
    scaleV[0] = 1.0;
    const int ntx = std::min(1024, findThreadNum(dim));
    const int nty = 1024 / ntx;
    const int nbx = (dim + ntx - 1) / ntx;
    const int nby = std::min(65535L, (long)(nEdge + nty - 1) / nty);
    const dim3 nblks(nbx, nby);
    const dim3 nthrs(ntx, nty);

    kernel_SDDMM_int8<Add, false><<<nblks, nthrs>>>(
        out, nullptr, src, dst, U, V, scaleU, scaleV, nEdge, dim);
    cudaDeviceSynchronize();
    for (int i = 0; i < nEdge; i++) {
        for (int j = 0; j < dim; j++) {
            printf("%.3f ", out[i * dim + j]);
        }
        printf("\n");
    }
}

#endif