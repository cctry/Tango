#include "utils.h"
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <torch/extension.h>

inline __device__ char clamp(int x) { return (char)max(-127, min(x, 127)); }

inline __device__ char4 quantize(float4 x, float scale) {
    char4 res;
    res.x = clamp(int(rintf(x.x * scale)));
    res.y = clamp(int(rintf(x.y * scale)));
    res.z = clamp(int(rintf(x.z * scale)));
    res.w = clamp(int(rintf(x.w * scale)));
    return res;
}

__global__ void kernel_quantize(int8_t *__restrict__ out,
                                const float *__restrict__ in,
                                const float *__restrict__ scale, int64_t size) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto nthd = blockDim.x * gridDim.x;
    float div_scale = 1.0f / __ldg(scale);
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
}

__global__ void kernel_get_scale(float *__restrict__ out,
                                 const float *__restrict__ in, int64_t size) {
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    const auto nthd = blockDim.x * gridDim.x;
    int size4 = size / 4;
    float4 *in4 = (float4 *)in;
    float res = 0;
    for (auto i = tid; i < size4; i += nthd) {
        float4 temp = __ldcs(&in4[i]);
        res = max(res, fabs(temp.x));
        res = max(res, fabs(temp.y));
        res = max(res, fabs(temp.z));
        res = max(res, fabs(temp.w));
    }
    int remain = size % 4;
    if (remain > 0 && tid < remain) {
        res = max(res, fabs(in[size4 * 4 + tid + 1]));
    }
    scale_reduce(res / 127.0f, out);
}

void quantize_impl(torch::Tensor &out, torch::Tensor &in,
                   torch::Tensor &scale) {
    kernel_quantize<<<12 * 80, 128>>>(out.data_ptr<int8_t>(),
                                      in.data_ptr<float>(),
                                      scale.data_ptr<float>(), in.numel());
}

torch::Tensor get_scale_impl(torch::Tensor &in) {
    auto scale = torch::zeros({1}, torch::dtype(torch::kFloat32).device(in.device()));
    // get scale: t/127
    kernel_get_scale<<<12 * 80, 128>>>(scale.data_ptr<float>(),
                                       in.data_ptr<float>(), in.numel());
    return scale;
}

template <int TILE_X, int TILE_Y, int nthd>
__global__ void kernel_transpose_quant(int8_t *__restrict__ dst,
                                       const float *__restrict__ src,
                                       const float *__restrict__ scale_,
                                       int64_t nrow, int64_t ncol) {
    constexpr int PADDING = 4;
    __shared__ int8_t smem_tile[TILE_X][TILE_Y + PADDING];
    int row_offset = blockIdx.x * TILE_X;
    int col_offset = blockIdx.y * TILE_Y;
    int remain_row = nrow - row_offset;
    int remain_col = ncol - col_offset;
    if (remain_col <= 0 || remain_row <= 0)
        return;
    float scale = 1.0f / (*scale_);
    auto src_ptr = src + row_offset * ncol + col_offset;
#pragma unroll(TILE_X *TILE_Y / nthd)
    for (int i = threadIdx.x; i < TILE_X * TILE_Y; i += nthd) {
        int r = i / TILE_Y;
        int c = i % TILE_Y;
        if (r < remain_row && c < remain_col) {
            float temp = src_ptr[r * ncol + c];
            smem_tile[r][c] = rintf(temp * scale);
        }
    }
    __syncthreads();
    auto dst_ptr = dst + col_offset * nrow + row_offset;
#pragma unroll(TILE_X *TILE_Y / nthd)
    for (int i = threadIdx.x; i < TILE_X * TILE_Y; i += nthd) {
        int r = i / TILE_X;
        int c = i % TILE_X;
        if (r < remain_col && c < remain_row) {
            dst_ptr[r * nrow + c] = smem_tile[c][r];
        }
    }
}

#define call_transpose_quant(n)                                                \
    {                                                                          \
        constexpr int TILE_Y = n;                                              \
        constexpr int TILE_X = 8192 / n;                                       \
        constexpr int thd = 256;                                               \
        dim3 block((nrow + TILE_X - 1) / TILE_X,                               \
                   (ncol + TILE_Y - 1) / TILE_Y);                              \
        kernel_transpose_quant<TILE_X, TILE_Y, thd>                            \
            <<<block, thd>>>(out_ptr, in_ptr, scale_ptr, nrow, ncol);          \        
    }


torch::Tensor transpose_quant_impl(torch::Tensor &in, torch::Tensor &scale,
                                   int64_t nrow, int64_t ncol) {
    const auto &device = in.device();
    auto out = torch::empty({ncol, nrow}, torch::dtype(torch::kInt8).device(device));
    auto out_ptr = out.data_ptr<int8_t>();
    auto in_ptr = in.data_ptr<float>();
    auto scale_ptr = scale.data_ptr<float>();
    if (ncol >= 64) {
        call_transpose_quant(64);
    } else if (ncol >= 32) {
        call_transpose_quant(32);
    } else if (ncol >= 16) {
        call_transpose_quant(16);
    } else if (ncol >= 8) {
        call_transpose_quant(8);
    } else if (ncol >= 4) {
        call_transpose_quant(4);
    } else {
        call_transpose_quant(2);
    }
    kernelErrchk();
    return out;
}