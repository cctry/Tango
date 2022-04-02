#pragma once
#include <stdio.h>

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

#define check_cutlass_err(status)                                              \
    do {                                                                       \
        if (status != cutlass::Status::kSuccess) {                             \
            fprintf(stderr, "cutlass error: %s %s %d\n",                       \
                    cutlassGetStatusString(status), __FILE__, __LINE__);       \
            gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__);            \
            exit(-1);                                                          \
        }                                                                      \
    } while (0)

#define check_cublas_err(err)                                                  \
    do {                                                                       \
        if (err != CUBLAS_STATUS_SUCCESS) {                                    \
            printf("cublas error: %d\n", err);                                 \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define kernelErrchk()                                                         \
    { gpuAssert((cudaGetLastError()), __FILE__, __LINE__); }
#define cudaErrchk(err)                                                        \
    { gpuAssert(err, __FILE__, __LINE__); }

static inline int findThreadNum(int dim) {
    int ret = 1024;
    while (ret > dim) {
        ret = ret >> 1;
    }
    return ret;
}

__device__ __forceinline__ float atomicMax(float *address, float val) {
    int *address_as_int = (int *)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        int update = __float_as_int(max(val, __int_as_float(assumed)));
        old = atomicCAS(address_as_int, assumed, update);
    } while (assumed != old);
    return __int_as_float(old);
}

template <typename T> __device__ T warp_max(T val) {
#pragma unroll
    for (int i = 1; i < 32; i *= 2)
        val = max(val, __shfl_xor_sync(0xffffffff, val, i));
    return val;
}

/**
 * \brief Returns the warp lane ID of the calling thread
 */
__device__ __forceinline__ unsigned int Lane_id() {
    unsigned int ret;
    asm("mov.u32 %0, %%laneid;" : "=r"(ret));
    return ret;
}

/**
 * \brief Returns the warp ID of the calling thread.  Warp ID is guaranteed to
 * be unique among warps, but may not correspond to a zero-based ranking within
 * the thread block.
 */
__device__ __forceinline__ unsigned int Warp_id() {
    unsigned int ret;
    asm("mov.u32 %0, %%warpid;" : "=r"(ret));
    return ret;
}

__device__ __forceinline__ void scale_reduce(float thd_max, float *out_ptr) {
    __shared__ float smem[32];
    thd_max = warp_max(thd_max);
    int num_warp = (blockDim.x * blockDim.y * blockDim.z) / 32;
    int warp_id = Warp_id();
    int lane_id = Lane_id();
    if (lane_id == 0)
        smem[warp_id] = thd_max;
    __syncthreads();
    if (warp_id == 0) {
        float global_max = lane_id < num_warp ? smem[lane_id] : 0;
        global_max = warp_max(global_max);
        if (lane_id == 0) {
            atomicMax(out_ptr, global_max);
        }
    }
}

__device__ __forceinline__ int32_t idp4a(char4 const a, char4 const b,
                                         int32_t C) {
    return C + a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
