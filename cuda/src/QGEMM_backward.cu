#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "epilogue.h"
#include "utils.h"
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cstdint>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <torch/extension.h>

using config =
    cutlass::gemm::device::DefaultGemmConfiguration<cutlass::arch::OpClassSimt,
                                                    cutlass::arch::Sm70, int8_t,
                                                    int8_t, float, int32_t>;

#define get_ref(t, type, l)                                                    \
    cutlass::make_TensorRef(t.data_ptr<type>(),                                \
                            cutlass::layout::l##Major(t.stride(0)))

void QGEMM_backward_gradX_impl(torch::Tensor &dX, torch::Tensor &dY,
                               torch::Tensor &W, torch::Tensor &scaledY,
                               torch::Tensor &scaleW) {
    using Gemm_gradX = cutlass::gemm::device::Gemm<
        int8_t,                     // ElementA
        cutlass::layout::RowMajor,  // LayoutA
        int8_t,                     // ElementB
        cutlass::layout::RowMajor,  // LayoutB
        float,                      // ElementOutput
        cutlass::layout::RowMajor,  // LayoutOutput
        int32_t,                    // ElementAccumulator
        cutlass::arch::OpClassSimt, // tag indicating Tensor Cores
        cutlass::arch::Sm70, // tag indicating target GPU compute architecture
        typename config::ThreadblockShape, // cutlass::gemm::GemmShape<128, 128,
                                           // 32>,
        typename config::WarpShape, // cutlass::gemm::GemmShape<32, 64, 32>,
        typename config::InstructionShape, // cutlass::gemm::GemmShape<1, 1, 4>,
        Dequant<float, int32_t>>;
    Gemm_gradX gemm_op;
    cutlass::Status status = gemm_op(
        {{dX.size(0), dX.size(1), dY.size(1)},
         {dY.data_ptr<int8_t>(), dY.stride(0)},
         {W.data_ptr<int8_t>(), W.stride(0)},
         {dX.data_ptr<float>(), dX.stride(0)},
         {dX.data_ptr<float>(), dX.stride(0)},
         {scaledY.data_ptr<float>(), scaleW.data_ptr<float>(), nullptr}});
    check_cutlass_err(status);
}

void QGEMM_backward_gradW_impl(torch::Tensor &dW, torch::Tensor &dY,
                               torch::Tensor &X, torch::Tensor &scaledY,
                               torch::Tensor &scaleX) {
    cutlass::Status status;
    cutlass::gemm::GemmCoord gemm_size(dW.size(0), dW.size(1), dY.size(0));
    auto A_ref = get_ref(dY, int8_t, Column);
    auto B_ref = get_ref(X, int8_t, Row);
    auto C_ref = get_ref(dW, float, Row);
    typename Dequant<float, int32_t>::Params dequant_ptrs(
        scaledY.data_ptr<float>(), scaleX.data_ptr<float>(), nullptr);
    if (gemm_size.k() > 4 * gemm_size.m()) { // splitK
        using Gemm_gradW_SplitK = cutlass::gemm::device::GemmSplitKParallel<
            int8_t,                       // ElementA
            cutlass::layout::ColumnMajor, // LayoutA
            int8_t,                       // ElementB
            cutlass::layout::RowMajor,    // LayoutB
            float,                        // ElementOutput
            cutlass::layout::RowMajor,    // LayoutOutput
            int32_t,                      // ElementAccumulator
            cutlass::arch::OpClassSimt,   // tag indicating non-Tensor Cores
            cutlass::arch::Sm70,          // tag indicating target GPU compute
                                          // architecture
            typename config::ThreadblockShape, // cutlass::gemm::GemmShape<128,
                                               // 128, 32>,
            typename config::WarpShape, // cutlass::gemm::GemmShape<32, 64, 32>,
            typename config::InstructionShape, // cutlass::gemm::GemmShape<1, 1,
                                               // 4>,
            Dequant<float, int32_t>>;
        Gemm_gradW_SplitK op;
        int split_k_slices = std::min(256, gemm_size.k() / 256);
        typename Gemm_gradW_SplitK::Arguments args{
            gemm_size, A_ref,        B_ref,         C_ref,
            C_ref,     dequant_ptrs, split_k_slices};
        int64_t workspace_size = Gemm_gradW_SplitK::get_workspace_size(args);
        // auto options = torch::TensorOptions()
        //                    .dtype(torch::kInt8)
        //                    .device(X.device().type(), X.device().index());
        auto workspace = torch::empty({workspace_size}, torch::dtype(torch::kInt8).device(X.device()));
        status = op.initialize(args, workspace.data_ptr<int8_t>());
        check_cutlass_err(status);
        status = op();
    } else {
        using Gemm_gradW = cutlass::gemm::device::Gemm<
            int8_t,                       // ElementA
            cutlass::layout::ColumnMajor, // LayoutA
            int8_t,                       // ElementB
            cutlass::layout::RowMajor,    // LayoutB
            float,                        // ElementOutput
            cutlass::layout::RowMajor,    // LayoutOutput
            int32_t,                      // ElementAccumulator
            cutlass::arch::OpClassSimt,   // tag indicating non-Tensor Cores
            cutlass::arch::Sm70,          // tag indicating target GPU compute
                                          // architecture
            typename config::ThreadblockShape, // cutlass::gemm::GemmShape<128,
                                               // 128, 32>,
            typename config::WarpShape, // cutlass::gemm::GemmShape<32, 64, 32>,
            typename config::InstructionShape, // cutlass::gemm::GemmShape<1, 1,
                                               // 4>,
            Dequant<float, int32_t>>;
        Gemm_gradW op;
        status = op({gemm_size, A_ref, B_ref, C_ref, C_ref, dequant_ptrs});
    }
    check_cutlass_err(status);
}