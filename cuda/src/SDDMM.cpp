#include <torch/extension.h>

void mySDDMM_impl(torch::Tensor &out, const torch::Tensor &src,
                  const torch::Tensor &dst, const torch::Tensor &U,
                  const torch::Tensor &V, const torch::Tensor &scaleU,
                  const torch::Tensor &scaleV);
void mySDDMM(torch::Tensor &out, const torch::Tensor &src,
             const torch::Tensor &dst, const torch::Tensor &U,
             const torch::Tensor &V, const torch::Tensor &scaleU,
             const torch::Tensor &scaleV) {
    mySDDMM_impl(out, src, dst, U, V, scaleU, scaleV);
}

void mySDDMM_impl_int8(torch::Tensor &out, const torch::Tensor &src,
                  const torch::Tensor &dst, const torch::Tensor &U,
                  const torch::Tensor &V, const torch::Tensor &scaleU,
                  const torch::Tensor &scaleV);


void mySDDMM_int8(torch::Tensor &out, const torch::Tensor &src,
             const torch::Tensor &dst, const torch::Tensor &U,
             const torch::Tensor &V, const torch::Tensor &scaleU,
             const torch::Tensor &scaleV) {
    mySDDMM_impl_int8(out, src, dst, U, V, scaleU, scaleV);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mySDDMM", &mySDDMM, "SDDMM for dot-product");
    m.def("mySDDMM_int8", &mySDDMM_int8, "SDDMM for dot-product (int8)");
}