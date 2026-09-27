#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <int Cols, int FullSlabs, int Stride>
void launch_split4_exact(const Tensor& x, const Weight& w, Tensor& residual_out,
                         cudaStream_t stream) {
    constexpr int kThreads = 4 * 32;
    const dim3 grid(static_cast<unsigned>(residual_out.ne[0]), 1u, 1u);
    const std::int32_t out_ld =
        static_cast<std::int32_t>(residual_out.nb[1] / sizeof(__nv_bfloat16));
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Cols, FullSlabs, Stride, false, 0,
                                        Q5Split4AddResidualEpilogue>
        <<<grid, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(residual_out.data), nullptr, residual_out.ne[0], out_ld,
            x.ne[0], x.ne[1], w.padded_shape[1], FullSlabs);
}

template <int Cols>
void dispatch_shape(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    if (w.k == 6144) {
        launch_split4_exact<Cols, 6, 6144>(x, w, residual_out, stream);
    } else if (w.k == 17408) {
        launch_split4_exact<Cols, 17, 17408>(x, w, residual_out, stream);
    } else {
        throw std::invalid_argument("q5 linear_add split4: unsupported exact K");
    }
}

} // namespace

void q5_linear_add_split4_exact_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    const auto launch = [&]<int Cols>() {
        dispatch_shape<Cols>(x, w, residual_out, stream);
    };
    switch (x.ne[1]) {
#define NINFER_Q5_LINEAR_ADD_SPLIT4_EXACT(COLS)                                                    \
    case COLS:                                                                                     \
        launch.template operator()<COLS>();                                                        \
        break
        NINFER_Q5_LINEAR_ADD_SPLIT4_EXACT(1);
        NINFER_Q5_LINEAR_ADD_SPLIT4_EXACT(2);
        NINFER_Q5_LINEAR_ADD_SPLIT4_EXACT(3);
        NINFER_Q5_LINEAR_ADD_SPLIT4_EXACT(4);
        NINFER_Q5_LINEAR_ADD_SPLIT4_EXACT(5);
        NINFER_Q5_LINEAR_ADD_SPLIT4_EXACT(6);
        NINFER_Q5_LINEAR_ADD_SPLIT4_EXACT(7);
        NINFER_Q5_LINEAR_ADD_SPLIT4_EXACT(8);
        NINFER_Q5_LINEAR_ADD_SPLIT4_EXACT(9);
        NINFER_Q5_LINEAR_ADD_SPLIT4_EXACT(10);
        NINFER_Q5_LINEAR_ADD_SPLIT4_EXACT(11);
        NINFER_Q5_LINEAR_ADD_SPLIT4_EXACT(12);
        NINFER_Q5_LINEAR_ADD_SPLIT4_EXACT(13);
        NINFER_Q5_LINEAR_ADD_SPLIT4_EXACT(14);
        NINFER_Q5_LINEAR_ADD_SPLIT4_EXACT(15);
        NINFER_Q5_LINEAR_ADD_SPLIT4_EXACT(16);
#undef NINFER_Q5_LINEAR_ADD_SPLIT4_EXACT
    default:
        throw std::invalid_argument("q5 linear_add split4: T must be in [1,16]");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
