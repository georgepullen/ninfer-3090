#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kN            = 34816;
constexpr int kK            = 5120;
constexpr int kIntermediate = kN / 2;
constexpr int kGroupK       = 64;
constexpr int kSlabs        = kK / 1024; // 5 slabs of 1024 elements
constexpr int kRowCodeBytes = kK / 2;
constexpr int kRowGroups    = kK / kGroupK;

// One CTA owns one gate/up row pair and streams both weight rows exactly once. The
// four warps split the K extent into four 256-element chunks per 1024-element slab;
// each lane dequantizes eight q4 weights and multiplies them into kTt register
// accumulators, so DRAM weight bytes stay constant while the token count grows.
// Epilogue: out[t * kIntermediate + pair] = silu(gate) * up.
template <int kTt>
__launch_bounds__(128, 8) __global__ void q4_linear_swiglu_split4_pair_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out) {
    __shared__ float s_gate[4][kTt];
    __shared__ float s_up[4][kTt];

    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int chunk = static_cast<int>(threadIdx.x) >> 5;
    const int pair  = static_cast<int>(blockIdx.x);

    const std::int64_t gate_row  = pair;
    const std::int64_t up_row    = pair + kIntermediate;
    const std::uint8_t* gate_code = codes + gate_row * kRowCodeBytes;
    const std::uint8_t* up_code   = codes + up_row * kRowCodeBytes;
    const std::uint16_t* gate_scale =
        reinterpret_cast<const std::uint16_t*>(scales + gate_row * kRowGroups * 2);
    const std::uint16_t* up_scale =
        reinterpret_cast<const std::uint16_t*>(scales + up_row * kRowGroups * 2);

    float gate_acc[kTt];
    float up_acc[kTt];
#pragma unroll
    for (int tt = 0; tt < kTt; ++tt) {
        gate_acc[tt] = 0.0f;
        up_acc[tt]   = 0.0f;
    }

#pragma unroll
    for (int s = 0; s < kSlabs; ++s) {
        const int off  = s * 1024 + chunk * 256 + lane * 8;
        const int group = s * 16 + chunk * 4 + (lane >> 3);

        const std::uint32_t gate_word =
            *reinterpret_cast<const std::uint32_t*>(gate_code + s * 512 + chunk * 128 + lane * 4);
        const std::uint32_t up_word =
            *reinterpret_cast<const std::uint32_t*>(up_code + s * 512 + chunk * 128 + lane * 4);
        std::uint32_t scale_bits = 0;
        if ((lane & 7) == 0) {
            scale_bits = static_cast<std::uint32_t>(gate_scale[group]) |
                         (static_cast<std::uint32_t>(up_scale[group]) << 16);
        }
        scale_bits = __shfl_sync(0xffffffffu, scale_bits, lane & ~7);
        const float gate_scale =
            __half2float(__ushort_as_half(static_cast<std::uint16_t>(scale_bits & 0xffffu)));
        const float up_scale =
            __half2float(__ushort_as_half(static_cast<std::uint16_t>(scale_bits >> 16)));

        const std::int64_t xoff = s * 1024 + chunk * 256 + lane * 8;
#pragma unroll
        for (int tt = 0; tt < kTt; ++tt) {
            const uint4 xv = load_vec<uint4>(x + static_cast<std::int64_t>(tt) * kK + xoff);
            const float2 f0 = bf16x2_bits_to_float2(xv.x);
            const float2 f1 = bf16x2_bits_to_float2(xv.y);
            const float2 f2 = bf16x2_bits_to_float2(xv.z);
            const float2 f3 = bf16x2_bits_to_float2(xv.w);
            const float xs[8] = {f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y};
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int gq = (static_cast<int>((gate_word >> (4 * e)) & 0x0fu) ^ 0x08) - 0x08;
                const int uq = (static_cast<int>((up_word >> (4 * e)) & 0x0fu) ^ 0x08) - 0x08;
                gate_acc[tt] = fmaf(static_cast<float>(gq) * gate_scale, xs[e], gate_acc[tt]);
                up_acc[tt]   = fmaf(static_cast<float>(uq) * up_scale, xs[e], up_acc[tt]);
            }
        }
    }

#pragma unroll
    for (int tt = 0; tt < kTt; ++tt) {
        float g = warp_reduce_sum(gate_acc[tt]);
        float u = warp_reduce_sum(up_acc[tt]);
        if (lane == 0) {
            s_gate[chunk][tt] = g;
            s_up[chunk][tt]   = u;
        }
    }
    __syncthreads();

    if (chunk == 0 && lane < kTt) {
        float gate = 0.0f;
        float up   = 0.0f;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            gate += s_gate[c][lane];
            up += s_up[c][lane];
        }
        out[static_cast<std::int64_t>(lane) * kIntermediate + pair] =
            __float2bfloat16_rn(silu(gate) * up);
    }
}

template <int kTt>
void launch_split4_pair(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(kIntermediate), 1u, 1u);
    q4_linear_swiglu_split4_pair_kernel<kTt>
        <<<grid, 4 * 32, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                      static_cast<const std::uint8_t*>(w.qdata),
                                      static_cast<const std::uint8_t*>(w.scales),
                                      static_cast<__nv_bfloat16*>(out.data));
}

} // namespace

void q4_linear_swiglu_split4_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    switch (x.ne[1]) {
    case 1:
        launch_split4_pair<1>(x, w, out, stream);
        break;
    case 2:
        launch_split4_pair<2>(x, w, out, stream);
        break;
    case 3:
        launch_split4_pair<3>(x, w, out, stream);
        break;
    case 4:
        launch_split4_pair<4>(x, w, out, stream);
        break;
    case 5:
        launch_split4_pair<5>(x, w, out, stream);
        break;
    case 6:
        launch_split4_pair<6>(x, w, out, stream);
        break;
    case 7:
        launch_split4_pair<7>(x, w, out, stream);
        break;
    case 8:
        launch_split4_pair<8>(x, w, out, stream);
        break;
    default:
        throw std::invalid_argument("q4 linear_swiglu split4: T must be in [1,8]");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
