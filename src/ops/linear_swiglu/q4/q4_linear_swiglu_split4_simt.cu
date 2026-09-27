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

    // fp16 bias 1032.0 = 1024 + 8: 0x6400 | n - 0x6408 recovers signed q4 in [-8, 7].
    const __half2 kQ4Bias = __half2half2(__ushort_as_half(static_cast<std::uint16_t>(0x6408u)));

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

    // Software pipeline: keep the current slab's packed words and scales in
    // registers and issue the next slab's independent loads before the fma block
    // consumes the current one, hiding one slab of DRAM latency per warp.
    std::uint32_t cur_gw = 0;
    std::uint32_t cur_uw = 0;
    float cur_gs = 0.0f;
    float cur_us = 0.0f;
    auto load_slab = [&](int s, std::uint32_t& gw, std::uint32_t& uw, float& gs, float& us) {
        gw = *reinterpret_cast<const std::uint32_t*>(gate_code + s * 512 + chunk * 128 + lane * 4);
        uw = *reinterpret_cast<const std::uint32_t*>(up_code + s * 512 + chunk * 128 + lane * 4);
        const int group = s * 16 + chunk * 4 + (lane >> 3);
        std::uint32_t bits = 0;
        if ((lane & 7) == 0) {
            bits = static_cast<std::uint32_t>(gate_scale[group]) |
                   (static_cast<std::uint32_t>(up_scale[group]) << 16);
        }
        bits = __shfl_sync(0xffffffffu, bits, lane & ~7);
        gs = __half2float(__ushort_as_half(static_cast<std::uint16_t>(bits & 0xffffu)));
        us = __half2float(__ushort_as_half(static_cast<std::uint16_t>(bits >> 16)));
    };

    load_slab(0, cur_gw, cur_uw, cur_gs, cur_us);
#pragma unroll
    for (int s = 0; s < kSlabs; ++s) {
        std::uint32_t nxt_gw = 0;
        std::uint32_t nxt_uw = 0;
        float nxt_gs = 0.0f;
        float nxt_us = 0.0f;
        const bool more = s + 1 < kSlabs;
        if (more) {
            load_slab(s + 1, nxt_gw, nxt_uw, nxt_gs, nxt_us);
        }
        const std::int64_t xoff = s * 1024 + chunk * 256 + lane * 8;
#pragma unroll
        for (int tt = 0; tt < kTt; ++tt) {
            const uint4 xv = load_vec<uint4>(x + static_cast<std::int64_t>(tt) * kK + xoff);
            const float2 f0 = bf16x2_bits_to_float2(xv.x);
            const float2 f1 = bf16x2_bits_to_float2(xv.y);
            const float2 f2 = bf16x2_bits_to_float2(xv.z);
            const float2 f3 = bf16x2_bits_to_float2(xv.w);
            const float xs[8] = {f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y};
            // q4 decode through the fp16 exponent trick. The stored nibbles are
            // two's complement, so first flip each nibble's sign bit (0x00080008
            // covers the low and high half words) and then let 0x6400 | m represent
            // 1024 + (n ^ 8) exactly in fp16; subtracting 1032 = 1024 + 8 recovers
            // the signed q in [-8, 7] with int-exact results at a fraction of the
            // integer-alu cost of per-nibble extraction.
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const std::uint32_t gb =
                    (((cur_gw >> (4 * p)) & 0x000f000fu) ^ 0x00080008u) | 0x64006400u;
                const std::uint32_t ub =
                    (((cur_uw >> (4 * p)) & 0x000f000fu) ^ 0x00080008u) | 0x64006400u;
                const float2 gq = __half22float2(__hsub2(half2_from_bits(gb), kQ4Bias));
                const float2 uq = __half22float2(__hsub2(half2_from_bits(ub), kQ4Bias));
                gate_acc[tt] = fmaf(gq.x * cur_gs, xs[p], gate_acc[tt]);
                gate_acc[tt] = fmaf(gq.y * cur_gs, xs[p + 4], gate_acc[tt]);
                up_acc[tt]   = fmaf(uq.x * cur_us, xs[p], up_acc[tt]);
                up_acc[tt]   = fmaf(uq.y * cur_us, xs[p + 4], up_acc[tt]);
            }
        }
        cur_gw = nxt_gw;
        cur_uw = nxt_uw;
        cur_gs = nxt_gs;
        cur_us = nxt_us;
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
