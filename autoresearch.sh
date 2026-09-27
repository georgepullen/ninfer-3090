#!/usr/bin/env bash
# Deterministic benchmark harness: NInfer-3090 Qwen3.8-27B (groupwise-int) token
# throughput on one RTX 3090.
#
# Primary metric:  decode_tps_plain     - tg256 decode output tokens/s without
#                                       speculation; deterministic and independent of
#                                       MTP acceptance trajectory.
# Secondary:       decode_tps           - tg256 decode output tokens/s with MTP3 +
#                                       optimized draft head (every accepted token is
#                                       validated by the target model logits); the
#                                       shipped fast path, but trajectory-sensitive.
#                  prefill_tps          - pp512 prefill tokens/s.
#                  mtp_acceptance_rate  - drafted tokens accepted per round window.
#
# Workload is fixed: bench/fixtures/bench_corpus.ids slices, greedy decoding
# (engine temperature defaults to 0), fixed repetition counts, int8 KV, CUDA
# Graph decode. No network, no clock-dependent behavior.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$ROOT/build-sm86"
BENCH="$BUILD/bench/ninfer_bench"
CORPUS="$ROOT/bench/fixtures/bench_corpus.ids"
WEIGHTS="${NINFER_WEIGHTS:-/mnt/hdd1tb/models/qwen3_8_27b.ninfer}"
CUDA_ENV="${NINFER_CUDA_ENV:-${HOME}/shared/envs/cuda134}"

export LD_LIBRARY_PATH="$CUDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- build (incremental; configures a fresh build dir on first run) ----------
if [[ ! -f "$BUILD/CMakeCache.txt" ]]; then
  cmake -S "$ROOT" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-13 \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
    -DCMAKE_CUDA_COMPILER="$CUDA_ENV/bin/nvcc" \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DNINFER_BUILD_APPS=OFF \
    -DBUILD_TESTING=OFF \
    -DNINFER_BUILD_BENCHMARKS=ON
fi
cmake --build "$BUILD" -j --target ninfer_bench

[[ -x "$BENCH" ]] || { echo "ninfer_bench not built at $BENCH" >&2; exit 1; }
[[ -f "$CORPUS" ]] || { echo "corpus fixture missing: $CORPUS" >&2; exit 1; }
[[ -f "$WEIGHTS" ]] || { echo "artifact missing: $WEIGHTS (set NINFER_WEIGHTS)" >&2; exit 1; }

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

# --- primary: tg256 decode with MTP3 + optimized draft head ------------------
"$BENCH" \
  --weights "$WEIGHTS" \
  --corpus "$CORPUS" \
  -n 256 \
  --mtp-draft-tokens 1 \
  --lm-head-draft \
  --kv-dtype int8 \
  --max-ctx 4096 \
  -r 3 --warmup 1 \
  -o json --output-file "$OUT/mtp.json" >/dev/null

# --- secondaries: plain tg256 decode + pp512 prefill (no speculation) --------
"$BENCH" \
  --weights "$WEIGHTS" \
  --corpus "$CORPUS" \
  -p 512 -n 256 \
  --kv-dtype int8 \
  --max-ctx 4096 \
  -r 3 --warmup 1 \
  -o json --output-file "$OUT/plain.json" >/dev/null

python3 - "$OUT/mtp.json" "$OUT/plain.json" <<'PY'
import json
import sys


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def test(report, label):
    for entry in report["tests"]:
        if entry["label"] == label:
            return entry
    raise KeyError(f"test {label!r} not in {sorted(t['label'] for t in report['tests'])}")


def stat(entry, key):
    value = entry[f"{key}_mean"]
    if value is None:
        raise KeyError(f"{key} is null for {entry['label']}")
    return value


mtp = load(sys.argv[1])
plain = load(sys.argv[2])

decode = test(mtp, "tg256")
plain_decode = test(plain, "tg256")
prefill = test(plain, "pp512")

acceptance = decode["speculative"]["acceptance_rate"]

# decode_tps_plain is the decision metric: deterministic per-kernel, no
# acceptance-trajectory lottery. decode_tps (MTP3, product fast path) is reported
# alongside with its acceptance rate.
print(f"METRIC decode_tps_plain={stat(plain_decode, 'decode_output_tok_s'):.2f}")
print(f"METRIC decode_tps={stat(decode, 'decode_output_tok_s'):.2f}")
print(f"METRIC prefill_tps={stat(prefill, 'prefill_tok_s'):.2f}")
if acceptance is not None:
    print(f"METRIC mtp_acceptance_rate={acceptance:.4f}")
PY
