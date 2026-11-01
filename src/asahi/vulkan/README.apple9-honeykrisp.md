# Honeykrisp compute on Apple9 (M4, T8132)

Honeykrisp runs Vulkan compute on the M4 Mac mini through the Apple9 compiler
and launch records (`hk_apple9.c`). Graphics are not supported. This note
records what was measured on hardware: an M4 Mac mini (Mac16,10, G16G,
16 GB), Gravity's `linux-gravity-m4` 7.1.13 kernel, Omarchy 4.0.

## LLM survey (omarchy-mlx v0.7.35, mlx-lm 0.31.3)

Method: chat-templated prompt "Explain in plain language how a rainbow forms.
Give four distinct sentences.", greedy decoding, 32 new tokens, one warmup and
the median of three runs, `MLX_OMARCHY_BATCH_WORK=120000`,
`AGX_APPLE9_MAX_SHARED=32768`, MLX's default kernel routes. Every model
produced coherent, correct text and no run logged a GPU fault.

| Model (mlx-community) | Decode tok/s | Prefill tok/s | Peak GiB |
|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct-4bit | 60.6 | 76.6 | 0.31 |
| gemma-3-1b-it-4bit | 34.8 | 24.1 | 0.77 |
| granite-4.0-1b-4bit | 31.9 | 20.6 | 0.97 |
| Qwen2.5-1.5B-Instruct-4bit | 31.0 | 31.7 | 0.93 |
| Llama-3.2-1B-Instruct-4bit | 27.4 | 52.3 | 0.71 |
| Llama-3.2-1B-Instruct-8bit | 25.9 | 26.4 | — |
| Qwen3.5-2B-MLX-4bit | 23.3 | 15.6 | 1.13 |
| Llama-3.2-1B-Instruct-bf16 | 16.8 | 42.7 | — |
| Mistral-7B-Instruct-v0.3-4bit | 12.7 | 3.5 | 4.15 |
| Qwen2.5-7B-Instruct-4bit | 12.1 | 8.0 | 4.77 |
| Llama-3.2-3B-Instruct-4bit | 11.9 | 18.7 | 1.85 |
| Qwen3.5-4B-MLX-4bit | 11.1 | 6.3 | 2.59 |
| Phi-3.5-mini-instruct-4bit | 10.9 | 6.4 | 2.27 |

## Where decode time goes

- Decode is GPU-bound. Launching every dispatch twice halves throughput
  (granite 1B: 30.0 to 15.3 tok/s), and two concurrent generators share one
  generator's total throughput.
- Raw read bandwidth is healthy: a streaming kernel reads 78-80 GB/s with
  32-bit or 128-bit loads.
- Decode GEMV is latency-bound on small layers. 4-bit group-64 bf16 GEMV:

  | N x K | Time | GB/s |
  |---|---:|---:|
  | 512 x 2048 | 32 us | 18 |
  | 2048 x 2048 | 60 us | 39 |
  | 3072 x 3072 | 108 us | 49 |
  | 8192 x 3072 | 221 us | 64 |
  | 4096 x 14336 | 427 us | 77 |

- Per-class GPU cost per launch (granite 1B, by duplicating one class with
  `HK_APPLE9_DUP`): the descriptor-table-ABI fused GEMV group (26 resources)
  costs 208 us and runs 85 times per token, 17.6 ms of the ~33 ms token. The
  next classes are a 6-resource GEMV (75 us) and a 1024-thread, 32 KiB shared
  memory kernel (57 us). Element-wise kernels cost 2-8 us.
- With batching, a token issues about one submit. Submit-and-wait round trip
  is 283 us average; each extra dispatch inside a submit costs 4-5 us.
- Prefill matmul is weak without cooperative matrix: 64-row 4-bit matmul
  reaches 66 GFLOP/s (bf16, tile kernel) and 162 GFLOP/s (f16, FMA kernel).

## Tuning that helps today

`MLX_OMARCHY_FUSED_GEMV=0` (and `MLX_OMARCHY_FUSED_TRIO=0` for Llama-style
models) routes decode GEMV around the fused descriptor-table kernel:

| Model | Default | FUSED_GEMV=0 | + FUSED_TRIO=0 |
|---|---:|---:|---:|
| granite-4.0-1b-4bit | 30.0 | 38.4 | 38.4 |
| Llama-3.2-3B-Instruct-4bit | 11.0 | 12.2 | 13.3 |

With fusion off, a sweep of the other decode switches on granite 1B found
nothing better than 38.4 tok/s: `SDPA_DECODE_SPLIT`, `SDPA_SPLIT_PV`,
`SDPA_DECODE_BATCH`, `QMV_BATCH`, `ROPE_VECTOR_OFFSET`,
`LSE_ARGREDUCE_PREFETCH`, `OUTGATE_FOLD` and `FUSED_CHAIN` all land within 1%.
Two settings are large regressions on this hardware; keep their defaults:
`MLX_OMARCHY_QMM_VEC_Q4_WORD=0` (4.2 tok/s) and
`MLX_OMARCHY_SDPA_DECODE_NATIVE=0` (29.7 tok/s).

## Why small GEMVs are latency-bound

A thread issuing four independent 128-bit loads per loop iteration gains only
1.17x over one load when few threads are resident (4 workgroups: 14.2 vs
12.1 GB/s), so loads within a thread are effectively serialized. The cause is
in the Apple9 back end, not in instruction order: selection and the VIR
optimizer keep the four loads together, but `apple9_schedule_physical`
(`agx_apple9_ra.c`) retires a pending load as soon as a later instruction
writes any register the load *read*. After allocation the index register of a
load is reused immediately by the next address computation, so every load is
retired (identity bridge with a scoreboard wait) right after it issues.

Moving loads earlier in NIR, waiting at the first consumer instead of after
the producer, and skipping `agx_pressure_schedule` were all tried and changed
nothing measurable (granite decode 38.3-38.5 tok/s either way, so they are not
merged). The fix is
either to show the hardware latches a device load's index at issue (then the
write-after-read check can exempt device-load sources), or to keep each load's
index register live until the load's first consumer during allocation.

## Kernel

The Mini runs Omarchy's `linux-gravity-m4` 7.1.13 package. Gravity's
`replacement/210-gpu` branch (Linux 7.3) adds a much larger shared G16C/G17P
driver (M4 Pro/Max), integrated in one commit that states it is not hardware
qualified; it keeps the ordinary M4 driver. Its compute changes are M4 Max
validated only. Completion polling at 200 us is shared code; with batching a
decode token waits about once, so it costs under 1% today. Not worth a kernel
switch for the base M4 until Gravity publishes a qualified build.

## Driver changes in this commit

- Apple9 compute control streams start at 4 KiB instead of 64 KiB. MLX ends
  the compute stream at nearly every barrier, so most streams carry one
  launch record; the old size made the command pool allocate a BO for every
  second dispatch.
- `AGX_APPLE9_MAX_SHARED` is read once rather than on every launch build.
- `HK_APPLE9_DUP=all` or `HK_APPLE9_DUP=<preamble_offset>/<local_x>/<resources>/<threadgroup_bytes>`
  launches matching dispatches twice for profiling (results are wrong while it
  is set). `HK_APPLE9_DEBUG=1` prints the key fields of every launch.

## Open work

- Make the descriptor-table ABI (more than 18 resources) as fast as the
  direct-buffer ABI, or lift the 18-resource direct limit.
- Reduce the GEMV latency floor on small layers: let device loads overlap
  within a thread (see "Why small GEMVs are latency-bound"), or split K in the
  MLX kernels.
- Cooperative matrix for Apple9, for prefill.
- Shuffles still use the broadcast fallback. Apple's Metal compiler output
  for applegpu_g16g shows xor/up/down with immediate offsets as single
  instructions (`c7 04` xor, `c7 05` down, `47 05` up, offset in byte 6), not
  yet wired in.
- Apple9 pipelines are not stored in the pipeline cache.
