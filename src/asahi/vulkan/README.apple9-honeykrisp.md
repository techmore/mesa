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

Both switches off, against the survey's default routes (before the load
change below): Qwen2.5-0.5B 60.6 to 79.6, Qwen2.5-1.5B 31.0 to 46.3,
Llama-3.2-1B 27.4 to 31.6, Qwen2.5-7B 12.1 to 16.5 tok/s.

With fusion off, a sweep of the other decode switches on granite 1B found
nothing better than 38.4 tok/s: `SDPA_DECODE_SPLIT`, `SDPA_SPLIT_PV`,
`SDPA_DECODE_BATCH`, `QMV_BATCH`, `ROPE_VECTOR_OFFSET`,
`LSE_ARGREDUCE_PREFETCH`, `OUTGATE_FOLD` and `FUSED_CHAIN` all land within 1%.
Two settings are large regressions on this hardware; keep their defaults:
`MLX_OMARCHY_QMM_VEC_Q4_WORD=0` (4.2 tok/s) and
`MLX_OMARCHY_SDPA_DECODE_NATIVE=0` (29.7 tok/s).

## Device loads now overlap within a thread

A thread issuing four independent 128-bit loads per loop iteration gained only
1.17x over one load when few threads were resident (4 workgroups: 14.2 vs
12.1 GB/s), so loads within a thread were serialized. `apple9_schedule_physical`
(`agx_apple9_ra.c`) retired a pending load as soon as a later instruction
wrote any register the load *read*. After allocation the index register of a
load is reused immediately by the next address computation, so every load was
retired (identity bridge with a scoreboard wait) right after it issued.

Apple's own compiler shows the hardware latches the index at issue. Compiling a
four-load loop with the Xcode Metal toolchain for `applegpu_g16g` gives four
back-to-back device loads, each writing its result into its own index register
(`load r5 <- [r5]`, index marked last use). The write-after-read check now
skips the sources of indexed device loads; result registers stay borrowed until
completion as before. Indirect (address-pair) loads, atomics, textures and
scratch keep the conservative rule. `AGX_APPLE9_STRICT_LOAD_HAZARD=1` restores
the old behaviour for comparison.

Read bandwidth, four 128-bit loads per thread per iteration:

| Workgroups | Before | After | After, `AGX_APPLE9_NO_PRESSURE_SCHED=1` |
|---:|---:|---:|---:|
| 4 | 14.3 GB/s | 17.7 GB/s | 36.0 GB/s |
| 16 | 48.4 GB/s | 58.1 GB/s | 82.1 GB/s |

4-bit GEMV: 2048 x 2048 goes from 40.1 to 45.1 GB/s, 512 x 2048 from 19.3 to
21.3 GB/s, 8192 x 3072 from 65.1 to 67.9 GB/s.

Decode tok/s with `MLX_OMARCHY_FUSED_GEMV=0 MLX_OMARCHY_FUSED_TRIO=0`, same
method as the survey:

| Model | Before | After | Change |
|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct-4bit | 79.6 | 86.3 | +8% |
| Qwen2.5-1.5B-Instruct-4bit | 46.3 | 50.1 | +8% |
| granite-4.0-1b-4bit | 38.1 | 44.8 | +18% |
| Llama-3.2-1B-Instruct-4bit | 31.6 | 31.0-32.1 | flat |
| Llama-3.2-3B-Instruct-4bit | 13.1 | 13.9 | +6% |
| Qwen2.5-7B-Instruct-4bit | 16.5 | 16.6 | +1% |

Greedy output text (96 tokens) is identical with and without the change on
granite 1B, Llama 3.2 3B and Qwen2.5 1.5B. All 39 compute and transfer tests
pass and the kernel logged no GPU faults.

`AGX_APPLE9_NO_PRESSURE_SCHED=1` skips `agx_pressure_schedule`, which places
loads next to their consumers. It doubles bandwidth with few resident threads
but leaves decode unchanged (granite 41.8 tok/s), so it stays an opt-in
experiment. Hoisting loads earlier in NIR and waiting at the first consumer
were also tried and changed nothing measurable.

## Latency-aware candidate selection

`agx_apple9_allocate_shared` already builds a latency-aware allocation
(`schedule_delay`, device loads weighted 16) next to the plain
pressure-scheduled one. It keeps the latency candidate only when it is cheaper
by spills, then instructions, bytes, registers, and only then dependency gaps.
`AGX_APPLE9_SCHED_DEBUG=1` prints every candidate's cost;
`AGX_APPLE9_SCHED_FORCE=plain|latency` forces one family for experiments.

For the four-load bandwidth kernel the latency candidate has the same
instructions and bytes, 22 instead of 59 dependency gaps, and 28 instead of 24
registers, so the register tie-break discards it. Forced, it matches
`NO_PRESSURE_SCHED`:

| Workgroups | Plain (default) | Latency forced |
|---:|---:|---:|
| 4 | 17.7 GB/s | 36.0 GB/s |
| 16 | 57.9 GB/s | 79.0 GB/s |
| 1024 | 78.3 GB/s | 76.7 GB/s |

It does not carry over to real kernels. 4-bit GEMV shapes (2048 x 2048,
3072 x 8192, 4096 x 4096, 1024 x 3072) are within run-to-run noise either way,
and decode is unchanged or worse:

| Model | Default | Latency forced | Plain forced |
|---|---:|---:|---:|
| granite-4.0-1b-4bit | 45.1 | 45.1 | 44.9 |
| Qwen2.5-0.5B-Instruct-4bit | 86.3 | 87.8 | 86.8 |
| Llama-3.2-3B-Instruct-4bit | 13.9 | 13.3 | 14.3 |

The selection order is therefore unchanged. After the load-hazard fix, decode
GEMV is not limited by load scheduling within a thread.

### Register cost on T8132

A probe kernel runs a dependent load chain per thread (latency-bound, L1
resident) while N extra values stay live across the loop, 262144 threads.
Throughput tracks resident threads:

| GPRs | 8 | 13 | 19 | 27 | 35 | 43 | 51 | 59 | 67 | 75 | 89 | 95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GB/s | 143 | 120 | 99 | 78 | 79 | 59-76 | 66 | 69 | 60 | 54 | 44 | 36 |

There are no sharp occupancy tiers like the older GPUs' table; cost rises
roughly continuously, about 1-2% of throughput per extra register, with
noise of up to 30% between neighbouring sizes as code layout changes. Above
about 95 GPRs the allocator rematerializes or spills. Extra registers are
not free on the M4, which supports keeping registers ahead of dependency gaps
in the selection order.

## Prefill: guarded loads become selects

Without cooperative matrix, omarchy-mlx runs f16 prefill through a scalar-FMA
kernel (`qmm_fma.comp`) and bf16 prefill through a shared-memory tile kernel
(`qmm_tile.comp`). The FMA kernel guards every load with a bounds check
(`row < m ? x[...] : 0`). On older Apple GPUs Honeykrisp marks loads
speculatable when the kernel enables soft faults, and NIR turns those guards
into selects. The M4 kernel reports no soft faults (`features=0x2`), so on
Apple9 each guard stayed a branch, and the allocator reconciled all 32
accumulators at every join: 5300 of the kernel's 6412 instructions were
register copies.

Apple9 now rewrites a load in a one-block arm to read element 0 of the same
binding when its arm is not taken, and lets peephole select flatten the
branch. That address is always mapped: null bindings resolve to the device's
read-only zero page instead of address 0. `CAN_REORDER` is granted only while
peephole select runs, so later passes cannot move these loads across stores to
an aliased binding. `AGX_APPLE9_NO_GUARDED_LOAD_SELECT=1` restores the branches.

The FMA kernel drops to 1977 instructions (901 copies). 4-bit matmul, 64 rows
x 8192 x 3072, goes from 174 to 778 GFLOP/s (the same kernel reaches 545 on an
M1 with soft faults). Prefill tok/s:

| Model (f16 activations) | Prompt | Before | After |
|---|---:|---:|---:|
| Llama-3.2-1B-Instruct-4bit | 44 | 40.3 | 163.1 |
| Llama-3.2-1B-Instruct-4bit | 512 | 67.4 | 309.7 |
| Qwen2.5-1.5B-Instruct-4bit | 44 | 29.0 | 132.5 |
| Qwen2.5-1.5B-Instruct-4bit | 512 | 53.8 | 249.2 |
| Mistral-7B-Instruct-v0.3-4bit | 44 | 7.1 | 33.1 |
| Mistral-7B-Instruct-v0.3-4bit | 512 | 12.0 | 57.0 |
| granite-4.0-1b-4bit (bf16) | 512 | 19.5 | 19.5 |

Greedy output text is identical before and after on all four models, decode
speed is unchanged, GPU results match the CPU within rounding on edge shapes
(m = 77, n = 1000; m = 130, n = 2050), and all 39 tests pass. Most
mlx-community 4-bit models use f16; bf16 models such as granite still take
the tile kernel at about 66 GFLOP/s. The null-binding redirect is not covered
by a hardware test yet: MLX never binds null descriptors.

### bf16 needs an FMA kernel in omarchy-mlx

omarchy-mlx routes only f16 to the FMA kernel. A bf16 port of the same kernel,
built at runtime through `mx.fast.metal_kernel` (x read as a uint32 view, two
bf16 values per load; scales and biases widened to f32 once), matches the CPU
within rounding and runs 8x faster than the shipped bf16 tile route:

| m x n x k | Tile (shipped) | bf16 FMA prototype |
|---|---:|---:|
| 64 x 8192 x 3072 | 66.6 GFLOP/s | 535.2 GFLOP/s |
| 512 x 2048 x 2048 | 66.0 | 558.2 |
| 130 x 2050 x 2048 | 58.7 | 341.8 |
| 77 x 1000 x 512 | 54.5 | 188.0 |

Routing granite-4.0-1b-4bit's multi-row 4-bit layers through it raises
prefill from 17.9 to 48.3 tok/s (44 tokens) and from 19.5 to 74.1 tok/s
(512 tokens), with the same greedy text.

The compiled version is techmore/omarchy-mlx commit `bf1406e` (branch
`m4-bf16-qmm-fma`): a `-DUSE_BF16=1` build of `qmm_fma.comp` that reads x as
uvec4 like the f16 build, routed for bf16 wherever the f16 FMA route is
reachable (`MLX_OMARCHY_NO_QMM_FMA_BF16=1` opts out). The f16 builds compile to
byte-identical SPIR-V. In a wheel built from it on the M4:

| | Tile (before) | bf16 FMA |
|---|---:|---:|
| 4-bit matmul, 64 x 8192 x 3072 | 66.4 GFLOP/s | 810.0 GFLOP/s |
| 4-bit matmul, 512 x 8192 x 3072 | 66.8 | 889.9 |
| granite-4.0-1b-4bit prefill, 512 tokens | 19.5 tok/s | 169.7 tok/s |
| Qwen3.5-2B-MLX-4bit prefill, 512 tokens | 14.6 | 62.6 |
| Qwen3.5-4B-MLX-4bit prefill, 512 tokens | 6.5 | 26.5 |

Greedy text is identical on all three models and decode is unchanged.

## Descriptor-table shaders: buffer roots from the preamble

Compute shaders with more than 18 resources (17 buffers plus push constants)
use descriptor tables. Every buffer access first loaded the binding's pointer
from the table, cached only within one block so loops reloaded it every
iteration, and then issued an address-pair load that was retired before the
next one issued. Apple's compiler, given pointers in an argument buffer, loads
them once in its constant program into uniform registers and uses ordinary
indexed loads with those registers as the root (`root=5`, `root=6` in a
`metal-tt -arch applegpu_g16g` build of such a kernel).

Apple9 now does the same for compute: the preamble loads each resource pointer
from the buffer table and stores it to uniform words 12 + 2a / 13 + 2a, and the
main program's loads, stores and atomics use root 6 + a. The preamble's own
accesses keep the table path; a shader without an extracted preamble gets an
empty one to carry the stores, and every fallback recompiles with the table
path. `AGX_APPLE9_NO_TABLE_ROOTS=1` restores it. `AGX_APPLE9_STATS` lines now
end in `table=1` for descriptor-table shaders.

Many-binding probe (`vkmany`, every thread reads all inputs in a loop, results
checked against the CPU): 8 workgroups / 256 workgroups, Gload/s.

| Buffers | Before | After |
|---:|---:|---:|
| 17 (direct ABI) | 1.19 / 37.3 | 1.19 / 36.3 |
| 18 | 1.03 / 31.7 | 1.21 / 37.2 |
| 24 | 1.07 / 32.2 | 1.28 / 39.4 |
| 30 | 1.10 / 39.7 | 1.59 / 46.0 |

omarchy-mlx's fused decode GEMV (26 resources) loses its per-iteration table
loads and its loads overlap; with the fused routes on, decode rises 8-14%
(granite 1B 34.0 to 36.6 tok/s, Qwen2.5 1.5B 32.1 to 36.6, Llama 3.2 3B 10.3
to 11.5). Greedy text is identical on six models with the table path and with
roots, all 35 compute tests pass and the kernel logged no GPU faults.

The per-node path is still faster than the fused groups on this GPU, so
omarchy-mlx now defaults both fused decode routes off on G16 (techmore/
omarchy-mlx, branch `m4-bf16-qmm-fma`). Decode tok/s with no environment
settings, old default (fused) to new default:

| Model | Fused | New default |
|---|---:|---:|
| granite-4.0-1b-4bit | 36.6 | 45.7 |
| Qwen2.5-0.5B-Instruct-4bit | 68.8 | 87.8 |
| Qwen2.5-1.5B-Instruct-4bit | 36.6 | 50.2 |
| Llama-3.2-1B-Instruct-4bit | 28.3 | 30.2 |
| Llama-3.2-3B-Instruct-4bit | 11.6 | 13.7 |
| Qwen3.5-2B-MLX-4bit | 26.1 | 33.0 |
| Mistral-7B-Instruct-v0.3-4bit | 14.8 | 16.7 |

Greedy text is unchanged on six of the seven; Qwen2.5 1.5B takes a different,
equally coherent wording from word 41, from the two routes' accumulation order.

## Native subgroup shuffles

`subgroupShuffleXor/Up/Down` and `subgroupShuffle` are now single
instructions when the lane, offset or mask is a constant or subgroup-uniform
(NIR divergence analysis). Encodings, from `metal-tt -arch applegpu_g16g`
builds varying registers, offsets and liveness:

    [op0] [op1] 54 [dest<<1] 03 [src<<2] [F] [B7] [B8] 00

| op0 op1 | Relation |
|---|---|
| `47 04` | index (`simd_shuffle` / `simd_broadcast`) |
| `c7 04` | xor |
| `47 05` | up |
| `c7 05` | down |

F is the operand: an immediate << 1 (`2c 02` in B7/B8) or the low
half-register << 1 of a GPR (`2e 12`). B8 bit 1 / bit 2 is Metal's keep /
last-use hint for the data register; for the operand, B7 bit 1 with B8 bit 4
is a retained GPR and B8 bits 3 and 4 a last-use GPR (the existing
broadcast's `2c 1c`, hence its clobbering inputs); bit 3 alone appears with
a uniform-register operand. Byte 1 bits 4-7 and byte 2 bits 0-1 are the
usual wait mask; 16-bit data uses B7 = `20`. The backend always emits the retaining forms, so no input is consumed;
narrow and boolean values shuffle as 32-bit lanes, 64-bit as two halves.

The register operand is not per lane. Probes with per-quad operands fit one
model exactly: each quad reads the operand from one of its active lanes, the
destination quad's value picks the source quad and the source quad's value
picks the position inside it (`sh_quad_mix` reproduces the predicted wrong
answer). Metal emits xor/up/down as one instruction but expands
`simd_shuffle` into four shuffles and selects; here per-lane operands keep
the broadcast fallback. Uniform operands are exact in divergent
control flow even when the inactive lanes' registers hold other values.
In the probes, source lanes past the subgroup end or inactive read as 0.
`AGX_APPLE9_NO_NATIVE_SHUFFLE=1` restores the fallback;
`AGX_APPLE9_NATIVE_SHUFFLE_DIVERGENT=1` forces the single instruction for
hardware probes only (wrong for per-lane operands).

A shuffle was 32 broadcasts and selects: `subgroupShuffleXor(v, 1)` compiles
to 9 instructions instead of 199, a five-step xor butterfly to 16 instead of
848. 45 new compute cases (xor/up/down by 1, 2, 4, 8, 16; f32, f16, u16, bool,
u64, uvec2; constant, push-constant and UBO operands; variable index,
out-of-range lanes, divergent control flow, loops) pass natively and with the
opt-out, as do the 35 existing tests; no kernel messages.

The decode paths of the 4-bit models below contain no shuffles (`qmm_vec`
avoids them; shuffles are in `matmul_vec`, the bf16 GEMV, the gated-delta
kernels and coopmat flash attention), so decode is unchanged within noise and
greedy text is identical. Decode tok/s, `llm_survey.py 32`, three interleaved
A/B rounds, mean (range):

| Model | Native (runs) | `AGX_APPLE9_NO_NATIVE_SHUFFLE=1` (runs) |
|---|---:|---:|
| granite-4.0-1b-4bit | 45.8 (45.8-45.9) | 45.8 (45.8-45.8) |
| Qwen2.5-1.5B-Instruct-4bit | 50.2 (50.2-50.3) | 50.2 (50.2-50.2) |
| Llama-3.2-1B-Instruct-4bit | 30.9 (28.9-32.0) | 29.7 (28.7-31.5) |

Llama 3.2 1B moves between about 29 and 32 tok/s in both settings (it has no
shuffles either), so that difference is run-to-run variation on the shared GPU.

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

- Descriptor-table graphics shaders still use the table path (the preamble
  roots are enabled for compute only).
- Reduce the GEMV floor on small layers (512 x 2048 still takes about 35 us).
  Load scheduling is ruled out (see "Latency-aware candidate selection");
  candidates are the kernel's per-slot structure, splitting K in the MLX
  kernels, and per-dispatch cost.
- Upstream the bf16 FMA route to omarchy-mlx (techmore/omarchy-mlx `bf1406e`).
- Cooperative matrix for Apple9, for prefill beyond the FMA kernels. Joshua
  Warren's G13 Honeykrisp (github.com/joshuaswarren/mesa-1, branches
  `asahi/hwmat-vec2-on` and `agent/hk-bf16-cmat`) gets about 2x over the FMA
  kernel on an M1 from simdgroup matrix instructions; the Apple9 encodings
  would come from the Xcode Metal toolchain as the subgroup ones did.
- Per-lane shuffle operands still use the broadcast fallback; Metal's
  four-shuffle expansion of `simd_shuffle` is not decoded yet.
- Apple9 pipelines are not stored in the pipeline cache.
