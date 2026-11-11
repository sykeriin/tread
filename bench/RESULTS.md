# Phase 1 results — reuse engine + TTFT benchmark (RTX 3050 laptop, 4 GB)

Workload: reordered RAG. 50 HotpotQA questions (validation 400–449, unused by Phase 0), each asked
twice with its 8 docs (2 gold + 6 distractors) in two different random orders, ~1,130 context tokens.
Pass 1 fills the block store; pass 2 is measured. Exact-prefix caching can reuse only the system
prompt on pass 2; tread reuses every doc block. Each warm method is timed 3×, median kept; the first
3 questions are warm-up. Quality is measured on the first answer token vs full prefill.

Reproduce: `.venv/Scripts/python bench/bench_ttft.py --model Qwen/Qwen2.5-0.5B-Instruct --n 50`

## Clean timing: Qwen2.5-0.5B fp16 (2.3 GB VRAM free, SM clock 2,017 MHz, peak 2.1 GB used)

| method | TTFT p50 | p95 | vs full | prefilled tokens | KL | top-1 |
|---|---|---|---|---|---|---|
| full prefill | 100.6 ms | 134.0 ms | 1.00× | 1142 | 0 | 1.000 |
| exact-prefix cache | 105.1 ms | 152.0 ms | 0.96× | 1120 | 0.0001 | 1.000 |
| tread, plain reuse | **81.5 ms** | 91.0 ms | **1.23×** | 0 | 0.986 | 0.553 |
| tread, 20% recompute | 169.7 ms | 180.3 ms | 0.59× | 0 | 0.526 | 0.745 |
| tread, 50% recompute | 168.5 ms | 180.0 ms | 0.60× | 0 | 0.208 | 0.830 |
| tread, cold (all misses) | 636.6 ms | 694.4 ms | 0.16× | 1117 | 1.272 | 0.468 |

Partial recompute made the first token worse than plain reuse on 11/47 requests at 10–20%,
7/47 at 30%, and 1/47 at 50%.

## What this shows
1. **At this scale, prefill is not compute-bound, so the engine's overhead decides.** Full prefill of
   1,100 tokens on 0.5B takes 100 ms, mostly per-layer launch and Python overhead. Skipping all
   of it saves only 19 ms (1.23×).
2. **Recompute cost is flat in the ratio (~170 ms at 10% and at 50%).** It is pure overhead from the
   per-layer Python loop in `blend` (28 layers × a dozen small ops), not FLOPs. That is SCOPE.md
   risk #3, materialized.
3. **The cold path is 6× slower than full prefill** because it runs one forward per doc (9 calls).
4. **Quality at k=8 is the bigger problem.** Plain reuse matches full prefill's first token only 55%
   of the time (k=4 in Phase 0 was much better). Recompute helps on average (83% at 50%), but it is
   not monotone per request.

## Invalid runs (kept for the record, do not cite)
`bench/results/k8_before_opt.*`, `k8_vram_spill.*`, `Qwen2.5-1.5B-Instruct_k8.*`, `bench/controlled/`.
With 1.5B fp16 loaded, the 4 GB GPU had **19 MiB free**. Windows (WDDM) then silently spills VRAM to
system RAM, so full prefill measured 300–2,700 ms depending on which other apps were open, while
tread's small passes were barely affected. That produced fake "14–25× speedups". The benchmark now
refuses to run with too little free VRAM (`--allow-low-vram` to override). Their *quality* numbers
are unaffected and match across runs.

Engine fix made along the way: blocks are now stored stacked over layers and rotated in one op.
Assembly went from 186 ms to 13 ms on 1.5B.

## Context-length scaling (0.5B fp16, 13 questions per length, `bench/bench_scaling.py`)

Plot: `results/scaling_Qwen2.5-0.5B-Instruct.png`. Each cell is median TTFT and speedup vs full prefill.

| ctx tokens | full | plain reuse | 20% recompute | 50% recompute |
|---|---|---|---|---|
| 1k | 96 ms | 95 ms (1.0×) | 194 ms (0.5×) | 189 ms (0.5×) |
| 2k | 180 ms | 97 ms (1.9×) | 193 ms (0.9×) | 218 ms (0.8×) |
| 4k | 364 ms | 116 ms (3.1×) | **257 ms (1.4×)** | 400 ms (0.9×) |
| 8k | 841 ms | 146 ms (5.8×) | **516 ms (1.6×)** | 938 ms (0.9×) |
| 16k* | 2162 ms | 235 ms (9.2×) | 1809 ms (1.2×) | (invalid) |

First-token top-1 agreement / KL vs full: plain reuse 0.15–0.69 / 0.6–2.0 (unusable alone);
20% recompute 0.69–0.92 / 0.12–0.58; 50% recompute 0.69–0.92 / 0.02–0.25. Recompute quality
*improves* with context length (KL at 50%: 0.21 at 1k → 0.02 at 16k).

\*At 16k, GPU memory peaked at 3.95 GB, so the large passes hit sysmem fallback (prefix-cache 18 s,
50% recompute 9.7 s). Treat 16k as invalid except plain reuse. Clocks were otherwise healthy
(median 2,002 MHz, max 79 °C).

**Crossover:** plain reuse wins from 2k tokens. 20% recompute (the usable-quality setting) wins from
about 4k tokens and holds 1.4–1.6× at 4–8k. 50% recompute never beats full prefill with the current
per-layer Python overhead.

Note on the exact-prefix baseline: it gets *slower* than full prefill as context grows (1.5 s vs
0.84 s at 8k). Passing a past cache makes HF use SDPA with an explicit mask instead of the causal
fast path. A production prefix cache (vLLM) would not pay this, so this baseline is pessimistic.

## Recompute overhead pass (0.5B fp16, one 1k/4k/8k request, median of 5)

The RoPE cos/sin and causal mask for the recomputed positions are now built once per blend, not
once per layer.

| ctx | full | 10% recompute TTFT | 20% recompute TTFT | 50% recompute TTFT |
|---|---|---|---|---|
| 1k | 99 ms | 224 ms | 211 ms | 246 ms |
| 4k | 355 ms | 255 ms | 237 ms (was 257) | 387 ms |
| 8k | 831 ms | 406 ms | 473 ms (was 516) | 845 ms |

- Small gain (~8%). Blend still costs ~100 ms at 1k regardless of ratio: 24 layers of small kernel
  launches from Python. Removing that floor needs CUDA graphs (capture the per-layer subset forward
  at a few padded sizes). `torch.compile` is not a good option on Windows: its Triton backend isn't officially supported there.
- Tried and reverted: `F.scaled_dot_product_attention(..., enable_gqa=True)` to avoid `repeat_kv`
  copies. With a boolean mask, torch 2.5 falls back to the math kernel. It was 3× slower at 4k and ran out of memory at 8k
  (3.4 GB attention matrix).
- Tried and reverted: removing the `repeat_kv` copies without `enable_gqa`, by folding each KV group's
  query heads into the query axis (`q` as `[1, kv_heads, groups*n, d]`) or into the batch axis (K/V
  expanded as views). Outputs are bit-identical to `repeat_kv`. Blend time, 0.5B fp16, deviation policy,
  median of 5, old vs new: batch-axis fold 179 vs 177 ms (4k, 20%), 467 vs 458 ms (8k, 20%), 1001 vs
  996 ms (8k, 50%), i.e. within noise, with ~50-90 MiB higher peak memory. The query-axis fold was 8-17%
  slower and peaked 1.1 GB higher at 8k, because the mask has to be materialised `groups` times. So the
  KV copies are not the bottleneck at long context; the attention itself is.

## CUDA graphs for the recompute loop (`tread/graphs.py`, `Engine(..., graphs=True)`)

The layers 1..L recompute loop is captured as one CUDA graph per size bucket. Requests are padded into
the bucket, with padding query slots placed past the real context so the causal mask hides them.
Tests: graphed output matches eager (deviation 20%, boundary 30%), and at 100% recompute it
still reproduces full prefill.

Blend-stage time, 0.5B fp16, median of 5 after capture:

| ctx | recompute | eager | graphed |
|---|---|---|---|
| 1k | 10% | 95 ms | **23 ms** |
| 1k | 20% | 83 ms | **37 ms** |
| 4k | 20% | 130 ms | 148 ms |
| 8k | 20% | 323 ms | 377 ms |

- **Graphs remove the launch-overhead floor at short contexts (2–4× faster blend at 1k)**, which is
  where it dominated.
- **At ≥4k they lose.** Recompute is FLOP/bandwidth-bound there, and copying the cache into and out of
  the graph's static buffers costs more than the saved launches. Finer size buckets (multiples of 256
  instead of powers of 2) did not change that. The runner therefore only engages at ≤2,048 context
  tokens (`max_ctx`) and falls back to eager above.
- Each new bucket pays a one-off capture of 0.5–2 s, so graphs suit steady traffic with repeated
  shapes, not one-shot benchmarks.
- What would help at long context: assembling blocks *directly* into the static buffer (removes
  the copy-in). Avoiding `repeat_kv`'s 7× KV copies per layer was tested later and did not help
  (see "Tried and reverted" above).

## Next steps (in order of expected payoff)
1. **Find the crossover context length.** Build longer contexts (e.g. 20–40 docs, 4–8k tokens) where
   prefill becomes compute-bound, and plot TTFT vs context length for full prefill vs tread.
2. **Cut recompute overhead.** Fuse the per-layer subset forward (CUDA graphs / `torch.compile` on a
   fixed-size gather), so recompute cost scales with the ratio instead of being a fixed ~90 ms.
3. **Batch the cold path.** Prefill all missed docs in one forward with a block-diagonal mask.
4. **Larger-model timing on a rented GPU** (A100/L40S), where the 1.5B/7B numbers are trustworthy.
