# Position-independent KV-cache reuse for reordered RAG contexts: an engineering study

**Status: draft technical report.** All numbers come from Qwen2.5 models on HotpotQA, measured on a single
RTX 3050 laptop (4 GB). Large-model timings and vLLM/llama.cpp baselines are **not yet measured** (scripts in
`bench/cloud/`, untested). Raw data and per-run caveats: `phase0/RESULTS.md`, `bench/RESULTS.md`.

## 1. Problem

Serving engines reuse KV cache only for an exact token prefix. In RAG, the same documents recur in different
orders and combinations, so everything after the first difference is recomputed. Two obstacles make
non-prefix reuse non-trivial:

1. **Positions.** RoPE bakes absolute position into cached keys.
2. **Missing cross-chunk attention.** A chunk cached on its own never attended to the chunks before it, so its
   deeper-layer K/V differ from what a full prefill would produce.

Obstacle 1 is exact to fix. Obstacle 2 is the real problem and can only be reduced, not removed, short of
recomputing.

## 2. System

`tread` (this repo) is a serving layer over HF transformers:

- **Content-addressed block store.** Key = hash(model, dtype, RoPE config, namespace, system prompt, chunk
  tokens). GPU tier with byte budget and LRU, spilling to a pinned-CPU tier.
- **Exact RoPE re-rotation.** Keys cached at base position *P* move to *P'* by rotating by *P' − P*
  (cos/sin of the delta; HF's `attention_scaling` divided out). Unit-tested against in-place computation.
- **Shared-prefix chunk prefill.** Each chunk is prefilled *after* the cached system prompt, so it sees the
  attention-sink tokens. Misses in one request are prefilled in one packed forward with a block-diagonal mask.
- **Selective recompute.** A fraction of chunk tokens is recomputed through all layers against the assembled
  cache and spliced back. Tokens are chosen by the drift of their layer-1 K/V from a fresh layer-0→1 pass
  (CacheBlend-style). Optional CUDA-graph replay of the layer loop for contexts ≤ 2k tokens.
- **Segment API.** Requests are `{prefix, documents[], suffix}`, tokenized separately, which removes
  tokenization-boundary ambiguity. Document text is tokenized with special tokens split; blocks are namespaced
  per tenant.

Correctness invariants (all tested): 100% recompute reproduces full prefill (KL < 1e-5); a reordered request
hits every block and equals a cold assembly; batched-miss prefill equals per-doc prefill; graphed recompute
equals eager; eviction/offload preserve results.

## 3. Quality: what reuse costs, and what recompute buys back

Setup: HotpotQA distractor, 2 gold + 2 distractor paragraphs, shuffled (~400–650 tokens). Metrics are
teacher-forced KL(full ‖ reuse) and top-1 agreement over the query end plus a 32-token continuation.

| finding | evidence |
|---|---|
| Caching after the system prompt halves the error vs caching alone | 1.5B n=100: KL 0.291 vs 0.379, paired Δ −0.087 [−0.156, −0.018]; 0.5B n=30: 0.243 vs 0.528 |
| Recompute reduces error monotonically on average | 1.5B, n=300 held-out: KL 0.263 (none) → 0.099 / 0.070 / 0.029 at 20 / 30 / 50% |
| Drift-based token choice beats random | paired ΔKL vs random significant at 10/20/30% (e.g. −0.124 [−0.212, −0.038] at 30%) |
| Drift-based choice vs first-tokens-of-each-doc | significant only at 10%; tied at 20–50% |
| Error is heavy-tailed | worst 10% of requests hold ~50–55% of naive KL |

**Negative result: adaptive per-request budgets.** A query-attention signal (share of the question's attention
on documents 2..N) cut KL 20–30% at equal recompute under repeated 2-fold CV on 100 examples. Fit on those 100
and scored once on 300 fresh examples, the gain vanished (ΔKL ≈ 0 at every budget; layer-1 drift was
significantly *worse*). Answer-token entropy correlates with hardness (ρ 0.47) but does not help allocate
recompute. A per-example oracle shows large headroom (KL 0.044 at 30% budget), so a good predictor would
matter, but none of the cheap signals tried here finds it. Methodological lesson: cross-validation on n=100 of a
heavy-tailed target overfits policy parameters; only fresh data caught it.

**Partial recompute is not monotone per request.** At k=8 documents, recompute at 10–30% made the first-token
distribution *worse* than plain reuse on 12–15 of 47 requests on 1.5B (5/47 at 50%) and 7–11 of 47 on 0.5B
(1/47 at 50%). A half-repaired cache can be less self-consistent than a uniformly stale one.

**Model size is not compared.** The 0.5B (n=30) and 1.5B (n=100) Phase 0 runs used different example sets, and
the first 100 examples were harder than the 300 held-out ones (deviation@30% KL 0.131 vs 0.070), so no claim
about robustness vs. model size is supported.

**Quality gets harder with more documents.** With 8 documents plain reuse matched full prefill's first token
only 55–64% of the time (vs ~82–88% teacher-forced at 4 documents).

## 4. Latency (Qwen2.5-0.5B fp16, reordered-RAG, same-codebase baselines)

Method: each question asked twice with documents in two different random orders; second ask timed
(median of 3, 13 questions per length). GPU clocks logged; run valid only with VRAM headroom (see §6).

| context | full prefill | plain reuse | 20% recompute | 50% recompute |
|---|---|---|---|---|
| 1k | 96 ms | 95 ms (1.0×) | 194 ms (0.5×) | 189 ms (0.5×) |
| 2k | 180 ms | 97 ms (1.9×) | 193 ms (0.9×) | 218 ms (0.8×) |
| 4k | 364 ms | 116 ms (3.1×) | 257 ms (1.4×) | 400 ms (0.9×) |
| 8k | 841 ms | 146 ms (5.8×) | 516 ms (1.6×) | 938 ms (0.9×) |

After removing per-layer RoPE/mask rebuilds, 20% recompute at 4k / 8k was 237 / 473 ms (single request, median
of 5). First-token top-1 agreement with full prefill (13 questions per length, so coarse): plain reuse 15–69%,
20% recompute 69–92%, 50% recompute 69–92%.

Reading: recompute cost is dominated by per-layer kernel-launch overhead, flat at ~100 ms regardless of
ratio, until contexts are long enough to be FLOP-bound. Reuse therefore only pays off from ~2k tokens (plain) or
~4k tokens (20% recompute), and the usable setting buys 1.4–1.6× with a measurable quality loss, not a free
speedup. CUDA graphs cut blend time 2–4× at 1k tokens but lose to eager execution at ≥4k (cache copy into
static buffers outweighs saved launches).

## 5. Limitations

- **Small models, one family, one dataset.** Qwen2.5-0.5B/1.5B on HotpotQA only. No 7B, no other families, no
  long-context benchmarks beyond padded HotpotQA.
- **Latency on a laptop GPU, 0.5B only.** 1.5B timings were invalid (VRAM spill); 16k results partly invalid.
- **No external baselines yet.** vLLM and llama.cpp numbers are missing. The in-repo exact-prefix baseline is
  pessimistic at long context (HF takes a slower attention path with a cache).
- **First-token quality metric in latency runs**, n=13 per length. Teacher-forced multi-token metrics are only
  in the Phase 0 study (shorter contexts).
- **Task accuracy is unresolved.** F1 at n=30–100 was too noisy to rank methods. No end-to-end accuracy claim is made.
- **Recompute is the whole speed/quality trade-off**, and its per-request non-monotonicity is unsolved.
- Single-threaded engine; no batching across requests; no decode-phase optimization.

## 6. Pitfall worth recording: silent VRAM spill

On Windows (WDDM), a GPU with no free memory silently spills to system RAM. Large passes slowed 3–9× while
small ones did not, producing fake "14–26×" speedups. Fix: the benchmarks now refuse to run without free VRAM
headroom. Any GPU timing study on consumer Windows hardware should check this.

## 7. Next steps

1. Run `bench/cloud/run_all.sh` on a 24–80 GB GPU: 1.5B/7B, vLLM and llama.cpp baselines, 16k contexts.
2. Multi-token and task-accuracy evaluation at n ≥ 300.
3. Assemble blocks directly into the graph buffer and use a GQA-aware masked attention kernel to make recompute
   win at long context.
4. A better recompute-allocation signal, validated on held-out data from the start.
