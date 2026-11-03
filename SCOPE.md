# tread — Position-Independent KV-Cache Reuse for LLM Inference

A standalone serving layer that reuses KV cache for text chunks (RAG docs, tool outputs)
regardless of where they appear in the prompt, with positional correction and bounded,
*measured* accuracy loss.

Timeline: ~14–16 weeks, solo. Backend: HF transformers (PyTorch). Two writeups: an
engineering blog post (ML systems roles) and an arXiv-style technical report (research).

---

## 1. Architecture

```
request ──► Chunker ──► Block Store lookup (content hash) ──► Planner
                                                               │
            ┌──────────────────────────────────────────────────┘
            ▼
  [hit blocks] ──► RoPE re-rotation to new positions ──┐
  [miss blocks] ─► normal prefill ─────────────────────┤
                                                       ▼
                                   Blender: selective recompute
                                   (boundary window + top-k deviation tokens)
                                                       ▼
                                   Assembled KV ──► decode (HF generate w/ past_key_values)
```

### Components
1. **Chunker.** Splits prompt into segments: system prompt, each retrieved doc, query.
   Chunk boundaries come from the API (caller sends a list of segments), not from
   heuristics. That avoids tokenization-boundary bugs and is how real RAG systems work.
2. **Content-addressed block store.** Key = `hash(model_id, tokenizer_id, token_ids, dtype)`.
   The value is the per-layer K (stored *un-rotated*, or rotated at position 0) and V,
   plus metadata: token count, creation time, hit count, validity flags. Storage tiers
   are GPU, then pinned CPU RAM, then (stretch) disk.
3. **Positional correction.** RoPE is a rotation, so `R(p') = R(p'−p)·R(p)`. Cache keys
   computed at position 0 and rotate them by `R(offset)` at assembly time. V needs no
   correction. This step is mathematically exact. **What is *not* exact is the missing
   cross-chunk attention**: a chunk prefilled alone never attended to earlier chunks, so
   its layer ≥1 hidden states (and hence K/V) differ. That gap is what blending fixes.
4. **Blender (bounded-error recompute).** Recompute a subset of tokens with full
   attention over the assembled cache, then splice their fresh K/V back in. Two policies
   to compare:
   - *Boundary window*: first `w` tokens of each reused chunk.
   - *Deviation-guided* (CacheBlend-style): run full attention for layer 1 only, pick the
     tokens whose KV deviates most, recompute those through later layers.
   The recompute ratio `r` is the knob, and every result is reported as a quality-vs-`r` curve.
5. **Validity & eviction.** Each block is tagged with model/tokenizer/RoPE-config/dtype
   version. It is invalidated if any changes. Eviction uses LRU weighted by
   `(hits × tokens saved) / bytes`, under a fixed byte budget. Blocks get refcounts while in
   use by an in-flight request.
6. **Server.** A thin FastAPI with an OpenAI-ish `/v1/completions` that accepts a
   `segments: [...]` field. Metrics are exported per request (TTFT, tokens reused,
   tokens recomputed, cache bytes).

### Non-goals
Batching/scheduling, paged attention, multi-GPU, custom allocator (that's the later
project), production-grade throughput. Custom kernels are optional (see §5).

---

## 2. Eval methodology

### Workloads
- **RAG reorder:** a fixed corpus of documents, where each query retrieves k=4–10 docs in
  varying order and combination. Datasets: 2WikiMultiHopQA, MuSiQue, HotpotQA (as used
  in the CacheBlend line of work), plus a summarization set (e.g. MultiNews).
- **Overlap sweep:** control the fraction of chunks previously seen (0%, 25%, 50%, 100%)
  and the degree of reordering.
- **Tool-output reuse (stretch):** an agent trace where the same tool outputs reappear.

### Systems compared
| System | Purpose |
|---|---|
| HF full prefill (no cache) | quality ground truth, same-codebase TTFT baseline |
| HF exact-prefix cache | what "prefix caching" gives inside the same codebase |
| tread, naive reuse (r=0, no blending) | shows why blending is needed |
| tread, boundary blend / deviation blend | the contribution |
| vLLM w/ prefix caching | external reference |
| llama.cpp w/ prompt cache | external reference |

**Fairness caveat (state it in the writeup):** HF is slower than vLLM in absolute terms.
Make the headline claim **relative to the same-codebase baselines** (HF full / HF prefix).
Report vLLM/llama.cpp absolute numbers alongside, and argue the reuse speedup is
orthogonal and would compose with a faster engine.

### Metrics
- **Latency:** TTFT (p50/p95), split into lookup, rotation, recompute, and prefill;
  end-to-end latency.
- **Savings:** % prefill tokens skipped and FLOPs saved (analytical).
- **Quality:** task F1/EM (QA), ROUGE-L (summ.), and **KL divergence / top-1 agreement of
  next-token distributions vs full prefill** over the first N generated tokens. That
  makes the error measurement direct, rather than inferring it from task scores.
  Also perplexity delta on held-out continuations.
- **Cache:** hit rate, bytes, eviction churn under a byte budget.

### Rigor
Greedy decoding for determinism, ≥3 seeds for sampled runs, bootstrap 95% CIs on quality,
warm-up runs excluded, and GPU clocks noted. Every plot shows the quality-vs-recompute-ratio
Pareto curve.

---

## 3. Hardware & models

**Your RTX 3050 laptop (4–6 GB VRAM)** is fine for *development*, not for headline results.

| Phase | Hardware | Model |
|---|---|---|
| Dev / correctness (wk 1–10) | 3050 laptop | Qwen2.5-0.5B / 1.5B-Instruct (fp16) |
| Final benchmarks (wk 11–14) | rented 1× **L40S 48GB** or **A100 80GB** | Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Mistral-7B-Instruct |

- **What to acquire:** don't buy a GPU. Rent from RunPod / Lambda / Vast.ai.
  A 24GB card (4090/L4) is enough for 8B fp16 at ≤8k context. Get an A100 80GB for
  16–32k context sweeps. Expect roughly **40–80 GPU-hours total, ≈ $50–150**. Script
  everything so cloud runs are one command.
- **Why these models:** all use RoPE and GQA, which is representative. Using two model
  families shows that the result generalizes. Qwen small models share the architecture
  with the 7B, so code written on the laptop transfers unchanged.
- Check RoPE variants per model (Llama 3.1 uses scaled RoPE). The rotation must use the
  model's own `rotary_emb`, not a hand-rolled formula.

---

## 4. Main technical risks

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| 1 | **Cross-chunk attention loss** makes quality drop more than expected even with blending | Core claim fails | Week 1–2 go/no-go experiment (§5); report honestly as a Pareto curve either way |
| 2 | **HF caching internals** (`DynamicCache`, RoPE applied inside attention) fight surgery; API churns between versions | Weeks lost | Pin `transformers` version; write own attention forward wrapper for 1–2 model classes only |
| 3 | **Selective recompute isn't actually faster** in eager PyTorch (gather/scatter overhead, small-model launch-bound) | No TTFT win at small scale | Measure at 7–8B and ≥4k context where prefill is compute-bound; use SDPA/FlashAttention with masks; Triton only if profiling demands |
| 4 | **Tokenization boundary effects:** a chunk tokenizes differently in context vs alone | Silent cache misses / wrong KV | Tokenize segments independently by API contract; test round-trips |
| 5 | **Attention sinks / BOS:** chunks prefilled alone lack the BOS sink, which skews attention | Quality loss | Prefill each chunk with a shared prefix (system prompt/BOS) and strip it; ablate this |
| 6 | **Unfair baselines** draw reviewer/interviewer skepticism | Credibility | Same-codebase baselines as headline; disclose configs |
| 7 | **Scope creep** (llama.cpp port, kernels, disk tier) | Miss deadline | All three are explicit stretch goals |

On kernels: yes, I can write the Triton code with you. But plan for you to understand
every line, because interviewers will ask. The design above deliberately needs **no
custom kernel** for a complete project; a fused rotate-and-gather kernel is a stretch
goal once profiling shows it matters.

---

## 5. Build order (de-risk first)

**Phase 0 — Go/no-go notebook (weeks 1–2, laptop, Qwen2.5-0.5B)**
1. Prefill a prompt `[sys][docA][docB][query]` in full and record KV and logits.
2. Prefill docA and docB separately at position 0, re-rotate K to their real offsets,
   and concatenate.
3. Verify layer-0 K matches the full prefill to ~1e-3 (proves the rotation is correct).
4. Plot per-layer KV deviation and next-token KL vs full prefill (shows the cross-attention gap).
5. Add boundary-window recompute and top-k deviation recompute. Plot KL vs recompute ratio.

✅ Go if ~15–20% recompute brings KL / top-1 agreement close to full prefill on a
handful of QA examples. ❌ Otherwise, pivot the framing to "characterizing the error,"
which is still a publishable negative/analysis result.

**Phase 1 — Core engine (weeks 3–6):** block store, hashing, rotation module, blender
policies, and correctness tests against full prefill. Unit tests: rotation exactness, and
cache hit/miss with identical output when r=1.0.

**Phase 2 — Serving + eviction (weeks 7–9):** FastAPI server, validity tags,
budgeted eviction, CPU offload tier, metrics.

**Phase 3 — Benchmarks (weeks 10–13):** workload generator, baselines (vLLM,
llama.cpp, HF), cloud runs on 7–8B, ablations (window size, policy, sink handling,
context length).

**Phase 4 — Writeups (weeks 14–16):** blog post (engineering story, profiling,
TTFT charts) and tech report (method, error analysis, Pareto curves, comparison
to CacheBlend).

**Stretch:** Triton fused rotate+gather, llama.cpp port, disk tier, agent/tool workload.

---

## Open questions
- Exact VRAM on the 3050 (4 GB vs 6 GB) decides whether 1.5B fits for dev.
- The papers cited (arXiv 2605.18825, 2606.21238) should be read in Phase 0. The
  report's novelty claim depends on what they already cover.
