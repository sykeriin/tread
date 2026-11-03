# tread

Position-independent KV-cache reuse for LLM inference. Reuse the KV cache of a text chunk
(a RAG document, a tool output) wherever it appears in a prompt, not only when it is part of an
exact token prefix.

Today's serving engines (vLLM, SGLang, llama.cpp) reuse KV only on exact prefix matches. If the
same documents come back in a different order or mix, everything after the first change is
recomputed. tread caches each chunk as a content-addressed block, shifts it to its new position
with an exact RoPE rotation, and optionally recomputes a small, selected subset of tokens to repair
the cross-chunk attention that independent caching misses.

Status: Phase 1 (reuse engine + TTFT benchmark) in progress. See [SCOPE.md](SCOPE.md) for the
plan and [phase0/RESULTS.md](phase0/RESULTS.md) for the feasibility study.

## How it works

```
request = [system prompt][doc 1]...[doc N][query]      (sent as separate segments)

system prompt ──► exact-prefix block (cached)
each doc ──────► block keyed by hash(model, dtype, RoPE cfg, system prompt, doc tokens)
                  miss: prefill doc on top of the cached system prompt, store at base position P
                  hit:  rotate cached keys by (new offset − P); values need no change
optional ──────► selective recompute of a fraction of doc tokens (deviation-guided)
query ─────────► normal forward over the assembled cache → first token
```

- **Exact positional correction.** RoPE is a rotation, so keys cached at position *p* are moved
  to *p′* by rotating them by *p′ − p*. Tested to match in-place computation.
- **Shared-prefix chunk prefill.** Docs are cached after the system prompt rather than alone.
  This halves the reuse error (attention-sink effect; Phase 0).
- **Selective recompute.** Picks the tokens whose layer-1 KV drifts most from a fresh pass, and
  recomputes them through every layer against the reused cache (CacheBlend-style).
- **Block store.** Content-addressed, byte budget, LRU eviction.

## Layout

| path | what |
|---|---|
| `tread/blend.py` | RoPE re-rotation, chunk prefill, selective recompute |
| `tread/store.py` | content-addressed block store with budget + LRU |
| `tread/engine.py` | request assembly, plus full-prefill and exact-prefix baselines |
| `tests/` | correctness tests (exactness, cache hits, eviction) |
| `bench/bench_ttft.py` | TTFT benchmark on a reordered-RAG workload |
| `phase0/` | feasibility study: quality vs recompute, predictor and held-out experiments |

## Setup

Python 3.11, CUDA GPU (developed on an RTX 3050 laptop, 4 GB).

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

```bash
.venv/Scripts/python -m pip install transformers==4.46.3 accelerate==1.1.1 datasets==3.1.0 matplotlib pytest
```

`transformers` is pinned: the engine patches into Qwen2 attention internals that change between versions.

## Run

```bash
.venv/Scripts/python -m pytest tests -q
```

```bash
.venv/Scripts/python bench/bench_ttft.py --model Qwen/Qwen2.5-1.5B-Instruct --dtype float16 --n 50 --k 8
```

## Findings so far (Qwen2.5, HotpotQA)

- 100% recompute reproduces full prefill exactly, and the RoPE shift is exact. Reuse is a
  quality/compute trade-off, not a correctness bug.
- On 300 held-out examples (1.5B, fp16), KL vs full prefill drops from 0.263 (plain reuse) to
  0.099 / 0.070 / 0.029 at 20 / 30 / 50% recompute.
- Error is heavy-tailed: the worst 10% of requests carry about half of it.
- Adaptive per-request recompute budgets looked promising in-sample but **did not replicate on
  held-out data**. See phase0/RESULTS.md, Phase 0e.
- Partial recompute is not monotone per request: on some requests it is worse than plain reuse.
