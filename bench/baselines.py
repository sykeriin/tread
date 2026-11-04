"""External baselines on the same reordered-RAG workload as bench_scaling.py: vLLM and llama.cpp.

UNTESTED on this repo's dev machine (Windows, 4 GB): vLLM has no Windows support and the 1.5B+ models do
not fit. Written to run on a Linux GPU box (see bench/cloud/README.md); expect to fix small API drift.
Versions are pinned in bench/cloud/requirements-cloud.txt.

Protocol (identical to bench_scaling.py): per question, pass 1 sends the docs in order 1 (cold, fills the
engine's prefix cache), pass 2 sends the same docs in order 2 and is timed. Prompts are sent as token ids,
so every system sees exactly the same tokens. TTFT is measured as the latency of a 1-token generation,
which is prefill plus one decode step. tread's TTFT excludes that decode step, so these baselines are
slightly *pessimistic* by one decode (~tens of ms); say so when quoting.

Usage:
  python bench/baselines.py --engine vllm --model Qwen/Qwen2.5-1.5B-Instruct --lengths 1000 4000 8000
  python bench/baselines.py --engine llamacpp --gguf models/qwen2.5-1.5b-instruct-fp16.gguf --lengths 1000 4000 8000
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformers import AutoTokenizer  # noqa: E402

from workload import Workload  # noqa: E402


def make_vllm(args):
    from vllm import LLM, SamplingParams

    llm = LLM(args.model, dtype="float16", enable_prefix_caching=True, gpu_memory_utilization=0.85,
              max_model_len=max(args.lengths) + 1024, seed=0)
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    def run(ids):
        t0 = time.perf_counter()
        llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)
        return (time.perf_counter() - t0) * 1000

    return run, "vllm(prefix_caching)"


def make_llamacpp(args):
    from llama_cpp import Llama

    llm = Llama(model_path=args.gguf, n_ctx=max(args.lengths) + 1024, n_gpu_layers=-1, seed=0, verbose=False)

    def run(ids):
        # llama-cpp-python reuses the longest common token prefix with the previous request's KV.
        t0 = time.perf_counter()
        llm.create_completion(ids, max_tokens=1, temperature=0.0)
        return (time.perf_counter() - t0) * 1000

    return run, "llama.cpp(prompt_cache)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vllm", "llamacpp"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="HF id (tokenizer; vLLM weights)")
    ap.add_argument("--gguf", help="GGUF path for llama.cpp (same model/dtype as --model)")
    ap.add_argument("--lengths", type=int, nargs="+", default=[1000, 4000, 8000])
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="bench/results")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    wl = Workload(tok)  # CPU tensors: only token ids are needed here
    qs = Workload.questions(400, args.n)
    run, name = (make_vllm if args.engine == "vllm" else make_llamacpp)(args)

    res = {}
    for L in args.lengths:
        t_cold, t_warm, ctx = [], [], []
        for i, ex in enumerate(qs):
            docs, query, o1, o2, n_tok = wl.request(ex, L)
            ids1, ids2 = wl.flat(o1, query), wl.flat(o2, query)
            c = run(ids1)  # cold: fills the cache
            # Pass 2 is timed `reps` times; the first rep sees whatever the engine kept from pass 1, later reps
            # also see pass 2's own prefix. To time pass 2 as a reordered request, re-run pass 1 before each rep.
            ws = []
            for _ in range(args.reps):
                run(ids1)
                ws.append(run(ids2))
            if i >= args.warmup:
                t_cold.append(c)
                t_warm.append(statistics.median(ws))
                ctx.append(n_tok)
            print(f"{name} L={L} [{i + 1}/{len(qs)}] warm={statistics.median(ws):.0f}ms", flush=True)
        res[L] = dict(ctx_median=statistics.median(ctx), cold_p50=statistics.median(t_cold),
                      reordered_p50=statistics.median(t_warm))

    lines = [f"{name} model={args.model} n={args.n - args.warmup} per length reps={args.reps}",
             f"{'target':>7}{'ctx':>7}{'cold p50':>11}{'reordered p50':>16}"]
    lines += [f"{L:>7}{r['ctx_median']:>7.0f}{r['cold_p50']:>9.0f}ms{r['reordered_p50']:>14.0f}ms" for L, r in res.items()]
    print("\n".join(lines))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"baseline_{args.engine}_{Path(args.model).name}"
    (out / f"{tag}.txt").write_text("\n".join(lines), encoding="utf-8")
    (out / f"{tag}.json").write_text(json.dumps(dict(args=vars(args), results=res)))


if __name__ == "__main__":
    main()
