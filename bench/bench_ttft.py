"""TTFT benchmark: non-prefix reuse vs full prefill vs exact-prefix caching, same HF codebase.

Workload ("reordered RAG"): each HotpotQA question is asked twice, with its k docs (2 gold +
distractors) in two different random orders. Pass 1 is cold (populates the block store);
pass 2 is the measured warm request. Exact-prefix caching can only reuse the system prompt on
pass 2 because the doc order changed; tread reuses every doc block.

Quality: KL(full ‖ method) and top-1 agreement of the first answer token's distribution.
Phase 0 has the fuller teacher-forced quality eval; this script is about latency.

Usage: python bench/bench_ttft.py --model Qwen/Qwen2.5-1.5B-Instruct --dtype float16 --n 50 --k 8
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tread.engine import Engine, prefill_full  # noqa: E402
from tread.store import BlockStore  # noqa: E402

SYSTEM = ("<|im_start|>system\nAnswer the question using the documents. "
          "Reply with a short phrase only.<|im_end|>\n<|im_start|>user\n")
RATIOS = [0.0, 0.1, 0.2, 0.3, 0.5]


def build(ex, tok, k, dev):
    titles, sents = ex["context"]["title"], ex["context"]["sentences"]
    gold = set(ex["supporting_facts"]["title"])
    idx = [i for i, t in enumerate(titles) if t in gold] + [i for i, t in enumerate(titles) if t not in gold]
    enc = lambda s: torch.tensor(tok(s, add_special_tokens=False).input_ids, device=dev)
    docs = [enc(f"Document: {titles[i]}\n{''.join(sents[i])}\n\n") for i in idx[:k]]
    query = enc(f"Question: {ex['question']}<|im_end|>\n<|im_start|>assistant\n")
    return enc(SYSTEM), docs, query


def shuffled(docs, g):
    return [docs[i] for i in torch.randperm(len(docs), generator=g)]


def pct(x, p):
    x = sorted(x)
    return x[min(len(x) - 1, int(round(p * (len(x) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--k", type=int, default=8, help="docs per request")
    ap.add_argument("--start", type=int, default=400, help="first validation example (unused by Phase 0)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3, help="timed repeats per warm method; median is kept")
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--allow-low-vram", action="store_true")
    ap.add_argument("--store-mb", type=int, default=256,
                    help="block store budget. On a 4 GB card, model + store must leave room for activations, "
                         "or Windows spills VRAM to system RAM and every timing is wrong")
    args = ap.parse_args()

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).to(dev).eval()
    # On WDDM (Windows) a full GPU silently spills VRAM to system RAM, and big prefills slow down
    # 3-9x while small ones don't, which inflates every speedup. Refuse to time in that state.
    free = torch.cuda.mem_get_info()[0] >> 20
    need = args.store_mb + 512
    print(f"free VRAM after model load: {free} MiB (need >= {need})", flush=True)
    if free < need and not args.allow_low_vram:
        sys.exit(f"only {free} MiB VRAM free: close other GPU apps, shrink --store-mb, use a smaller model, "
                 f"or pass --allow-low-vram (timings will not be trustworthy)")
    data = load_dataset("hotpotqa/hotpot_qa", "distractor", split=f"validation[{args.start}:{args.start + args.n}]")
    store = BlockStore(args.store_mb << 20)  # pass 2 follows pass 1 directly, so evicting old questions is harmless
    eng = Engine(model, store)
    g = torch.Generator().manual_seed(0)

    methods = ["full", "prefix_cache", "cold"] + [f"tread_r{r}" for r in RATIOS]
    res = {m: dict(ttft=[], kl=[], top1=[], prefilled=[], eff=[]) for m in methods}
    ctx_lens = []

    def log(m, logits, st, ref):
        lp, lr = F.log_softmax(logits.float(), -1), F.log_softmax(ref.float(), -1)
        res[m]["ttft"].append(st.t["ttft"] * 1000)
        res[m]["kl"].append(F.kl_div(lp, lr, log_target=True, reduction="sum").item())
        res[m]["top1"].append(float(lp.argmax() == lr.argmax()))
        res[m]["prefilled"].append(st.tokens_prefilled)
        res[m]["eff"].append(st.recompute_eff)

    for i, ex in enumerate(data):
        prefix, docs, query = build(ex, tok, args.k, dev)
        order1, order2 = shuffled(docs, g), shuffled(docs, g)
        measured = i >= args.warmup

        # Pass 1, cold: populates the store. Its latency is the miss-path cost.
        logits, _, st = eng.prefill(prefix, order1, query)
        ref1, _, _ = prefill_full(model, prefix, order1, query)
        if measured:
            log("cold", logits, st, ref1)

        # Pass 2, reordered: the measured warm request. The GPU is shared with other apps on this
        # laptop, so each method is timed `reps` times and the median kept.
        def timed(fn):
            runs = [fn() for _ in range(args.reps)]
            runs.sort(key=lambda x: x[2].t["ttft"])
            return runs[len(runs) // 2]

        ref, _, st = timed(lambda: prefill_full(model, prefix, order2, query))
        if measured:
            log("full", ref, st, ref)
            ctx_lens.append(len(prefix) + sum(len(d) for d in docs))
        logits, _, st = timed(lambda: eng.prefill_prefix_only(prefix, order2, query))
        if measured:
            log("prefix_cache", logits, st, ref)
        for r in RATIOS:
            logits, _, st = timed(lambda: eng.prefill(prefix, order2, query, ratio=r, policy="deviation"))
            if measured:
                log(f"tread_r{r}", logits, st, ref)
        print(f"[{i + 1}/{len(data)}]", flush=True)

    full_p50 = statistics.median(res["full"]["ttft"])
    lines = [f"model={args.model} dtype={args.dtype} k={args.k} n={len(ctx_lens)} "
             f"ctx tokens median={statistics.median(ctx_lens):.0f}",
             f"{'method':<14}{'TTFT p50':>10}{'p95':>9}{'speedup':>9}{'prefilled':>11}{'eff':>6}{'KL':>9}{'top1':>7}"]
    for m in methods:
        v = res[m]
        p50 = statistics.median(v["ttft"])
        lines.append(f"{m:<14}{p50:>8.1f}ms{pct(v['ttft'], 0.95):>7.1f}ms{full_p50 / p50:>8.2f}x"
                     f"{statistics.mean(v['prefilled']):>11.0f}{statistics.mean(v['eff']):>6.2f}"
                     f"{statistics.mean(v['kl']):>9.4f}{statistics.mean(v['top1']):>7.3f}")
    # Partial recompute is not monotone per example: count requests it makes worse than plain reuse.
    base = res["tread_r0.0"]["kl"]
    for r in RATIOS[1:]:
        kl = res[f"tread_r{r}"]["kl"]
        worse = sum(b > a + 1e-3 for a, b in zip(base, kl))
        lines.append(f"r={r}: worse than plain reuse on {worse}/{len(kl)} requests")
    lines.append(f"store: {len(store)} blocks, {store.stats.bytes_used / 2**20:.0f} MiB, "
                 f"hits={store.stats.hits} misses={store.stats.misses} evictions={store.stats.evictions}")
    text = "\n".join(lines)
    print(text)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(args.model).name}_k{args.k}"
    (out / f"{tag}.txt").write_text(text, encoding="utf-8")
    (out / f"{tag}.json").write_text(json.dumps(dict(args=vars(args), ctx_lens=ctx_lens, res=res)))


if __name__ == "__main__":
    main()
