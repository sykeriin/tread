"""TTFT vs context length: where does non-prefix reuse start paying off?

Same reordered-RAG protocol as bench_ttft.py, but each request is padded to a target context length
with distractor paragraphs drawn from other HotpotQA questions (validation 1000+, unused elsewhere).
Each question is asked twice with its docs in two random orders; pass 2 is measured.

Usage: python bench/bench_scaling.py --model Qwen/Qwen2.5-0.5B-Instruct --lengths 1000 2000 4000 8000 --n 15
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tread.engine import Engine, prefill_full  # noqa: E402
from tread.store import BlockStore  # noqa: E402
from workload import Workload  # noqa: E402

METHODS = ["full", "prefix_cache", "cold", "tread_r0.0", "tread_r0.2", "tread_r0.5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--lengths", type=int, nargs="+", default=[1000, 2000, 4000, 8000])
    ap.add_argument("--n", type=int, default=15, help="questions per length")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--store-mb", type=int, default=512)
    ap.add_argument("--allow-low-vram", action="store_true")
    ap.add_argument("--out", default="bench/results")
    args = ap.parse_args()

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).to(dev).eval()
    free = torch.cuda.mem_get_info()[0] >> 20
    need = args.store_mb + 1024  # long prefills need more activation headroom than bench_ttft
    print(f"free VRAM after model load: {free} MiB (need >= {need})", flush=True)
    if free < need and not args.allow_low_vram:
        sys.exit(f"only {free} MiB VRAM free; timings would hit WDDM sysmem fallback (see bench/RESULTS.md)")

    wl = Workload(tok, device=dev)
    qs = Workload.questions(400, args.n)
    prefix = wl.prefix

    results = {}
    for L in args.lengths:
        store = BlockStore(args.store_mb << 20)
        eng = Engine(model, store)
        res = {m: dict(ttft=[], kl=[], top1=[]) for m in METHODS}
        ctx_lens = []
        for i, ex in enumerate(qs):
            docs, query, o1, o2, n_tok = wl.request(ex, L)
            measured = i >= args.warmup

            def timed(fn, reps=args.reps):
                runs = sorted((fn() for _ in range(reps)), key=lambda x: x[2].t["ttft"])
                return runs[len(runs) // 2]

            def log(m, logits, st, ref):
                lp, lr = F.log_softmax(logits.float(), -1), F.log_softmax(ref.float(), -1)
                res[m]["ttft"].append(st.t["ttft"] * 1000)
                res[m]["kl"].append(F.kl_div(lp, lr, log_target=True, reduction="sum").item())
                res[m]["top1"].append(float(lp.argmax() == lr.argmax()))

            # Pass 1 (cold) populates the store; compare it with full prefill of the same order.
            logits, _, st = eng.prefill(prefix, o1, query)
            ref1 = prefill_full(model, prefix, o1, query)[0]
            if measured:
                log("cold", logits, st, ref1)
            del ref1
            # Pass 2 (reordered, warm).
            ref, _, st = timed(lambda: prefill_full(model, prefix, o2, query))
            if measured:
                log("full", ref, st, ref)
                ctx_lens.append(n_tok)
            logits, _, st = timed(lambda: eng.prefill_prefix_only(prefix, o2, query))
            if measured:
                log("prefix_cache", logits, st, ref)
            for r in [0.0, 0.2, 0.5]:
                logits, _, st = timed(lambda: eng.prefill(prefix, o2, query, ratio=r, policy="deviation"))
                if measured:
                    log(f"tread_r{r}", logits, st, ref)
            torch.cuda.empty_cache()
            print(f"L={L} [{i + 1}/{len(qs)}] ctx={n_tok} docs={len(docs)}", flush=True)
        results[L] = dict(ctx_median=statistics.median(ctx_lens), res=res)

    lines = [f"model={args.model} dtype={args.dtype} n={args.n - args.warmup} per length, reps={args.reps}",
             f"{'target':>7}{'ctx':>7}" + "".join(f"{m:>16}" for m in METHODS)]
    for L, r in results.items():
        full = statistics.median(r["res"]["full"]["ttft"])
        cells = []
        for m in METHODS:
            p50 = statistics.median(r["res"][m]["ttft"])
            cells.append(f"{p50:7.0f}ms {full / p50:4.2f}x")
        lines.append(f"{L:>7}{r['ctx_median']:>7.0f}" + "".join(f"{c:>16}" for c in cells))
    lines.append("\nfirst-token top-1 agreement with full prefill")
    for L, r in results.items():
        lines.append(f"{L:>7}" + "".join(f"{statistics.mean(r['res'][m]['top1']):>16.3f}" for m in METHODS))
    text = "\n".join(lines)
    print(text)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"scaling_{Path(args.model).name}"
    (out / f"{tag}.txt").write_text(text, encoding="utf-8")
    (out / f"{tag}.json").write_text(json.dumps(dict(args=vars(args), results=results)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    xs = [r["ctx_median"] for r in results.values()]
    for m in METHODS:
        ys = [statistics.median(r["res"][m]["ttft"]) for r in results.values()]
        ax[0].plot(xs, ys, marker="o", label=m)
        ax[1].plot(xs, [statistics.mean(r["res"][m]["top1"]) for r in results.values()], marker="o", label=m)
    ax[0].set_xlabel("context tokens"); ax[0].set_ylabel("TTFT p50 (ms)"); ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[1].set_xlabel("context tokens"); ax[1].set_ylabel("first-token top-1 vs full"); ax[1].set_xscale("log")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle(f"TTFT and quality vs context length — {Path(args.model).name}")
    fig.tight_layout()
    fig.savefig(out / f"{tag}.png", dpi=120)


if __name__ == "__main__":
    main()
