"""Phase 0 go/no-go: how much quality does non-prefix KV reuse lose, and how much recompute buys it back?

Workload: HotpotQA (distractor) — 2 gold + 2 distractor paragraphs, shuffled, one doc per chunk.
Reference: full prefill of [prefix][doc1..docN][query]. Everything is compared against it.

Usage: python phase0/run_phase0.py --n 30 --dtype float32
"""
import argparse
import json
import re
import string
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from blend import assemble, blend, prefill_kv, to_cache

SYSTEM = ("<|im_start|>system\nAnswer the question using the documents. "
          "Reply with a short phrase only.<|im_end|>\n<|im_start|>user\n")
RATIOS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
POLICIES = ["boundary", "deviation", "random"]


def f1(pred, gold):
    norm = lambda s: " ".join(w for w in re.sub(r"\b(a|an|the)\b", " ",
                              "".join(c for c in s.lower() if c not in string.punctuation)).split())
    p, g = norm(pred).split(), norm(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return float(p == g)
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def build_example(ex, tok, rng, n_distract=2):
    titles, sents = ex["context"]["title"], ex["context"]["sentences"]
    gold = set(ex["supporting_facts"]["title"])
    idx = [i for i, t in enumerate(titles) if t in gold]
    idx += [i for i, t in enumerate(titles) if t not in gold][:n_distract]
    idx = [idx[i] for i in torch.randperm(len(idx), generator=rng)]
    enc = lambda s: torch.tensor(tok(s, add_special_tokens=False).input_ids)
    docs = [enc(f"Document: {titles[i]}\n{''.join(sents[i])}\n\n") for i in idx]
    query = enc(f"Question: {ex['question']}<|im_end|>\n<|im_start|>assistant\n")
    return enc(SYSTEM), docs, query


@torch.inference_mode()
def greedy(model, cache, first_ids, n, eos):
    out, ids = [], first_ids
    for _ in range(n):
        o = model(ids[None], past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        nxt = o.logits[0, -1].argmax()
        if nxt.item() in eos:
            break
        out.append(nxt.item())
        ids = nxt[None]
    return out


@torch.inference_mode()
def score(model, tok, K, V, query, cont, ref_logp, eos, answers=True):
    """Teacher-forced KL / top-1 agreement vs full prefill over query-end + continuation, plus answer."""
    logits = model(torch.cat([query, cont[:-1]])[None], past_key_values=to_cache(K, V)).logits[0]
    logp = F.log_softmax(logits[len(query) - 1:].float(), -1)
    kl = F.kl_div(logp, ref_logp, log_target=True, reduction="none").sum(-1).mean().item()
    top1 = (logp.argmax(-1) == ref_logp.argmax(-1)).float().mean().item()
    ans = tok.decode(greedy(model, to_cache(K, V), query, 24, eos)).strip().split("\n")[0] if answers else ""
    return kl, top1, ans


def rel_err(A, B):
    return [((a - b).norm() / b.norm()).item() for a, b in zip(A, B)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", default="phase0/results")
    ap.add_argument("--start", type=int, default=0, help="first validation example (held-out sets use >0)")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--modes", default="alone,shared")
    ap.add_argument("--no-answers", action="store_true", help="skip greedy answers / F1 (KL and top-1 only)")
    args = ap.parse_args()
    policies, modes = args.policies.split(","), args.modes.split(",")

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).to(dev).eval()
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    data = load_dataset("hotpotqa/hotpot_qa", "distractor", split=f"validation[{args.start}:{args.start + args.n}]")
    # Separate streams: doc order must not shift when the policy/ratio grid changes.
    # Seeded by --start so predictor.py can replay the same doc order.
    data_rng = torch.Generator().manual_seed(args.start)

    # Checkpoint after every example so a killed or stalled run resumes where it stopped.
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "ckpt.json"
    rows, layer_err = [], defaultdict(list)
    if ckpt.exists():
        state = json.loads(ckpt.read_text())
        rows, layer_err = state["rows"], defaultdict(list, state["layer_err"])
    done = {r["ex"] for r in rows}
    if done:
        print(f"resuming: {len(done)} examples already done", flush=True)

    t0 = time.time()
    for ei, ex in enumerate(data, start=args.start):
        # Always draw the doc order, even for finished examples, to keep data_rng in sync.
        prefix, docs, query = build_example(ex, tok, data_rng)
        if ei in done:
            continue
        # Per-example seed so resumed runs draw the same random-policy tokens.
        pol_rng = torch.Generator().manual_seed(1000 + ei)
        prefix, docs, query = prefix.to(dev), [d.to(dev) for d in docs], query.to(dev)
        ctx = torch.cat([prefix] + docs)

        with torch.inference_mode():
            Kf, Vf = prefill_kv(model, ctx)
            cont = torch.tensor(greedy(model, to_cache(Kf, Vf), query, 32, eos) or [tok.eos_token_id], device=dev)
            # Only project the scored positions through lm_head.
            hidden = model.model(torch.cat([ctx, query, cont[:-1]])[None]).last_hidden_state[0]
            ref_logp = F.log_softmax(model.lm_head(hidden[len(ctx) + len(query) - 1:]).float(), -1)

        cands, off = [], len(prefix)
        for d in docs:
            cands.append(torch.arange(off, off + len(d), device=dev))
            off += len(d)
        doc_pos = torch.cat(cands)

        def record(method, mode, ratio, K, V, eff=None):
            kl, top1, ans = score(model, tok, K, V, query, cont, ref_logp, eos, answers=not args.no_answers)
            rows.append(dict(ex=ei, method=method, mode=mode, ratio=ratio, eff=ratio if eff is None else eff,
                             kl=kl, top1=top1,
                             f1=f1(ans, ex["answer"]), ans=ans, gold=ex["answer"], ctx_len=len(ctx)))

        record("full", "-", 1.0, Kf, Vf)
        with torch.inference_mode():
            for mode in modes:
                Ka, Va = assemble(model, prefix, docs, mode)
                # Layer 0 keys depend only on token + position: this checks the RoPE shift is exact.
                l0 = ((Ka[0][:, :, doc_pos] - Kf[0][:, :, doc_pos]).abs().max().item())
                ke = rel_err([k[:, :, doc_pos] for k in Ka], [k[:, :, doc_pos] for k in Kf])
                layer_err[mode].append(ke)
                record("naive", mode, 0.0, Ka, Va)
                rows[-1]["l0_max_abs"] = l0
                rows[-1]["l0_rel"] = ke[0]
                if mode == "shared":
                    for pol in policies:
                        for r in RATIOS:
                            Kb, Vb, _, eff = blend(model, ctx, Ka, Va, cands, r, pol, gen=pol_rng)
                            record(pol, mode, r, Kb, Vb, eff)
                    # Sanity: recomputing every doc token must reproduce full prefill.
                    Kb, Vb, _, eff = blend(model, ctx, Ka, Va, cands, 1.0, "boundary")
                    record("boundary", mode, 1.0, Kb, Vb, eff)
        ckpt.write_text(json.dumps({"rows": rows, "layer_err": layer_err}))
        print(f"[{ei - args.start + 1}/{len(data)}] ctx={len(ctx)} tok  {time.time() - t0:.0f}s", flush=True)

    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    (out / "layer_err.json").write_text(json.dumps({m: torch.tensor(v).mean(0).tolist() for m, v in layer_err.items()}))
    summarize(rows, out)


def bootstrap_ci(x, n=2000, seed=0):
    """95% percentile bootstrap CI of the mean, resampling examples."""
    t = torch.tensor(x, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    means = t[torch.randint(len(t), (n, len(t)), generator=g)].mean(1)
    return means.quantile(0.025).item(), means.quantile(0.975).item()


def summarize(rows, out):
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["method"], r["mode"], r["ratio"])
        for m in ["kl", "top1", "f1", "eff"]:
            agg[key][m].append(r[m])
    lines = [f"{'method':<10}{'mode':<8}{'ratio':>6}{'eff':>6}{'KL [95% CI]':>24}{'top1 [95% CI]':>24}{'F1':>8}"]
    table = []
    for (meth, mode, ratio), v in sorted(agg.items(), key=lambda x: (x[0][0] != "full", x[0][0], x[0][1], x[0][2])):
        mean = {m: sum(x) / len(x) for m, x in v.items()}
        ci = {m: bootstrap_ci(v[m]) for m in ["kl", "top1"]}
        table.append(dict(method=meth, mode=mode, ratio=ratio, **mean, ci=ci))
        fmt = lambda m, p: f"{mean[m]:.{p}f} [{ci[m][0]:.{p}f},{ci[m][1]:.{p}f}]"
        lines.append(f"{meth:<10}{mode:<8}{ratio:>6.2f}{mean['eff']:>6.2f}{fmt('kl', 3):>24}{fmt('top1', 3):>24}{mean['f1']:>8.3f}")
    l0 = [r["l0_max_abs"] for r in rows if "l0_max_abs" in r]
    lines.append(f"\nlayer-0 K max |diff| vs full (RoPE shift check): max over examples = {max(l0):.2e}")
    l0r = [r["l0_rel"] for r in rows if "l0_rel" in r]
    if l0r:
        lines.append(f"layer-0 K relative error vs full: max over examples = {max(l0r):.2e}")
    text = "\n".join(lines)
    print(text)
    (out / "summary.txt").write_text(text)
    (out / "summary.json").write_text(json.dumps(table, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for pol in POLICIES:
        pts = sorted((t["ratio"], t) for t in table if t["method"] == pol and t["ratio"] < 1)
        xs = [0.0] + [p[0] for p in pts]
        naive = next(t for t in table if t["method"] == "naive" and t["mode"] == "shared")
        for ax, m in zip(axes[:2], ["kl", "top1"]):
            ax.plot(xs, [naive[m]] + [p[1][m] for p in pts], marker="o", label=pol)
    full = next(t for t in table if t["method"] == "full")
    axes[1].axhline(1.0, color="k", ls="--", lw=0.8)
    for ax, m, yl in zip(axes[:2], ["kl", "top1"], ["KL(full || reuse)", "top-1 agreement w/ full"]):
        ax.set_xlabel("recompute ratio"); ax.set_ylabel(yl); ax.legend(); ax.grid(alpha=0.3)
    errs = json.loads((out / "layer_err.json").read_text())
    for mode, e in errs.items():
        axes[2].plot(e, label=f"naive ({mode})")
    axes[2].set_xlabel("layer"); axes[2].set_ylabel("rel. K error on doc tokens"); axes[2].legend(); axes[2].grid(alpha=0.3)
    fig.suptitle(f"Phase 0 — full-prefill F1 = {full['f1']:.3f}")
    fig.tight_layout()
    fig.savefig(out / "phase0.png", dpi=120)


if __name__ == "__main__":
    main()
