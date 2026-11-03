"""Held-out check: fit adaptive-recompute policies on one sweep, score them once on a fresh one.

Policy (as in predictor.py): recompute at ratio `lo` if feature < threshold, else `hi`. The ratios and
the *absolute* threshold are chosen on the train sweep only, then applied unchanged to the test sweep.
Baseline: fixed recompute at the same mean cost, interpolated by mixing the two neighbouring fixed
ratios. Reported with a paired bootstrap CI on the per-example KL difference.

Usage:
  python phase0/holdout.py --train phase0/results_1.5b --test phase0/results_1.5b_holdout --test-start 100
"""
import argparse
import itertools
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from predictor import BUDGETS, RATIOS, features, query_features

FEATURES = ["xdoc_attn", "ans_entropy", "rel_top10"]


def load_matrix(rows):
    """Per-example KL / eff at each deviation ratio, ordered by example id."""
    exs = sorted({r["ex"] for r in rows})
    idx = {e: i for i, e in enumerate(exs)}
    kl = torch.zeros(len(exs), len(RATIOS), dtype=torch.float64)
    eff = torch.zeros_like(kl)
    for r in rows:
        if r["method"] == "naive" and r["mode"] == "shared":
            kl[idx[r["ex"]], 0] = r["kl"]
        elif r["method"] == "deviation" and r["ratio"] in RATIOS:
            j = RATIOS.index(r["ratio"])
            kl[idx[r["ex"]], j], eff[idx[r["ex"]], j] = r["kl"], r["eff"]
    return kl, eff


def apply(f, lo, hi, thr, kl, eff):
    n = torch.arange(len(f))
    choice = torch.where(f >= thr, hi, lo)
    return kl[n, choice], eff[n, choice]


def fit(f, kl, eff, budget):
    """Best (lo, hi, threshold) on train under a mean-eff budget."""
    best = None
    for lo, hi in itertools.combinations(range(len(RATIOS)), 2):
        for q in torch.linspace(0.3, 0.95, 14, dtype=f.dtype):
            thr = f.quantile(q).item()
            k, e = apply(f, lo, hi, thr, kl, eff)
            if e.mean() <= budget and (best is None or k.mean() < best[0]):
                best = (k.mean().item(), lo, hi, thr)
    return best[1:]


def fixed_at(kl, eff, target):
    """Per-example KL of fixed recompute mixed between neighbouring ratios to hit mean eff `target`."""
    me = eff.mean(0)
    j = int((me <= target).nonzero().max())
    if j == len(RATIOS) - 1:
        return kl[:, j]
    w = ((target - me[j]) / (me[j + 1] - me[j])).clamp(0, 1)
    return (1 - w) * kl[:, j] + w * kl[:, j + 1]


def paired_ci(d, n=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = d[torch.randint(len(d), (n, len(d)), generator=g)].mean(1)
    return d.mean().item(), m.quantile(0.025).item(), m.quantile(0.975).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--train", default="phase0/results_1.5b")
    ap.add_argument("--test", default="phase0/results_1.5b_holdout")
    ap.add_argument("--test-start", type=int, default=100)
    args = ap.parse_args()
    train, test = Path(args.train), Path(args.test)

    kl_tr, eff_tr = load_matrix(json.loads((train / "rows.json").read_text()))
    kl_te, eff_te = load_matrix(json.loads((test / "rows.json").read_text()))
    n_te = len(kl_te)

    ftr = [{**a, **b} for a, b in zip(json.loads((train / "features.json").read_text()),
                                      json.loads((train / "features_query.json").read_text()))]
    fpath = test / "features_all.json"
    if fpath.exists():
        fte = json.loads(fpath.read_text())
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).to("cuda").eval()
        data = load_dataset("hotpotqa/hotpot_qa", "distractor",
                            split=f"validation[{args.test_start}:{args.test_start + n_te}]")
        a = features(model, tok, data, "cuda", seed=args.test_start)
        b = query_features(model, tok, data, "cuda", seed=args.test_start)
        fte = [{**x, **y} for x, y in zip(a, b)]
        fpath.write_text(json.dumps(fte, indent=1))

    lines = [f"Held-out: fit on {len(kl_tr)} examples ({train}), test on {n_te} ({test})",
             f"{'feature':<12}{'budget':>7}{'test eff':>10}{'adaptive KL':>13}{'fixed KL':>10}"
             f"{'Δ KL [95% CI]':>28}{'Δ %':>7}"]
    for name in FEATURES:
        f_tr = torch.tensor([x[name] for x in ftr], dtype=torch.float64)
        f_te = torch.tensor([x[name] for x in fte], dtype=torch.float64)
        for B in BUDGETS:
            lo, hi, thr = fit(f_tr, kl_tr, eff_tr, B)
            k, e = apply(f_te, lo, hi, thr, kl_te, eff_te)
            fk = fixed_at(kl_te, eff_te, e.mean())
            d, a, b = paired_ci(k - fk)
            lines.append(f"{name:<12}{B:>7.2f}{e.mean():>10.3f}{k.mean():>13.3f}{fk.mean():>10.3f}"
                         f"{f'{d:+.3f} [{a:+.3f}, {b:+.3f}]':>28}{100 * d / fk.mean():>+7.0f}")
    text = "\n".join(lines)
    print(text)
    (test / "holdout.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
