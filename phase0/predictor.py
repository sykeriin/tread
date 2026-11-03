"""Does cheap layer-1 KV deviation predict which requests are hard, and does an adaptive budget beat a fixed one?

Reuses a finished sweep's rows.json (per-example KL at every deviation ratio) and recomputes only the
per-example deviation features, which cost a layer-0 pass over the doc tokens.

Adaptive policy: recompute at ratio `lo` if feature < threshold, else `hi`. Its parameters are picked
on one half of the examples and scored on the other (2-fold, both ways), so the gain is out-of-sample.

Usage: python phase0/predictor.py --model Qwen/Qwen2.5-1.5B-Instruct --dtype float16 --rows phase0/results_1.5b
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch.nn.functional as F

from transformers.models.qwen2.modeling_qwen2 import repeat_kv

from blend import _qkv, assemble, blend, layer1_deviation, to_cache
from run_phase0 import build_example

RATIOS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
BUDGETS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]


def spearman(a, b):
    rank = lambda x: torch.tensor(x, dtype=torch.float64).argsort().argsort().double()
    ra, rb = rank(a), rank(b)
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


@torch.inference_mode()
def features(model, tok, data, dev, seed=0):
    data_rng = torch.Generator().manual_seed(seed)  # same doc order as run_phase0 --start <seed>
    feats = []
    for ex in data:
        prefix, docs, query = build_example(ex, tok, data_rng)
        prefix, docs = prefix.to(dev), [d.to(dev) for d in docs]
        ctx = torch.cat([prefix] + docs)
        K, V = assemble(model, prefix, docs, "shared")
        all_c = torch.arange(len(prefix), len(ctx), device=dev)
        d = layer1_deviation(model, ctx, K, V, all_c)[0].float()
        scale = (K[1][:, :, all_c].norm(dim=-1).sum(1) + V[1][:, :, all_c].norm(dim=-1).sum(1))[0].float()
        rel = d / scale
        k = max(1, len(d) // 10)
        feats.append(dict(dev_mean=d.mean().item(), dev_top10=d.topk(k).values.mean().item(),
                          rel_mean=rel.mean().item(), rel_top10=rel.topk(k).values.mean().item(),
                          ctx_len=float(len(ctx))))
    return feats


@torch.inference_mode()
def query_features(model, tok, data, dev, probe_ratio=0.05, seed=0):
    """Query-conditioned hardness signals, all computed from the reused (naive) cache.

    - ans_entropy / ans_margin: uncertainty of the first answer token under the naive cache.
    - xdoc_attn: share of the last query token's doc attention that lands on docs 2..N
      (docs 2..N are the ones that never saw earlier docs, i.e. where cross-doc info is missing).
    - probe_kl / probe_flip: how much a cheap boundary recompute of `probe_ratio` moves the
      first answer token's distribution. Costs `probe_ratio` recompute, charged in the eval.
    """
    data_rng = torch.Generator().manual_seed(seed)  # same doc order as run_phase0 --start <seed>
    feats = []
    for ex in data:
        prefix, docs, query = build_example(ex, tok, data_rng)
        prefix, docs, query = prefix.to(dev), [d.to(dev) for d in docs], query.to(dev)
        ctx = torch.cat([prefix] + docs)
        K, V = assemble(model, prefix, docs, "shared")

        # output_attentions would force HF's eager attention, which overflows to NaN in fp16 on Qwen.
        # Keep SDPA for the logits and rebuild the last query token's attention in fp32 instead.
        out = model(query[None], past_key_values=to_cache(K, V), output_hidden_states=True)
        logp = F.log_softmax(out.logits[0, -1].float(), -1)
        p = logp.exp()
        top2 = p.topk(2).values
        pos = torch.tensor([len(ctx) + len(query) - 1], device=dev)
        doc = slice(len(prefix), len(ctx))
        first_end = len(prefix) + len(docs[0]) - len(prefix)
        shares = []
        for l, layer in enumerate(model.model.layers):
            q = _qkv(layer, model, layer.input_layernorm(out.hidden_states[l][:, -1:]), pos)[0]
            groups = q.shape[1] // K[l].shape[1]
            Kd = repeat_kv(K[l][:, :, doc], groups)
            # Share of doc attention on docs 2..N: a ratio over doc keys only, so softmax over
            # the doc subset gives the same value as the full softmax.
            w = (q.float() @ Kd.float().transpose(-1, -2)).squeeze(2).softmax(-1)
            shares.append(w[..., first_end:].sum(-1).mean())
        xdoc = torch.stack(shares).mean()

        cands, off = [], len(prefix)
        for d in docs:
            cands.append(torch.arange(off, off + len(d), device=dev))
            off += len(d)
        Kb, Vb, _, _ = blend(model, ctx, K, V, cands, probe_ratio, "boundary")
        logq = F.log_softmax(model(query[None], past_key_values=to_cache(Kb, Vb)).logits[0, -1].float(), -1)

        row = dict(ans_entropy=-(p * logp).sum().item(), ans_margin=-(top2[0] - top2[1]).item(),
                   xdoc_attn=xdoc.item(),
                   probe_kl=F.kl_div(logq, logp, log_target=True, reduction="sum").item(),
                   probe_flip=float(logp.argmax() != logq.argmax()))
        assert all(math.isfinite(v) for v in row.values()), row
        feats.append(row)
    return feats


def evaluate(policy_kl, policy_eff, idx):
    return policy_kl[idx].mean().item(), policy_eff[idx].mean().item()


def adaptive_curve(f, kl, eff, n_folds=2, seed=0, extra=0.0):
    """Out-of-sample mean KL of the best two-level policy under each average-eff budget.

    `extra` is a per-request cost paid to compute the feature itself (e.g. a probe recompute).
    """
    n = len(f)
    eff = eff + extra
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    folds = [perm[i::n_folds] for i in range(n_folds)]
    # Enumerate policies: (lo, hi, threshold quantile) -> per-example ratio index.
    pols = []
    for lo, hi in itertools.combinations(range(len(RATIOS)), 2):
        for q in [0.5, 0.6, 0.7, 0.8, 0.9]:
            pols.append((lo, hi, q))
    out = {}
    for B in BUDGETS:
        kls, effs = [], []
        for k in range(n_folds):
            test = folds[k]
            train = torch.cat([folds[j] for j in range(n_folds) if j != k])
            best = None
            for lo, hi, q in pols:
                thr = f[train].quantile(q)
                choice = torch.where(f >= thr, hi, lo)
                pk, pe = kl[torch.arange(n), choice], eff[torch.arange(n), choice]
                tk, te = evaluate(pk, pe, train)
                if te <= B and (best is None or tk < best[0]):
                    best = (tk, lo, hi, thr)
            if best is None:
                continue
            _, lo, hi, thr = best
            choice = torch.where(f >= thr, hi, lo)
            pk, pe = kl[torch.arange(n), choice], eff[torch.arange(n), choice]
            tk, te = evaluate(pk, pe, test)
            kls.append(tk), effs.append(te)
        out[B] = (sum(kls) / len(kls), sum(effs) / len(effs))
    return out


def fixed_curve(kl, eff):
    """Best fixed deviation ratio whose mean eff fits each budget (no training needed)."""
    out = {}
    for B in BUDGETS:
        ok = [j for j in range(len(RATIOS)) if eff[:, j].mean() <= B]
        j = max(ok, key=lambda j: eff[:, j].mean())
        out[B] = (kl[:, j].mean().item(), eff[:, j].mean().item())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--rows", default="phase0/results_1.5b")
    args = ap.parse_args()
    out = Path(args.rows)
    rows = json.loads((out / "rows.json").read_text())
    n = 1 + max(r["ex"] for r in rows)

    kl = torch.zeros(n, len(RATIOS), dtype=torch.float64)
    eff = torch.zeros_like(kl)
    for r in rows:
        if r["method"] == "naive" and r["mode"] == "shared":
            kl[r["ex"], 0] = r["kl"]
        elif r["method"] == "deviation" and r["ratio"] in RATIOS:
            j = RATIOS.index(r["ratio"])
            kl[r["ex"], j], eff[r["ex"], j] = r["kl"], r["eff"]

    loaded = {}

    def load():
        if not loaded:
            loaded["tok"] = AutoTokenizer.from_pretrained(args.model)
            loaded["model"] = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=getattr(torch, args.dtype)).to("cuda").eval()
            loaded["data"] = load_dataset("hotpotqa/hotpot_qa", "distractor", split=f"validation[:{n}]")
        return loaded["model"], loaded["tok"], loaded["data"], "cuda"

    def cached(name, fn):
        path = out / name
        if path.exists():
            return json.loads(path.read_text())
        feats = fn(*load())
        path.write_text(json.dumps(feats, indent=1))
        return feats

    feats = cached("features.json", features)
    qfeats = cached("features_query.json", query_features)
    feats = [{**a, **b} for a, b in zip(feats, qfeats)]

    lines = ["Spearman correlation of per-example feature with KL (n=%d)" % n,
             f"{'feature':<12}{'naive KL':>10}{'dev@0.2 KL':>12}"]
    j20 = RATIOS.index(0.2)
    for name in feats[0]:
        f = [x[name] for x in feats]
        lines.append(f"{name:<12}{spearman(f, kl[:, 0].tolist()):>10.3f}{spearman(f, kl[:, j20].tolist()):>12.3f}")

    fixed = fixed_curve(kl, eff)
    curves = {"fixed": fixed}
    # Probe features pay for their boundary recompute on every request.
    extra = {"probe_kl": 0.05, "probe_flip": 0.05}
    # One 50/50 split is noisy at n=100: average over many random splits and report the spread.
    def repeated(f, extra_cost=0.0, seeds=20):
        runs = [adaptive_curve(f, kl, eff, seed=s, extra=extra_cost) for s in range(seeds)]
        return {B: (torch.tensor([r[B][0] for r in runs]), torch.tensor([r[B][1] for r in runs])) for B in BUDGETS}

    curves = {"fixed": {B: (torch.tensor([v[0]]), torch.tensor([v[1]])) for B, v in fixed.items()}}
    for name in ["rel_top10", "ans_entropy", "ans_margin", "xdoc_attn", "probe_kl"]:
        f = torch.tensor([x[name] for x in feats], dtype=torch.float64)
        curves[name] = repeated(f, extra.get(name, 0.0))
    # Not a true oracle: it ranks requests by how much naive reuse loses, not by how much recompute helps.
    curves["naive-KL rank"] = repeated(kl[:, 0].clone())

    lines.append("\nOut-of-sample mean KL ± sd over 20 random 2-fold splits (mean eff), per average budget")
    lines.append(f"{'budget':<8}" + "".join(f"{k:>24}" for k in curves))
    for B in BUDGETS:
        cell = lambda k: f"{curves[k][B][0].mean():.3f}±{curves[k][B][0].std() if len(curves[k][B][0]) > 1 else 0:.3f} ({curves[k][B][1].mean():.2f})"
        lines.append(f"{B:<8.2f}" + "".join(f"{cell(k):>24}" for k in curves))

    # True per-example oracle: for each budget, choose each example's ratio to minimise total KL
    # (exhaustive Lagrangian sweep), giving a real upper bound on any adaptive policy.
    lines.append("\nPer-example oracle (upper bound on adaptive policies)")
    for B in BUDGETS:
        best = None
        for lam in torch.logspace(-3, 1, 200):
            choice = (kl + lam * eff).argmin(1)
            e = eff[torch.arange(n), choice].mean()
            k = kl[torch.arange(n), choice].mean()
            if e <= B and (best is None or k < best[0]):
                best = (k.item(), e.item())
        lines.append(f"  budget {B:.2f}: KL {best[0]:.3f} (eff {best[1]:.2f})")
    text = "\n".join(lines)
    print(text)
    (out / "predictor.txt").write_text(text)


if __name__ == "__main__":
    main()
