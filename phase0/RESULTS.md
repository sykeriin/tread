# Phase 0 results — Qwen2.5-0.5B-Instruct, fp32, RTX 3050 4GB

30 HotpotQA-distractor examples (2 gold + 2 distractor docs, shuffled), ~380–630 context tokens.
Reference = full prefill. KL(full ‖ reuse) and top-1 agreement are teacher-forced over the
query end + a 32-token greedy continuation. Raw data: `results/rows.json`, table with 95%
bootstrap CIs: `results/summary.txt`, plot: `results/phase0.png`.

Reproduce: `.venv/Scripts/python phase0/run_phase0.py --n 30`

## Correctness checks (pass)
- RoPE re-rotation: layer-0 K matches full prefill to max |diff| 5e-4 (fp32 noise).
- Recomputing 100% of doc tokens reproduces full prefill (KL 0.000, top-1 1.000).

## Main numbers
`eff` = recompute cost in token-layers. Deviation pays for a layer-0 pass over every doc token
to pick its tokens (+~4% on 24 layers), so its nominal ratio understates cost.

| config | ratio | eff | KL [95% CI] | top-1 [95% CI] |
|---|---|---|---|---|
| naive, docs prefilled alone | 0 | 0 | 0.528 [0.319, 0.785] | 0.780 [0.688, 0.863] |
| naive, docs after shared prefix | 0 | 0 | 0.243 [0.086, 0.480] | 0.881 [0.810, 0.946] |
| deviation | 0.20 | 0.23 | 0.094 [0.038, 0.174] | 0.925 [0.872, 0.969] |
| deviation | 0.30 | 0.33 | 0.069 [0.021, 0.143] | 0.939 [0.887, 0.979] |
| deviation | 0.50 | 0.52 | 0.017 [0.004, 0.037] | 0.962 [0.917, 0.995] |
| boundary | 0.50 | 0.50 | 0.036 [0.018, 0.061] | 0.942 [0.896, 0.980] |
| random | 0.50 | 0.50 | 0.118 [0.044, 0.214] | 0.906 [0.838, 0.968] |

## Paired comparisons (KL difference, same examples, 95% bootstrap CI)
| comparison | Δ KL | significant? |
|---|---|---|
| alone − shared prefix (naive) | +0.286 [+0.148, +0.453] | yes |
| deviation − random @ 0.2 | −0.124 [−0.279, −0.008] | yes, barely |
| deviation − boundary @ 0.2 | −0.124 [−0.250, −0.021] | yes |
| deviation − boundary @ 0.3 | −0.024 [−0.069, +0.021] | no |
| deviation − random @ 0.5 | −0.101 [−0.200, −0.029] | yes |
| deviation − boundary @ 0.5 | −0.019 [−0.030, −0.009] | yes, small |
| deviation @ 0.3 − naive | −0.174 [−0.419, +0.010] | no (heavy-tailed examples) |

## What the data supports
1. **Prefilling each chunk after the shared system prefix clearly helps** (sink handling).
   Keep it as the default.
2. **Deviation-guided selection is the best policy where it is distinguishable**, with significant
   wins at 20% and 50%. At 30%, boundary is statistically tied.
3. **Error is heavy-tailed.** A few examples dominate the KL mean, so the CIs are wide. n=30 is
   too small for fine-grained policy claims. Phase 1 should use n ≥ 200 and report per-example
   distributions, not just means.
4. **Go threshold not met on this model.** Top-1 is ~0.93 at ~23% effective recompute, versus the
   ~0.98 target.
5. F1 is too noisy at n=30 to rank anything. It is not used for conclusions.

Note: these numbers differ from the first run because document order is now drawn from its own
seed (independent of the random policy), so the examples saw different shuffles.

## Verdict (0.5B): conditional GO
The mechanism works and the harness is trustworthy. Next step was to rerun on 1.5B with larger n.

---

# Phase 0b — Qwen2.5-1.5B-Instruct, fp16, n=100

Reproduce: `.venv/Scripts/python phase0/run_phase0.py --model Qwen/Qwen2.5-1.5B-Instruct --dtype float16 --n 100 --out phase0/results_1.5b`
(checkpoints per example; rerun the same command to resume).

Correctness: 100% recompute → KL 0.000. Layer-0 K *relative* error 4.3e-4. The absolute max diff
of 0.5 is fp16 rounding on keys with magnitudes in the hundreds, not a RoPE error.

| config | eff | KL [95% CI] | top-1 [95% CI] | F1 |
|---|---|---|---|---|
| full prefill | — | 0 | 1.000 | 0.582 |
| naive, alone | 0 | 0.379 [0.293, 0.480] | 0.871 [0.829, 0.909] | 0.479 |
| naive, shared prefix | 0 | 0.291 [0.198, 0.404] | 0.876 [0.828, 0.916] | 0.540 |
| deviation 0.20 | 0.23 | 0.182 [0.102, 0.277] | 0.902 [0.856, 0.943] | 0.553 |
| deviation 0.30 | 0.32 | 0.131 [0.069, 0.213] | 0.931 [0.894, 0.963] | 0.535 |
| deviation 0.50 | 0.52 | 0.091 [0.036, 0.172] | 0.968 [0.947, 0.986] | 0.554 |
| boundary 0.30 | 0.30 | 0.189 [0.113, 0.280] | 0.927 [0.893, 0.957] | 0.524 |
| random 0.30 | 0.30 | 0.255 [0.163, 0.371] | 0.886 [0.840, 0.928] | 0.531 |

Paired ΔKL (95% CI): deviation beats random at 0.1, 0.2 and 0.3 (e.g. −0.124 [−0.212, −0.038] @0.3).
It beats boundary only at 0.1 (−0.061 [−0.115, −0.011]) and ties it at 0.2/0.3/0.5. It beats naive at
every ratio. Shared-prefix prefill beats alone: −0.087 [−0.156, −0.018].

## What changed vs 0.5B
1. **The "small model is pessimistic" hypothesis is not supported.** 1.5B is *not* more robust:
   naive KL is similar (0.29 vs 0.24), and blending recovers less (deviation@0.5 KL 0.091 vs 0.017).
2. **Error is very heavy-tailed.** The worst 10% of examples carry 55% of naive KL. The median
   KL at deviation@0.2 is 0.027, small enough to be near-lossless for most requests.
3. **Deviation vs boundary is still unresolved.** Deviation is clearly better than random, but only
   marginally better than the much cheaper boundary window.
4. **Task F1 moves the right way.** Naive-shared loses about 0.04 F1 vs full. Deviation@0.2 is within 0.03.

## Verdict: GO, with a reframed goal
A fixed recompute ratio cannot hit "near-full quality" on average, because a minority of requests
are hard. The project's contribution should be **adaptive recompute**: detect the hard requests
cheaply and spend the budget there. The layer-1 deviation signal is already computed and is a natural
predictor. Next experiment: check whether per-example layer-1 deviation predicts per-example
naive KL. If it does, a threshold policy (recompute more only when deviation is high) should beat
every fixed ratio on the quality/compute Pareto curve.

---

# Phase 0c — Does layer-1 deviation predict hard requests? (1.5B, n=100)

Script: `phase0/predictor.py`, output: `results_1.5b/predictor.txt`. The adaptive policy picks
recompute ratio `lo`/`hi` by thresholding a per-example feature. Its parameters are chosen on half the
examples and scored on the other half (2-fold), so the numbers below are out-of-sample.

**Answer: no.** Per-example Spearman correlation with naive KL:

| feature | ρ vs naive KL |
|---|---|
| mean layer-1 deviation | 0.02 |
| top-10% layer-1 deviation (raw / relative) | 0.08 / 0.08 |
| context length | −0.10 |

Out-of-sample mean KL at a given average recompute budget:

| budget | fixed ratio | adaptive (rel_top10) | oracle (true naive KL) |
|---|---|---|---|
| 0.15 | 0.222 | 0.202 | 0.160 |
| 0.20 | 0.182 | 0.172 | 0.158 |
| 0.25 | 0.182 | 0.152 | 0.129 |
| 0.35 | 0.131 | 0.137 | 0.123 |

The adaptive gains are small and inconsistent, which is what you'd expect from ρ≈0.08. Some budgets slightly
overshoot out-of-sample.

## What this means
1. **The headroom is real.** An oracle that knew which requests are hard reaches deviation@0.35-level
   quality (KL ≈ 0.13) at about 20% compute, and beats fixed recompute by 20–30% KL at equal budget.
2. **Layer-1 deviation is the wrong signal for *request* hardness.** It ranks tokens *within* a request
   well (deviation beats random), but it is query-independent: it measures how much a doc's
   KV changed, not whether the *question* needs that cross-doc information.
3. **Next predictor must be query-conditioned.** These are cheap to compute from the naive cache:
   - entropy / top-1 margin of the first answer token under the naive cache;
   - query→doc attention mass on non-first documents (cross-doc dependence);
   - disagreement between naive and a tiny (e.g. 5%) recompute on the first answer token.
   The third is the most direct and costs only the 5% recompute plus a query-only forward.

---

# Phase 0d — Query-conditioned predictors (1.5B, n=100)

Same harness (`phase0/predictor.py`, output in `results_1.5b/predictor.txt`). Each policy is now evaluated
over 20 random 2-fold splits (mean ± sd). Probe features are charged their 5% recompute.

Bug fixed on the way: `output_attentions=True` forces HF eager attention, which overflows to NaN
in fp16 on Qwen, so the first run's query features were all NaN. Attention shares are now rebuilt
in fp32 from hidden states, and the script asserts every feature is finite.

| feature | ρ vs naive KL | ρ vs KL after 20% recompute |
|---|---|---|
| layer-1 deviation (rel_top10) | 0.08 | 0.00 |
| first-answer-token entropy | **0.36** | **0.43** |
| first-answer-token margin | 0.36 | 0.39 |
| query→docs 2..N attention share (xdoc_attn) | 0.13 | 0.22 |
| 5% probe KL | 0.24 | 0.24 |

Out-of-sample mean KL at equal average recompute (fixed = linear mix of neighbouring fixed ratios):

| budget | fixed (interp.) | entropy | **xdoc_attn** | per-example oracle |
|---|---|---|---|---|
| 0.15 | ~0.21 | 0.246 ± 0.013 | **0.190 ± 0.018** | 0.060 |
| 0.25 | ~0.176 | 0.165 ± 0.015 | **0.140 ± 0.025** | 0.047 |
| 0.30 | ~0.154 | 0.170 ± 0.015 | **0.107 ± 0.019** | 0.044 |
| 0.35 | ~0.127 | 0.141 ± 0.013 | **0.103 ± 0.017** | 0.043 |

## Findings
1. **Cross-doc attention share is the useful signal.** At 25–35% budget it cuts KL by about 20–30% vs fixed
   recompute, and it costs nothing beyond the query forward we run anyway.
2. **Correlation is not usefulness.** Answer entropy correlates best with hardness (ρ 0.36) but does
   *not* beat fixed recompute. It also correlates with KL *after* recompute (0.43), so it flags requests
   that are intrinsically uncertain, not ones recompute can fix. xdoc_attn flags requests whose answer
   depends on cross-document information, which is exactly the error recompute repairs.
3. **There is a lot of headroom left.** A per-example oracle reaches KL ≈ 0.044 at 30% budget
   versus 0.107 for xdoc_attn. A better predictor, or combining features, is the research contribution.
4. The earlier "oracle" (rank by naive KL) was not an upper bound; it has been renamed `naive-KL rank`.

## Caveats
- One model, one dataset, n=100: policy parameters are chosen on 50 examples per fold.
  Confirm on a fresh set of examples (e.g. HotpotQA validation 100–400) before claiming it.
- Two-level policies only. A continuous mapping from feature to ratio may do better.

## Verdict: superseded by Phase 0e
This section's adaptive-policy gain did **not** replicate on held-out data (see below).

---

# Phase 0e — Held-out check (fit on val 0–99, test on val 100–399, 1.5B fp16)

Scripts: `run_phase0.py --start 100 --n 300 --policies deviation --modes shared --no-answers`, then
`phase0/holdout.py` (run from the repo root). Output: `results_1.5b_holdout/holdout.txt`. Policy ratios and
the absolute feature threshold are fit on the original 100 and applied unchanged to the fresh 300.
The baseline is fixed deviation recompute at the same test cost. Δ is the paired per-example difference.

| feature | budget | adaptive KL | fixed KL | Δ KL [95% CI] |
|---|---|---|---|---|
| xdoc_attn | 0.25 | 0.089 | 0.092 | −0.003 [−0.016, +0.009] |
| xdoc_attn | 0.30 | 0.074 | 0.075 | −0.001 [−0.019, +0.017] |
| xdoc_attn | 0.35 | 0.067 | 0.064 | +0.002 [−0.016, +0.020] |
| ans_entropy | 0.20 | 0.103 | 0.112 | −0.009 [−0.019, −0.001] |
| rel_top10 | 0.20–0.25 | 0.15 / 0.14 | 0.12 / 0.10 | **worse**, +0.036 (CI excludes 0) |

Spearman ρ with naive KL on the test set: answer entropy 0.47, probe KL 0.44, answer margin 0.42,
xdoc_attn 0.11, layer-1 deviation −0.05.

## Findings
1. **The cross-doc attention result was overfitting.** On fresh examples it is indistinguishable from
   fixed recompute at every budget. The Phase 0d gain came from choosing policy parameters on 50
   examples of a heavy-tailed distribution. Repeated splits of the *same* 100 examples did not
   protect against that.
2. **Answer entropy is a robust *hardness* signal (ρ 0.36 → 0.47 on fresh data), but not a useful
   *allocation* signal.** Its best held-out gain is −8% at one budget, with a CI touching zero.
   Hard-by-entropy requests are not the ones recompute fixes.
3. **Fixed deviation-guided recompute is solid on the larger set:** KL 0.263 (naive) → 0.099 @20%
   → 0.070 @30% → 0.029 @50%, with 100% recompute exact. The first 100 examples were a harder,
   heavier-tailed sample (deviation@30% KL 0.131 there).
4. Error stays heavy-tailed (worst 10% = 50% of naive KL). The per-example oracle headroom is real,
   but no cheap signal tested here captures it.

## Verdict: GO on the engine, NO-GO on adaptive allocation (for now)
- The project's core is validated: exact RoPE re-rotation, shared-prefix chunk prefill, and
  deviation-guided selective recompute give a clean, monotone quality/compute curve.
- The contribution for Phase 1 is a **working non-prefix reuse engine with an honest Pareto curve**
  and TTFT measurements, not adaptive allocation.
- Adaptive allocation stays an open research question for the writeup, including this negative
  result and the methodology lesson (in-sample CV on n=100 is not enough).
