# Pre-registration

**Registered:** Phase 0, before any training run, any evaluation, and any result was seen.
**Registered at commit:** the commit that introduces this file. Any later edit must be an
additional dated section below, never a rewrite of what is above — an amendment history is
what makes a pre-registration worth anything.

---

## 1. Background

GraLoRA splits a LoRA adapter's `A` and `B` matrices into `k × k` sub-blocks, giving each
sub-block its own low-rank map plus cross-block information exchange. The block count `k`
is a single global hyperparameter: every adapted layer in the model uses the same one.

The parameter count is exactly invariant to `k`. `gralora_A` has shape
`[k, in_features/k, r]` and `gralora_B` has shape `[k, r, out_features/k]`, so their element
counts are `in_features · r` and `out_features · r` for any legal `k`. Only the *structure*
of the adapter changes — its size does not. This is verified in
`tests/test_patching.py::test_trainable_parameter_count_is_invariant_to_uniform_k`.

## 2. Hypothesis

> **H1.** Different layers of a transformer benefit from different block granularities, and a
> per-layer allocation of `k` chosen from a cheap gradient signal outperforms the best uniform
> `k` at equal parameter count.

The parameter-count invariance is what makes this testable cleanly: every allocation strategy
is budget-matched *by construction*. There is no capacity confound to control for, and no
balancing logic anywhere in the pipeline. Any difference in accuracy is attributable to the
allocation itself.

**H0 (what H1 is tested against).** Accuracy is insensitive to how `k` is distributed across
layers. Any observed gap between allocators is seed noise.

## 3. Method under test

`gradnorm`: attach GraLoRA, run N = 8 calibration batches from the training set, collect
per-layer gradient norms, RMS-normalise them by layer width (so the ranking reflects
per-parameter gradient magnitude rather than which layer is simply widest), and assign finer
`k` to higher-ranked layers, subject to the legality constraints `r % k == 0`,
`in_features % k == 0`, `out_features % k == 0`.

## 4. Conditions

| Factor | Levels |
|---|---|
| Allocator | `uniform`, `gradnorm`, `inverse`, `random` |
| Rank `r` | 16, 32, 64, 128 |
| Seed | 0, 1, 2 |

Plus a plain-LoRA reference run at each rank.

**Controls, and what each one rules out:**

- **`random`** — assigns legal `k` values at random, budget-matched and constraint-respecting.
  This is the most important control in the project. If `random` matches `gradnorm`, then the
  gradient signal contributes nothing and any gain over `uniform` comes from heterogeneity
  alone. That is a finding, not a failure, and it will be reported as the headline result if
  it happens.
- **`inverse`** — `gradnorm`'s ranking reversed. If `inverse` also beats `uniform`, the
  direction of the heuristic is not what matters.
- **plain LoRA** — without it there is no way to tell whether GraLoRA of *any* kind is helping
  over the more familiar baseline.

## 5. Decision rule — fixed in advance

Primary metric: mean commonsense-benchmark accuracy across 3 seeds, per (allocator, rank).

> **`gradnorm` is declared to beat `uniform` at a given rank only if the gap between their
> means exceeds one pooled standard deviation across the 3 seeds at that rank.**

Pooled std is `sqrt((s²_gradnorm + s²_uniform) / 2)`, computed over the 3 seeds.

Three seeds cannot support a meaningful significance test, and none will be claimed. The
one-pooled-std rule is a deliberately blunt threshold chosen *before* seeing data precisely so
it cannot be tuned afterwards.

**Binding consequences:**

- A gap within one pooled std is reported as **"no detectable difference"** — never as an
  improvement, a trend, or "promising". This applies whichever direction it points.
- The rule applies identically to `random` vs `uniform` and to `inverse` vs `uniform`.
- If `gradnorm` beats `uniform` at some ranks and not others, every rank is reported. No
  rank is dropped, and the best one is not promoted to the abstract.
- The rank grid includes **128** deliberately: GraLoRA's own claim is that its edge over LoRA
  appears at higher ranks. Dropping 128 because it is slow would remove the regime where the
  effect is most likely to exist.

## 6. What a negative result means

A negative result — `gradnorm` fails to clear one pooled std, or `random` matches it — is a
publishable outcome of this project, not a reason to keep searching for a configuration that
works.

Specifically it would mean: *within this budget-matched setting, on this model family and
task, the layer-wise distribution of GraLoRA's block granularity does not measurably affect
downstream accuracy, and a gradient-norm heuristic does not identify a better one.* That is
useful, because the parameter-count invariance makes per-layer `k` allocation an obvious
thing to try, and a documented null saves the next person the same 12 GPU-hours.

**Pre-committed:** the negative result gets written up with the same figures, the same tables,
and the same prominence as a positive one. The k-map heatmap is reported either way — whether
that map turns out structured or arbitrary is interesting in both directions.

## 7. Prohibited after unblinding

- Adding allocator strategies, ranks, or seeds *after* seeing results and reporting the best.
- Switching the primary metric, or reporting a per-sub-task accuracy in place of the mean
  because it looks better.
- Dropping a seed as an outlier.
- Relaxing the decision rule in §5.
- Reporting an unpowered comparison as significant.

If the eval pipeline is found to be broken (for example, accuracy is flat across all ranks,
which points at the prompt template rather than the method), it is fixed and **every**
condition is re-run from scratch. Partial re-runs mixing pre-fix and post-fix numbers are not
reported.

## 8. Amendments

*None. Any amendment must be appended here with its date and its reason, before the results it
affects are examined.*
