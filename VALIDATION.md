# XGES — validation and self-review

This documents what was implemented for **XGES (Extremely Greedy Equivalence
Search)**, what is verified, and what is explicitly deferred. It is written to
the honesty bar the brief asked for: a search that "returns a plausible graph"
is *not* evidence of a correct Markov equivalence class, so every operator and
the greedy engine were checked against a brute-force recompute before the
top-level search was trusted.

## What was built

- `causallearn/search/ScoreBased/XGES.py`
  - `xges(...)` — top-level entry mirroring `ges(...)` (same score-func options,
    same `Record['G']` / `Record['score']` return contract, same `GeneralGraph`
    edge convention).
  - The three operators **Insert / Delete / Reverse** with their score-deltas
    and validity conditions (paper Eqs. 6–12 and Table 2). Reverse is the
    operator GES does not have.
  - The **XGES-0** three-operator greedy loop (paper Algorithm 2), with the
    Delete → Reverse → Insert priority.
  - The full **XGES** extended local-optima escape (paper Algorithm 3),
    selectable via `extended_search` (default `True`).
- `tests/TestXGES.py` — the SHD correctness-floor test, an XGES≡GES equivalence
  test, the operator-vs-brute-force honesty test, and the `xges`-PyPI parity
  oracle (skipped when the package is absent — see below).
- `docs/.../XGES.rst` + `index.rst` entry (mirroring DGES).

## Clean-room statement

This is a clean-room implementation written **only** from the paper
(arXiv:2502.19551). The reference implementation (`github.com/ANazaret/XGES`,
C++/Python) and the derived `xges` PyPI package were **not** read, copied,
adapted, or paraphrased into the causal-learn code. The `xges` package is used
solely as an *output* oracle in the emitted parity test (comparing learned
CPDAGs, never importing its code into the diff). This matches house practice —
causal-learn's own GES is a from-algorithm port, not vendored source.

## Highest-risk parts and how they were tested

The two silently-wrong-prone parts are **(a) the CPDAG operator
correctness** (especially the Reverse operator, which GES does not have) and
**(b) the `alpha`↔`lambda_value` score calibration**.

**(a) Operators.** `tests/TestXGES.py::test_operator_change_scores_match_bruteforce`
enumerates *every* valid Insert/Delete/Reverse candidate on the ground-truth
CPDAG and asserts that each operator's advertised local score-delta equals the
change in the *total* decomposable score computed by a full
`score_g(pdag2dag(after)) - score_g(pdag2dag(before))` recompute. For a *valid*
operator these must be equal; they match to ~1e-12 (well under the `places=6`
assertion), including for the 6 Reverse candidates exercised. The greedy loop
additionally re-scores every accepted move against a full recompute and only
commits score-improving edits, so a mis-derived validity condition degrades
gracefully (the move is skipped) rather than corrupting the result.

**(b) Score calibration.** The paper's BIC is
`S = log p_hat(D) − (alpha / 2) · log(n) · |theta|`, with `alpha = 2` in the
paper's experiments (`alpha = 1` is textbook BIC). causal-learn's BIC penalizes
`lambda_value · (|Pa| + 1) · log(n)` per node, i.e. `lambda_value · log(n)` per
free parameter. Matching the per-parameter penalty gives

    lambda_value = alpha / 2

so the paper default `alpha = 2` → `lambda_value = 1.0` and textbook `alpha = 1`
→ causal-learn's default `lambda_value = 0.5`. The extra `+1` variance
parameter per node is a constant that cancels in every operator score-delta, so
it does not affect the search (only the absolute total score). `xges(...)`
accepts `alpha` and applies this mapping when `lambda_value` is not given.

## What the SHD test proves (and its limits)

`test_xges_simulate_linear_gaussian_shd` defines a known 5-node DAG, converts it
to the ground-truth CPDAG with causal-learn's own `dag2cpdag`, simulates
linear-Gaussian data from it (fixed seed), runs `xges(...)` in **both**
`extended_search=False` (XGES-0) and `True` (full XGES) modes, and asserts
`SHD(truth_CPDAG, learned) <= 1`. In practice it recovers the truth exactly
(SHD = 0). This proves XGES recovers the correct Markov equivalence class on a
small, faithful, well-sampled problem — the correctness floor. It does **not**
prove asymptotic-consistency guarantees or behavior under near-unfaithfulness,
latent confounding, or large/high-dimensional graphs.

`test_xges_matches_ges` further asserts `SHD(GES, XGES) == 0` on the same data:
XGES and GES land in the *same* equivalence class, as they must for a correct
score-based CPDAG learner.

> Note on the repo's committed GES fixture: the pre-existing
> `tests/TestGES.py::test_ges_simulate_linear_gaussian_with_local_score_BIC`
> currently fails in this checkout (its committed data/CPDAG fixture is stale —
> plain GES scores SHD = 6 on it). XGES reproduces GES *exactly* on that same
> fixture (identical graph and score), which is the correct behavior. Rather than
> depend on the stale fixture, `TestXGES.py` generates its own faithful data with
> a fixed seed, where both GES and XGES recover SHD = 0. The stale GES fixture is
> pre-existing and out of scope for this change.

## Parity oracle — DEFERRED to a human (not run here)

**The `xges` PyPI parity oracle was NOT run in this environment.** Installing new
packages is blocked here (dependency-install hardening), so `pip install xges`
could not be executed. `test_xges_pypi_parity_oracle` is therefore **emitted and
guarded**: it `self.skipTest(...)`s when `xges` is not importable, so it does not
give a false green.

A human must run it on Colab before upstreaming:

```bash
pip install xges
python -m pytest tests/TestXGES.py::TestXGES::test_xges_pypi_parity_oracle -v
```

It runs the `xges` reference solver on the **same** synthetic data with the
`alpha = 2.0` ↔ `lambda_value = 1.0` mapped penalty and asserts the two learned
CPDAGs are identical (SHD = 0). Two caveats for the human runner:

- The reference package's public API is resolved defensively in the test
  (`_run_reference_xges`); if the installed version exposes a different entry
  point, adapt that one helper (it lives in the test file, never in the shipped
  algorithm).
- Do **not** copy any reference code into `causallearn/`; the package is an
  output oracle only.

## Explicitly out of scope (intentionally, not stubbed)

- **Incremental candidate maintenance.** The engine currently re-enumerates and
  re-evaluates candidates each step (an O(N²)-per-step recompute), which the
  brief explicitly permits as an acceptable first pass provided it returns the
  same CPDAG. The paper's reparametrized incremental invalidation (Sec. 4.2) is
  a pure performance optimization and is intentionally deferred; correctness and
  the returned equivalence class are unaffected.
- **numba acceleration** — optional in the paper and not a dependency here.
- The stale pre-existing `tests/TestGES.py` fixture is left untouched (the brief
  forbids modifying GES and its tests).
