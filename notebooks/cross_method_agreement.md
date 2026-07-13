# Cross-method sanity sweep — agreement notes

Every example notebook now ends with a *cross-method sanity check*: the
one-liner `effector.explain` run with every engine the notebook's model
supports (all five where a Jacobian exists; DerPDP has no variance ledger by
design — derivative scale). This file summarizes what agreed and what is
worth a closer look. The per-notebook tables live in the notebooks
themselves (last cell, tagged `explain-sweep`).

Generated 2026-07-13 on branch `feat/calm-chain`. 19/20 notebooks executed
end to end; `03_california_housing_tabpfn` skipped (needs `TABPFN_TOKEN`).

## Verdict at a glance

| Notebook | Agreement | Note |
|---|---|---|
| synthetic 01 linear | perfect | identical rankings, GAM R² = 100%, no splits, all five methods |
| synthetic 02 methods-comparison | good | PDP GAM 70% vs others ~92–94% (the notebook's own point); mirrored split choice; all converge ≥ 97.6% |
| synthetic 03 regional (synthetic f) | strong | PDP/ALE/RHALE identical (x1 on x3 → ~100%); ShapDP adds the twin split |
| synthetic 04 regional (real f) | strong | same split everywhere (~99%); ShapDP flips x3/x1 rank |
| synthetic 05 ×3 (conditional interaction) | good | PDP/ALE/ShapDP agree (GAM ~85%); RHALE GAM 70% — see signature C; all accept x0-on-x1, all ≥ 96.6% |
| synthetic 06 general interaction | near-perfect | identical rankings, same split, same R² (94.0% → 99.6%) across all |
| synthetic 07 four regions | good | PDP/ALE/RHALE identical; ShapDP splits on x2 only + twin, ranks x2 above x0 |
| synthetic 08 categorical | good | PDP/RHALE/ShapDP: x1-on-level,x2 (98.4%); ALE picks the mirror level-on-x1,x2 (97.4%) |
| synthetic 09 explain/importance | strong | PDP/ALE/RHALE identical (GAM ~2.7% → 100%); ShapDP GAM −0.6% → 87.3% + twin |
| quickstart simple_api | good | pure interaction: GAM ≈ 0 for all; ALE picks the mirrored split (both valid on a symmetric model) |
| quickstart flexible_api | **examine** | **ALE accepts no split at all (28.8% flat) while PDP 99.8% / RHALE 99.6% / ShapDP 93.4%** |
| quickstart readme_example (bike, keras) | good | x_3 top everywhere; same two core splits in PDP/ALE/RHALE; ShapDP adds smaller extras; final 82.7–90.8% |
| guides efficiency_global | **examine** | ALE takes the mirrored split and stops at 78.2% vs ~100% for PDP/RHALE; RHALE GAM 3.7% vs ~36% |
| guides efficiency_regional | strong | PDP/ALE/RHALE identical (24.5% → ~100%); ShapDP twin + 92.1% |
| real 01 bike sharing | **strong** | all methods: `hr` dominant, the SAME accepted split (`hr` on temp, workingday, yr), final 88.4–88.8% |
| real 02 california housing | fair | Latitude/Longitude rank swap between methods; different conditioning sets accepted; final 81.6–87.1% (correlated features — expected spread) |
| real 04 NO2 | strong | `cars_per_hour` top everywhere; `wind_speed` split accepted by all; final 95.3–95.5% |
| real 03 tabpfn | skipped | `TABPFN_TOKEN` unset |

## Recurring signatures (properties, not bugs)

**A. ShapDP accepts both twins of a symmetric interaction.** Where
PDP/ALE/RHALE accept one split (x0-on-x1) and reject its mirror as
redundant, ShapDP routinely accepts both (x0-on-x1 *and* x1-on-x0), each
with real marginal R². Shapley attribution splits the interaction credit
between the two features, so each feature's own curves carry half the
deviation and each partition genuinely explains variance the other does
not. Seen in notebooks 02, 03, 04, 06, 07, 09, efficiency_regional.
ShapDP's final R² is also consistently a few points below the
PDP/ALE/RHALE ceiling (87–98% vs ~100% on synthetics) — the smeared curves
are not exactly leaf-conditional means.

**B. Mirrored-split choice on symmetric models.** For targets like
`x_i · sign(x_j)`, splitting x_i on x_j or x_j on x_i are equally valid
single-split stories, and methods legitimately pick different twins
(ALE tends to pick the opposite one from PDP/RHALE: notebooks 02, 08,
simple_api, efficiency_global). Both resolve the heterogeneity; the ledger
R² is the arbiter of whether the chosen twin explains as much.

**C. RHALE reads a smaller GAM share on conditional-interaction models.**
On the 05x family (70% vs ~85%), efficiency_global (3.7% vs ~36%) and
flexible_api (1.3% vs ~38%), RHALE's global curves capture visibly less
first-order variance than PDP/ALE — sign-cancelling interactions flatten
the averaged derivative more aggressively — yet after the accepted splits
RHALE lands at ~100%, often the highest. Reading: RHALE is the most
conservative about crediting the global read and the most rewarded by
regions. Worth one focused example in the paper.

**D. DerPDP.** No variance ledger by design (its curves live in ∂f/∂x
units, so an output-scale additive surrogate is undefined); rankings are
in derivative units and mostly track RHALE's. On real data with ordinal
features the derivative methods drop those features (unsupported type),
so their rankings can be shorter than PDP/ALE/ShapDP's.

## Worth examining further

1. **flexible_api + ALE: no split accepted (28.8% → 28.8%).** Every other
   method converts the same heterogeneity into ~93–100% explained. Either
   ALE's candidate partition for x_0 dies at the finder's drop threshold,
   or its marginal R² lands under `min_r2_gain`. One session with
   `ale.find_regions(0)` + `ale.select_regions()` on that notebook's model
   will tell which.
2. **efficiency_global + ALE: mirrored split underperforms (78.2%).** Same
   family as (B), but here the twin choices are NOT equivalent in R² —
   a clean demonstration that the ledger catches what heterogeneity alone
   cannot; could become a docs/paper example rather than a fix.
3. **California housing:** Latitude/Longitude importance swaps and
   different accepted conditioning sets across methods — correlated
   geography features. Expected, but a good candidate for the paper's
   "methods disagree ⇒ data property" narrative.

## Two defects the sweep caught (both fixed on this branch)

- `utils.compute_jacobian_numerically` crashed on models returning
  `(N, 1)` column vectors (RHALE/DerPDP with `model_jac=None`); it now
  flattens the model output like the rest of the pipeline.
- The bike-sharing notebook's two report cells passed the raw keras
  `model` (returns `(N, 1)`) instead of the notebook's own handshake-checked
  `model_forward` wrapper; RHALE's nominal-feature kernel choked on it.
  The cells now respect the numpy-only contract. (Open hardening question:
  should `ingest` probe the model output shape up front, the way
  `adapters.check` does, and fail loudly at construction?)
