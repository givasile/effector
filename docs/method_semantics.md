# Method semantics by feature type

What every global method returns from `eval` (mean effect), `eval_heter` (heterogeneity
curve h), `heter_score` (scalar H), and `plot`, for **continuous**, **ordinal**, and
**nominal** features. This is the exactness contract of the package: implementations must
match these formulas.

**Notation.** Model $f$, data $\{x^{(i)}\}_{i=1}^N$, feature of interest $s$, the rest $c$.
Continuous: interval $[a,b]$ (= `axis_limits`). Ordinal/nominal: observed levels
$v_1,\dots,v_K$ with counts $N_k$ and frequencies $w_k = N_k/N$. Ordinal levels are in
their natural numeric order. Nominal levels are indexed by an order $\pi$: declared
(`order=[...]`), induced (`order="similarity"`), or by default ascending encoded value.

## Shared rules

- **Discrete features are evaluated only at levels.** `eval`/`eval_heter` at any other
  value raises `ValueError`; the model is never queried at non-existing category values.
- **Centering** (applies to `eval` only; h is centering-invariant):
  `zero_integral` subtracts $c = \frac{1}{30}\sum_{j=1}^{30}\hat\mu(x_j)$ (uniform grid)
  for continuous, $c = \sum_k w_k\, \hat\mu(v_k)$ (frequency-weighted level mean) for
  discrete. `zero_start` subtracts $c=\hat\mu(a)$ for continuous, $c=\hat\mu(v_1)$ for
  discrete (reference level = 0, as in dummy coding). Constants are always
  derived from the cached local effects (design contract R14) — computing one,
  globally or masked, never touches the model.
- **heter_score — the units contract.** The scalar $H$ is a **std-type quantity in
  output units**, data-weighted: "a typical instance's effect deviates from the mean
  effect by about this much." The curve $h$ stays a *variance in the method's native
  units* (that is what the plot whiskers $\sqrt h$ read); only the scalar is unified:

  $$H = \underbrace{\sqrt{\tfrac{1}{N}\textstyle\sum_i h(x^{(i)}_s)}}_{\text{RMS over the (masked) data values}} \times\ \text{bridge},
  \qquad H_{discrete} = \sqrt{\textstyle\sum_k w_k\, h(v_k)} \times \text{bridge}$$

  The **bridge** converts slope-native methods into output units — slope
  disagreement × typical excursion of the feature = output-level disagreement:
  $\text{std}(x_s)$ for ALE/RHALE/DerPDP-continuous, $\text{std(codes)}$ for their
  ordinal variants, and $1$ (no bridge) for PDP/ShapDP, whose $h$ is already in
  output² units. The bridge always reads the FULL data column — frozen under masks,
  so regional heterogeneity drops measure dispersion reduction, not range shrinkage.
  Nominal features in the difference-based methods bypass the chain entirely
  (all-pairs, below). $H$ is what `find_regions` consumes, and it is comparable
  across feature types within a method (exact on the canaries), and across methods
  in magnitude. Canary: for $f = x_s\, g(x_c)$, PDP, ALE, RHALE and DerPDP all give
  $H = \text{std}(x_s)\,\text{std}(g)$.
- **Plots**: continuous → curve with ±$\sqrt{h}$ band (or ICE curves). Ordinal → bars at
  the true numeric positions $v_k$ with whiskers $\sqrt{h(v_k)}$; `heterogeneity="ice"` →
  per-level jittered dots. Nominal → same bars at positions $1..K$ with level labels as
  ticks; bar order follows $\pi$; PDP/ShapDP may reorder for display (`order="effect"`).

## Masked evaluation (regional ≡ masked global)

`eval`/`eval_heter`/`heter_score`/`plot` all accept `mask=` — a boolean array
over the $N$ instances — and return the method's own summary restricted to the
selected instances. This is exactly what a region is (design contract R11):
`find_regions(feature)` returns a `Partition`, and `partition.plot(idx)` =
`effect.plot(feature, mask=region_mask)` (R12). Masked calls are model-free —
they re-summarize the stored per-instance local effects (one exception below).
Per method, what re-runs and what stays frozen:

- **PDP / DerPDP** — the canonical grid is frozen (global frame); the position
  store only grows as new x-positions are evaluated. Masked mean and variance
  are re-averaged over the masked *columns* of the cached ICE (d-ICE) table;
  centering constants are derived from those cached columns — per instance,
  global and masked alike. The one allowed model touch: `eval(mask=)` at
  points off the cached positions recomputes ICE on `data[mask]` exactly
  (transient, never cached); heterogeneity and plot always interpolate from
  the canonical grid.
- **ALE** — bin edges are frozen (the edge-bound secants cannot be recomputed
  without touching the model). Masked per-bin means/variances are re-averaged
  from the stored per-instance effects; bins left empty by the mask are
  interpolated (`fill_nans`) and edge bins extend flat.
- **RHALE / ShapDP** — the stored local effects are per-instance (jacobian
  values / SHAP values), so a masked call re-runs binning on the masked data.
  The `binning_scope` fit kwarg selects the range handed to the binner:
  `"global"` (default) = the global `axis_limits` interval, `"effective"` =
  the masked column's `[min, max]`. It is recorded at fit and replayed on
  every masked call, so the split search's `heter_score(mask)` and every
  masked eval/plot share the same scope.
- **Categorical features** keep the parent's level frame: means/variances are
  re-aggregated per level within the mask, level weights $w_k$ become the
  within-mask frequencies, and levels absent from the region are interpolated
  (ALE transitions) or dropped from the bars.

Masked plots window the x-axis to the effective interval, so node plots look
"zoomed" by default. A degenerate mask (empty, or collapsing the feature to a
point) raises `ValueError`.

## PDP

ICE curve: $\hat f^{(i)}(x) = f(x,\, x^{(i)}_c)$.

| | eval (uncentered) | eval_heter |
|---|---|---|
| continuous | $\hat\mu(x)=\frac{1}{N}\sum_i \hat f^{(i)}(x)$ | $h(x)=\mathrm{Var}_i\big[\tilde f^{(i)}(x)\big]$, each ICE centered by its own grid mean |
| ordinal / nominal | same, defined at $v_k$ only | same at $v_k$; ICE centered by its frequency-weighted level mean |

$H$: continuous $= \sqrt{\frac{1}{N}\sum_i \mathrm{Var}_i[\tilde f^{(i)}(x^{(i)}_s)]}$
(data-weighted RMS); ordinal/nominal
$= \sqrt{\sum_k w_k\, \mathrm{Var}_i[\tilde f^{(i)}(v_k)]}$. No bridge: centered ICE
values are already in output units.

Ordinal vs nominal: identical math — only axis semantics and display order differ.

## DerPDP

**Continuous**: $\hat\mu(x)=\frac{1}{N}\sum_i \partial_s f(x,\,x^{(i)}_c)$,
$h(x)=\mathrm{Var}_i[\partial_s f(x, x_c^{(i)})]$ (d-ICE variance, no centering),
$H = \sqrt{\frac{1}{N}\sum_i h(x^{(i)}_s)}\ \times \text{std}(x_s)$ (data-weighted RMS,
std-bridged into output units).

**Ordinal** — the derivative becomes the finite difference of the *plain* ICE values
(the jacobian is never used on a discrete axis), over the **marginal** distribution
(every instance contributes to every transition, unlike ALE's conditional $S_t$):

$$d^{(i)}_t = f(v_t, x^{(i)}_c) - f(v_{t-1}, x^{(i)}_c) \quad \text{for all } i$$

$\mathrm{eval}(v_j) = \mu_{t(j)}$ and $h(v_j) = \sigma^2_{t(j)}$ with ALE's step-into
convention ($v_1$ carries the first transition); the plot shows $K-1$ transition bars
labeled $v_{t-1}{\to}v_t$. $H = \sqrt{\sum_k w_k h(v_k)} \times \text{std(codes)}$;
importance $= \sum_k w_k |\mu_{t(k)}| \times \text{std(codes)}$.

**Nominal** — scalars from **all pairs** of the cached ICE-at-levels matrix (order-free,
zero extra model cost, marginal distribution): $H = \sqrt{\sum_{a<b} w_a w_b\,
\mathrm{Var}_i[f(v_b,x^{(i)}_c) - f(v_a,x^{(i)}_c)]}$, importance likewise over the
squared pair means. The transition-bar display follows the encoded order with the usual
nominal caveat.

## ALE

**Continuous** (bins $z_0<\dots<z_M$, $S_k$ = instances in bin $k$):

$$d^{(i)} = \frac{f(x^{(i)}_{s=z_k}) - f(x^{(i)}_{s=z_{k-1}})}{z_k - z_{k-1}}, \quad
\mu_k = \mathrm{mean}_{i\in S_k}\, d^{(i)}, \quad \sigma_k^2 = \mathrm{Var}_{i \in S_k}\,d^{(i)}$$

$\mathrm{eval}$: $\widehat{ALE}(x) = \sum_{k \le k(x)} \mu_k\,\Delta z_k$ (last bin
truncated at $x$), then centered. $\mathrm{eval\_heter}$: $h(x) = \sigma^2_{k(x)}$ (step
function, slope-native units). $H = \sqrt{\sum_k p_k\,\sigma_k^2}\ \times \text{std}(x_s)$
— the data-mass-weighted RMS of the per-bin slope dispersions ($p_k$ = fraction of the
(masked) data in bin $k$), bridged into output units by the full column's std.

**Ordinal** — bin edges are the observed values themselves; transitions $t=2,\dots,K$
between $v_{t-1}$ and $v_t$; contributors $S_t = \{i : x^{(i)}_s \in \{v_{t-1}, v_t\}\}$:

$$d^{(i)}_t = \frac{f(v_t, x^{(i)}_c) - f(v_{t-1}, x^{(i)}_c)}{v_t - v_{t-1}}$$

$\widehat{ALE}(v_j) = \sum_{t=2}^{j} \mu_t\,(v_t - v_{t-1})$ — the gap normalization
cancels, so accumulated values equal the sum of mean raw adjacent differences and do not
depend on level spacing. $h(v_j) = \sigma^2_j$, the variance of the step *into* level $j$
(convention: $h(v_1)$ = first transition's variance).
$H = \sqrt{\sum_k w_k\, h(v_k)} \times \text{std(codes)}$ (frequency-weighted RMS,
code-space bridge). Exact: the model is only queried at real levels.

**Nominal** — the *curve* is identical after reindexing levels by $\pi$, with the caveat
(documented wherever plotted) that the accumulated shape depends on $\pi$ and only the
adjacent differences $\mu_t$ are meaningful. The *scalars* never see $\pi$: they are
computed from **all level pairs** — for every pair $a<b$, every instance at $v_a$ or
$v_b$ contributes the raw difference $d^{(i)}_{ab} = f(v_b, x^{(i)}_c) - f(v_a, x^{(i)}_c)$
(conditional contributors, as in the chain), and

$$H = \sqrt{\textstyle\sum_{a<b} w_a w_b\, \mathrm{Var}_i\, d^{(i)}_{ab}},
\qquad \text{importance} = \sqrt{\textstyle\sum_{a<b} w_a w_b\, \big(\mathrm{mean}_i\, d^{(i)}_{ab}\big)^2}$$

(the $a<b$ sum is the $\tfrac12\sum_k\sum_a$ of the symmetric form — the factor that makes
the all-pairs dispersion equal the level-value variance, $E[(X-X')^2] = 2\,\mathrm{Var}$).
No bridge: raw differences are already in output units. Order-free by construction —
`order=` affects the plotted accumulation only, never $H$ or importance.

## RHALE

**Continuous**: as ALE but $d^{(i)} = \partial_s f(x^{(i)})$ evaluated at the data points,
and bin limits chosen by Greedy/DP minimizing a variance-based cost. $h$ and $H$ as in
ALE (per-bin variance step function; width-weighted grid mean).

**Ordinal**: $d^{(i)}_t$ as in ALE-ordinal (the discrete derivative); Greedy/DP merges
*adjacent transitions* → **adaptive level grouping**; within a merged bin, $\mu_B, \sigma_B^2$
over its pooled contributors; accumulation as in ALE over merged bins.
$h(v_j) = \sigma^2_{B(v_j)}$ (the merged bin containing level $j$'s transition);
$H = \sqrt{\sum_k w_k\, h(v_k)} \times \text{std(codes)}$ (as ALE-ordinal).

**Nominal** — identical to ALE-nominal: one bin per transition, no grouping, and the
same order-free all-pairs scalars. The Jacobian is irrelevant on a discrete axis, and
merging *adjacent* transitions presumes the adjacency is real — under an arbitrary or
induced order it is our artifact, so grouping would launder it into the statistics. The
ALE-nominal order caveat applies unchanged.

## ShapDP

$\phi^{(i)}$ = SHAP value of feature $s$ for instance $i$ (computed at data points).

| | eval (uncentered) | eval_heter |
|---|---|---|
| continuous | linear interpolation of per-bin means $\big(\bar z_k,\ \mu_k = \mathrm{mean}_{i \in S_k} \phi^{(i)}\big)$ | interpolation of per-bin variances $\sigma_k^2$ |
| ordinal / nominal | step lookup: $\hat s(v_k) = \bar\phi_k = \frac{1}{N_k}\sum_{i:\,x^{(i)}_s = v_k} \phi^{(i)}$ | $h(v_k) = \mathrm{Var}_{i:\,x^{(i)}_s=v_k}\,\phi^{(i)}$ |

$H$: continuous $= \sqrt{\frac{1}{N}\sum_i h(x^{(i)}_s)}$ (data-weighted RMS over the
interpolated variance); ordinal/nominal
$= \sqrt{\sum_k w_k\,\mathrm{Var}_{i:\,x^{(i)}_s=v_k}\,\phi^{(i)}}$. No bridge: $\phi$
is already in output units.

No order enters the math (coalitional); nominal vs ordinal differ in display only.
Plot: bars + jittered per-level $\phi$ dots (continuous: curve + scatter, as now).

## Capability matrix

| method | continuous | ordinal | nominal |
|---|---|---|---|
| PDP / ICE | ✓ (as now) | ✓ levels, bars | ✓ levels, bars |
| DerPDP | ✓ (as now) | ✓ ICE differences, transition bars | ✓ transition bars + all-pairs scalars |
| ALE | ✓ (as now) | ✓ exact, value-edged bins | ✓ order caveat; all-pairs scalars |
| RHALE | ✓ (as now) | ✓ discrete derivative + level grouping | ✓ as ALE (no grouping) |
| ShapDP | ✓ (as now) | ✓ per-level, step lookup | ✓ per-level, step lookup |

No holes: every method explains every feature type; feature type selects the kernel
(gradients/secants on continuous axes, level differences on discrete ones), never
whether a feature appears at all.

## Importance (R13)

`importance(feature, mask=None) -> float >= 0` is the **μ-twin of `heter_score`**:
where `heter_score` measures the per-instance spread of the effect, `importance`
measures how much the **mean** effect varies — same output units, same data
weighting, so the two are a mean/spread pair on one scale (the triage plane's
two axes):

- **continuous**: the standard deviation of the mean effect over the (masked)
  data values. For a linear model this is `|a_j| * std(x_j)`.
- **discrete**: the frequency-weighted standard deviation over the levels;
  **nominal** in the difference-based methods (ALE/RHALE) is order-free via the
  all-pairs means (the chain's accumulated values are path-dependent under an
  arbitrary order).
- **ShapDP** overrides the default with the canonical `mean(|phi_s|)` over the
  (masked) instances — `phi_s` is the cached local effect (already output units).
- **DerPDP** overrides it with the mean `|derivative|` over the (masked) data
  values × the std bridge: its mean effect is already a derivative, so the
  *dispersion* would be ~0 for a linear model (a useless ranking), while the
  bridged magnitude recovers `|a_j| * std(x_j)` — the same closed form as the
  other methods.

It is centering-invariant (no `centering` argument — the std of the mean effect
does not depend on the additive centering constant) and model-free (re-summarized
from the cached local effects; a `mask` restricts it to a subregion through the
same memo). `importances(mask=None) -> (D,)` is the per-feature vector, returning
`NaN` (with one `UserWarning`) for feature types a method cannot explain.
effector never sees `y`, so loss/permutation importance is out of scope by
construction.
