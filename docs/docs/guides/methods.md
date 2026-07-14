---
title: "Methods: the math reference"
---

???+ success "Description"

    The mathematical reference of `effector`: how each of the five methods
    (PDP, DerPDP, ALE, RHALE, ShapDP) defines and estimates the global
    effect, its heterogeneity, and the two scalars, for continuous, ordinal
    and nominal features. Implementations must match these formulas; the
    canonical source is `docs/method_semantics.md` in the repository.

???+ note "Reading time"

	Approx. 12' to read; written to be consulted, not memorized.

## Notation

- $x_s$ is the feature of interest and $\mathbf{x}_c$ the rest, so that $\mathbf{x} = (x_s, \mathbf{x}_c)$.
- Data $\{x^{(i)}\}_{i=1}^N$. Continuous features live on the interval $[a,b]$
  (= `axis_limits`); ordinal/nominal features have observed levels
  $v_1,\dots,v_K$ with counts $N_k$ and frequencies $w_k = N_k/N$.
- $f^m(x_s)$ is the global effect of $x_s$ under method $m$ — what
  [`eval`](../quickstart/interactive/eval.md) returns.
- $h^m(x_s)$ is the **heterogeneity function** (`eval_heter`): a *variance*,
  in the method's native units; the plotted band is $\pm\sqrt{h}$.
- $H^m_{x_s}$ is the **heterogeneity value** (`heter_score`) and importance
  its μ-twin: scalars in **output units** (below).

**Centering** applies to `eval` only ($h$ is centering-invariant):
`zero_integral` subtracts the mean of the effect over the feature's range
(frequency-weighted level mean for discrete features); `zero_start` subtracts
$f^m(a)$ (or $f^m(v_1)$: reference level = 0, as in dummy coding).

???+ note "The units contract"

    The scalar $H$ is a **std type quantity in output units**, data-weighted:
    *"a typical instance's effect deviates from the mean effect by about this
    much."* The curve $h$ stays a variance in the method's native units; only
    the scalar is unified:

    $$H = \underbrace{\sqrt{\tfrac{1}{N}\textstyle\sum_i h(x^{(i)}_s)}}_{\text{RMS over the (masked) data values}} \times\ \text{bridge},
    \qquad H_{discrete} = \sqrt{\textstyle\sum_k w_k\, h(v_k)} \times \text{bridge}$$

    The **bridge** converts slope native methods into output units (slope
    disagreement × typical excursion of the feature): $\text{std}(x_s)$ for
    ALE/RHALE/DerPDP on continuous features, $\text{std(codes)}$ on ordinal
    ones, and $1$ (no bridge) for PDP/ShapDP, whose $h$ is already in output²
    units. The bridge always reads the **full** data column — frozen under
    masks, so regional heterogeneity drops measure dispersion reduction, not
    range shrinkage. Canary: for $f = x_s\, g(x_c)$, PDP, ALE, RHALE and
    DerPDP all give $H = \text{std}(x_s)\,\text{std}(g)$.

## PDP

The ICE curve of instance $i$ is $\hat f^{(i)}(x) = f(x,\, x^{(i)}_c)$, and
the Partial Dependence Plot is their average:

\begin{equation}
PDP(x_s) = \frac{1}{N} \sum_{i=1}^{N} f(x_s, \mathbf{x}_c^{(i)})
\end{equation}

The heterogeneity function is the variance of the centered ICE curves
(each ICE centered by its own mean) around the centered PDP:

\begin{equation}
h^{PDP}(x_s) = \frac{1}{N} \sum_{i=1}^{N} \left ( ICE_c^{(i)}(x_s) - PDP_c(x_s) \right )^2
\end{equation}

| feature type | eval | eval_heter | H |
|---|---|---|---|
| continuous | $\hat\mu(x)=\frac{1}{N}\sum_i \hat f^{(i)}(x)$ | ICE variance at $x$ | data-weighted RMS, **no bridge** |
| ordinal / nominal | same, defined at the levels $v_k$ only | same at $v_k$; ICE centered by its frequency-weighted level mean | $\sqrt{\sum_k w_k\, h(v_k)}$ |

Ordinal vs nominal: identical math; only axis semantics and display order
differ.

## DerPDP

The derivative PDP averages the model's partial derivative instead of its
value; everything lives on the **derivative scale**:

\begin{equation}
\hat\mu(x)=\frac{1}{N}\sum_i \partial_s f(x,\,x^{(i)}_c),
\qquad h(x)=\mathrm{Var}_i\big[\partial_s f(x, x_c^{(i)})\big]
\end{equation}

$H = \sqrt{\frac{1}{N}\sum_i h(x^{(i)}_s)} \times \text{std}(x_s)$: the
d-ICE dispersion, std-bridged into output units.

- **Ordinal**: the derivative becomes the finite difference of the plain ICE
  values over the *marginal* distribution (every instance contributes to
  every transition): $d^{(i)}_t = f(v_t, x^{(i)}_c) - f(v_{t-1}, x^{(i)}_c)$.
  The plot shows $K-1$ transition bars labeled $v_{t-1}{\to}v_t$;
  $H = \sqrt{\sum_k w_k h(v_k)} \times \text{std(codes)}$.
- **Nominal**: scalars from **all pairs** of the cached ICE-at-levels matrix
  (order-free, zero extra model cost):
  $H = \sqrt{\sum_{a<b} w_a w_b\, \mathrm{Var}_i[f(v_b,x^{(i)}_c) - f(v_a,x^{(i)}_c)]}$.

## ALE

ALE partitions $[a, b]$ into $K$ bins with edges $z_0<\dots<z_K$; $S_k$ holds
the instances whose $x_s$ falls in bin $k$. Each contributes a local slope:

\begin{equation}
d^{(i)} = \frac{f(z_k, x^{(i)}_c) - f(z_{k-1}, x^{(i)}_c)}{z_k - z_{k-1}},
\qquad
\mu_k = \mathrm{mean}_{i\in S_k}\, d^{(i)}, \quad \sigma_k^2 = \mathrm{Var}_{i \in S_k}\,d^{(i)}
\end{equation}

The effect accumulates the mean slopes, and the heterogeneity is the bin's
slope variance as a step function:

\begin{equation}
ALE(x_s) = \sum_{k \le k(x_s)} \mu_k\,\Delta z_k, \qquad h^{ALE}(x_s) = \sigma^2_{k(x_s)}
\end{equation}

\begin{equation}
H_{x_s}^{ALE} = \sqrt{\sum_{k=1}^{K} p_k \, \sigma_k^2} \; \times \; \text{std}(x_s)
\end{equation}

with $p_k$ the fraction of the (masked) data in bin $k$.

- **Ordinal**: bin edges are the observed values themselves; transitions run
  between adjacent levels with *conditional* contributors
  ($S_t = \{i : x^{(i)}_s \in \{v_{t-1}, v_t\}\}$). The gap normalization
  cancels in the accumulation, so the values do not depend on level spacing,
  and the model is only queried at real levels (exact).
- **Nominal**: the accumulated *curve* depends on the level order and only
  the adjacent differences are meaningful (a caveat printed wherever it is
  plotted). The *scalars* never see the order: they are computed from **all
  level pairs**,
  $H = \sqrt{\sum_{a<b} w_a w_b\, \mathrm{Var}_i\, d^{(i)}_{ab}}$ — raw
  differences already in output units, no bridge, order-free by construction.

## RHALE

As ALE on continuous features, with two changes: the local slope is the
**analytic derivative** evaluated at the data points,
$d^{(i)} = \partial_s f(x^{(i)})$, and the bin limits are chosen by a
variance-minimizing search (`"dp"` by default) instead of a fixed grid.
$h$ and $H$ read exactly as in ALE.

- **Ordinal**: the discrete derivative of ALE-ordinal, with dynamic
  programming merging *adjacent transitions* into groups; within a merged
  bin, mean and variance pool its contributors.
- **Nominal**: identical to ALE-nominal — one bin per transition, **no
  grouping** (merging adjacent transitions presumes the adjacency is real;
  under an arbitrary order it would launder an artifact into the statistics),
  and the same order-free all-pairs scalars. The jacobian is irrelevant on a
  discrete axis.

## ShapDP

$\phi^{(i)}$ is the SHAP value of feature $s$ for instance $i$, computed at
the data points. The curve summarizes the $(x^{(i)}_s, \phi^{(i)})$ cloud
per bin:

| feature type | eval | eval_heter |
|---|---|---|
| continuous | linear interpolation of per-bin means $\big(\bar z_k,\ \mu_k = \mathrm{mean}_{i \in S_k} \phi^{(i)}\big)$ | interpolation of per-bin variances $\sigma_k^2$ |
| ordinal / nominal | step lookup: $\hat s(v_k) = \bar\phi_k$ | $h(v_k) = \mathrm{Var}_{i:\,x^{(i)}_s=v_k}\,\phi^{(i)}$ |

$H$ is the data-weighted RMS (frequency-weighted over levels), **no bridge**:
$\phi$ is already in output units. No order enters the math (SHAP is
coalitional); nominal vs ordinal differ in display only.

## Importance

`importance(feature)` is the **μ-twin of `heter_score`**: where $H$ measures
the per-instance spread of the effect, importance measures how much the
**mean** effect varies — same output units, same data weighting, so the two
form a mean/spread pair on one scale (the
[triage plane](../quickstart/report/triage_plane.md)'s two axes).

- **PDP / ALE / RHALE / ShapDP**: the standard deviation of the mean effect
  over the (masked) data values (frequency-weighted over levels). For a
  linear model all four recover $|a_j| \cdot \text{std}(x_j)$.
- **DerPDP** is the one override: its mean effect is already a derivative,
  whose dispersion is ~0 for a locally linear model, so it uses the mean
  $|\text{derivative}|$ × the std bridge — recovering the same closed form.
- **Nominal features** in the difference based methods use the all-pairs
  means (order-free), mirroring $H$.

???+ note "No `y`, ever"

    effector never sees ground truth labels: importance here is a property of
    the fitted effect, not a loss based or permutation importance.

## Masked evaluation: regional ≡ masked global

Every verb accepts `mask=` (boolean over the $N$ instances) and returns the
method's own summary restricted to those instances — this **is** what a
region is (design contract [R11](./design.md)). Masked calls are model-free:
they re-summarize the stored per-instance local effects. What re-runs and
what stays frozen, per method:

| method | frozen (global frame) | recomputed under a mask |
|---|---|---|
| PDP / DerPDP | the evaluation grid | means/variances over the masked ICE columns; centering constants |
| ALE | the bin edges | per-bin means/variances from the stored per-instance effects; empty bins interpolated |
| RHALE / ShapDP | `binning_scope="global"` keeps the binner's range global | binning re-runs on the masked per-instance effects |
| categorical (all) | the level frame | per-level stats; level weights become within-mask frequencies |

The one allowed exception to zero model calls: PDP `eval(mask=)` at points
off the cached grid recomputes ICE on `data[mask]` exactly, transiently.
A degenerate mask (empty, or collapsing the feature to a point) raises
`ValueError`.

## Capability matrix

| method | continuous | ordinal | nominal |
|---|---|---|---|
| PDP / ICE | ✓ | ✓ levels, bars | ✓ levels, bars |
| DerPDP | ✓ | ✓ ICE differences, transition bars | ✓ transition bars + all-pairs scalars |
| ALE | ✓ | ✓ exact, value-edged bins | ✓ order caveat; all-pairs scalars |
| RHALE | ✓ | ✓ discrete derivative + level grouping | ✓ as ALE (no grouping) |
| ShapDP | ✓ | ✓ per-level, step lookup | ✓ per-level, step lookup |

No holes: every method explains every feature type; the feature type selects
the kernel (gradients or secants on continuous axes, level differences on
discrete ones), never whether a feature appears at all.

---

## Where to next

- [The design contract](./design.md): the rules these formulas live under
- [`importance` and `heter_score`](../quickstart/interactive/scores.md): the
  scalars, hands on
- [The interactive API](../quickstart/interactive_api.md): the verbs
