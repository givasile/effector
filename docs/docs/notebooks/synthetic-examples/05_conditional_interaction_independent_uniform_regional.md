# Model with conditional interaction

- Author: [givasile](https://givasile.github.io/)
- Runtime: ~5 s
- Description: Regional effects (PDP, ALE, RHALE) on a model with a
  conditional interaction: every method must find the split on $x_2$ at 0 and
  recover the per-region closed forms, tested against `effector.benchmarks`.

In this example, we show regional effects of a model with conditional interactions using PDP, ALE, and RHALE.
In particular, we:

1. show how to use `effector` to estimate the regional effects using PDP, ALE, and RHALE
2. provide the analytical formulas for the regional effects
3. test that (1) and (2) match

We will use the following model: 

$$ 
f(x_1, x_2, x_3) = -x_1^2 \mathbb{1}_{x_2 <0} + x_1^2 \mathbb{1}_{x_2 \geq 0} + e^{x_3} 
$$

where the features $x_1, x_2, x_3$ are independent and uniformly distributed in the interval $[-1, 1]$.


The model has an _interaction_ between $x_1$ and $x_2$ caused by the terms: 
$f_{1,2}(x_1, x_2) = -x_1^2 \mathbb{1}_{x_2 <0} + x_1^2 \mathbb{1}_{x_2 \geq 0}$.
This means that the effect of $x_1$ on the output $y$ depends on the value of $x_2$ and vice versa.
Therefore, there is no golden standard on how to split the effect of $f_{1,2}$ to two parts, one that corresponds to $x_1$ and one to $x_2$.
Each global effect method has a different strategy to handle this issue.
Below we will see how PDP, ALE, and RHALE handle this interaction.

In contrast, $x_3$ does not interact with any other feature, so its effect can be easily computed as $e^{x_3}$.


```python
import numpy as np
import matplotlib.pyplot as plt
import effector

np.random.seed(21)

bench = effector.benchmarks.ConditionalInteractionUniform()
model = bench.model
dataset = bench.dataset
x = bench.generate_data(1_000)
```

## Why regional effects?

As shown in the [global-effects notebook](./05_conditional_interaction_independent_uniform_global.md),
the global effect of $x_1$ is **zero with high heterogeneity**: for instances
with $x_2 < 0$ the local effect is $-x_1^2$ and for $x_2 \geq 0$ it is $+x_1^2$,
so the average washes out.

Regional effect methods ask the natural follow-up: *is there a split of the
input space that makes the local effects agree within each subregion?* Here the
answer is known by construction — splitting on $x_2 = 0$ yields two regions
where the effect of $x_1$ is deterministic:

$$
\text{effect}(x_1 \mid x_2 < 0) = -x_1^2, \qquad
\text{effect}(x_1 \mid x_2 \geq 0) = +x_1^2,
$$

each with **zero** heterogeneity. After zero-integral centering over
$x_1 \in [-1, 1]$ (the mean of $x_1^2$ is $1/3$), the closed forms are
$\mp x_1^2 \pm 1/3$. Every regional method below must recover: the split
feature ($x_2$), the split position ($\approx 0$), and the per-region curves.

## New-API: importance and one-click explain

Before drilling into the regional splits, the new API offers two shortcuts on
top of the global effect. `importances()` ranks features by the **dispersion of
their mean effect** (the μ-twin of heterogeneity): here $x_3$ (the monotone
$e^{x_3}$) carries a large mean effect, while $x_1$'s mean effect washes out to
$\approx 0$ (its signal lives entirely in the *heterogeneity* that the regional
split below explains). `effector.explain(...)` runs the whole pipeline once and
returns a serializable `Report`.


```python
# per-feature importance = dispersion of the mean effect (mu-twin of heterogeneity)
fx = effector.PDP(
    data=x, model=model.predict,
    axis_limits=dataset.axis_limits,
    nof_instances="all",
)
print("importances:", np.round(fx.importances(), 3))

# one-click auto-explanation -> Report (serializable; self-contained HTML)
report = effector.explain(x, model.predict, method="pdp", nof_instances="all")
report.show()
```

    importances: [0.008 0.33  0.686]
    
    PDP report — target: y
    ============================================================
    feature                   importance     heter  #regions
    ------------------------------------------------------------
    x_2                           0.6851    0.0000         1
    x_1                           0.3302    0.0852         7
    x_0                           0.0076    0.1005         3
    ============================================================
    
    
    Feature 1 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    x_1 🔹 [id: 0 | heter: 0.09 | inst: 1000 | w: 1.00]
        x_0 ≤ -0.70 🔹 [id: 1 | heter: 0.02 | inst: 167 | w: 0.17]
            x_0 ≤ -0.90 🔹 [id: 2 | heter: 0.00 | inst: 56 | w: 0.06]
            x_0 > -0.90 🔹 [id: 3 | heter: 0.01 | inst: 111 | w: 0.11]
        x_0 > -0.70 🔹 [id: 4 | heter: 0.06 | inst: 833 | w: 0.83]
            x_0 ≤ 0.60 🔹 [id: 5 | heter: 0.02 | inst: 648 | w: 0.65]
            x_0 > 0.60 🔹 [id: 6 | heter: 0.03 | inst: 185 | w: 0.18]
    --------------------------------------------------
    Feature 1 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.09
        Level 1🔹heter: 0.05 | 🔻0.03 (36.45%)
            Level 2🔹heter: 0.02 | 🔻0.04 (67.03%)
    
    
    
    
    Feature 0 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    x_0 🔹 [id: 0 | heter: 0.10 | inst: 1000 | w: 1.00]
        x_1 ≤ 0.00 🔹 [id: 1 | heter: 0.00 | inst: 488 | w: 0.49]
        x_1 > 0.00 🔹 [id: 2 | heter: 0.00 | inst: 512 | w: 0.51]
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.10
        Level 1🔹heter: 0.00 | 🔻0.10 (100.00%)
    
    


## Regional PDP

### Effector


```python
pdp = effector.PDP(
    data=x, model=model.predict,
    axis_limits=dataset.axis_limits,
    nof_instances="all",
)
pdp.fit("all", centering=True)
finder = effector.space_partitioning.Best()
partitions_pdp = {feat: pdp.find_regions(feat, finder=finder) for feat in range(3)}
partitions_pdp[0].show()
```

    
    
    Feature 0 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    x_0 🔹 [id: 0 | heter: 0.10 | inst: 1000 | w: 1.00]
        x_1 ≤ 0.00 🔹 [id: 1 | heter: 0.00 | inst: 488 | w: 0.49]
        x_1 > 0.00 🔹 [id: 2 | heter: 0.00 | inst: 512 | w: 0.51]
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.10
        Level 1🔹heter: 0.00 | 🔻0.10 (100.00%)
    
    



```python
for region_idx in [1, 2]:
    partitions_pdp[0].plot(region_idx, heterogeneity="ice", centering=True, y_limits=[-1.5, 1.5])
```


    
![png](05_conditional_interaction_independent_uniform_regional_files/05_conditional_interaction_independent_uniform_regional_8_0.png)
    



    
![png](05_conditional_interaction_independent_uniform_regional_files/05_conditional_interaction_independent_uniform_regional_8_1.png)
    


### Tests


```python
# the same closed forms the test suite asserts against
# (tests/test_functional_conditional_interaction.py::TestRegionalEffects)
xx = np.linspace(-1, 1, 100)


def check_regions(partition):
    children = [r for r in partition if r.level == 1]
    assert len(children) == 2
    for r in children:
        # the split must be on x2 at ~0
        assert r.foc_index == bench.regional_split_feature
        assert abs(r.foc_split_position - bench.regional_split_position) <= 0.15
        # inside each region: -+x1^2 (centered), with ~zero heterogeneity
        side = "left" if r.comparison == "<=" else "right"
        y = partition.eval(r.idx, xx, centering=True)
        heter = partition.eval_heter(r.idx, xx)
        np.testing.assert_allclose(y, bench.regional_effect_gt(side, xx), atol=1e-1)
        np.testing.assert_allclose(heter, np.zeros_like(xx), atol=1e-1)


check_regions(partitions_pdp[0])
```

## Regional ALE

### Effector


```python
ale = effector.ALE(
    data=x, model=model.predict,
    axis_limits=dataset.axis_limits,
    nof_instances="all",
)
ale.fit("all", centering=True)
finder = effector.space_partitioning.Best()
partitions_ale = {feat: ale.find_regions(feat, finder=finder) for feat in range(3)}
partitions_ale[0].show()
```

    
    
    Feature 0 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    x_0 🔹 [id: 0 | heter: 1.39 | inst: 1000 | w: 1.00]
        x_1 ≤ 0.00 🔹 [id: 1 | heter: 0.00 | inst: 488 | w: 0.49]
        x_1 > 0.00 🔹 [id: 2 | heter: 0.00 | inst: 512 | w: 0.51]
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 1.39
        Level 1🔹heter: 0.00 | 🔻1.39 (100.00%)
    
    



```python
for region_idx in [1, 2]:
    partitions_ale[0].plot(region_idx, centering=True, y_limits=[-1.5, 1.5])
```


    
![png](05_conditional_interaction_independent_uniform_regional_files/05_conditional_interaction_independent_uniform_regional_13_0.png)
    



    
![png](05_conditional_interaction_independent_uniform_regional_files/05_conditional_interaction_independent_uniform_regional_13_1.png)
    


### Tests


```python
check_regions(partitions_ale[0])
```

## Regional RHALE

### Effector


```python
rhale = effector.RHALE(
    data=x, model=model.predict, model_jac=model.jacobian,
    axis_limits=dataset.axis_limits,
    nof_instances="all",
)
rhale.fit("all", centering=True)
finder = effector.space_partitioning.Best()
partitions_rhale = {feat: rhale.find_regions(feat, finder=finder) for feat in range(3)}
partitions_rhale[0].show()
```

    
    
    Feature 0 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    x_0 🔹 [id: 0 | heter: 1.32 | inst: 1000 | w: 1.00]
        x_1 ≤ 0.00 🔹 [id: 1 | heter: 0.00 | inst: 488 | w: 0.49]
        x_1 > 0.00 🔹 [id: 2 | heter: 0.00 | inst: 512 | w: 0.51]
    --------------------------------------------------
    Feature 0 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 1.32
        Level 1🔹heter: 0.00 | 🔻1.32 (99.75%)
    
    



```python
for region_idx in [1, 2]:
    partitions_rhale[0].plot(region_idx, centering=True, y_limits=[-1.5, 1.5])
```


    
![png](05_conditional_interaction_independent_uniform_regional_files/05_conditional_interaction_independent_uniform_regional_18_0.png)
    



    
![png](05_conditional_interaction_independent_uniform_regional_files/05_conditional_interaction_independent_uniform_regional_18_1.png)
    


### Tests


```python
check_regions(partitions_rhale[0])
```

## Conclusions

All three regional methods recover the ground truth: they split on $x_2$ at
$\approx 0$ and, inside each region, report the deterministic effect
$\mp x_1^2$ (centered) with heterogeneity dropping from $\sim 0.1$ at the root
to $\approx 0$ — the model's conditional interaction is *fully explained* by a
single split. This is the ideal-case benchmark for regional methods: when a
crisp subspace structure exists, the methods must find exactly it.
