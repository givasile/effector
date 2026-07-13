# Adult Census Income — explaining a classifier's probability

- Author: [givasile](https://givasile.github.io/)
- Runtime: ~5 min
- Description: the variance ledger on a **classifier** — effector explains
  the predicted probability `P(income > 50K)` of a gradient-boosted model on
  the [Adult census dataset](https://archive.ics.uci.edu/dataset/2/adult)
  (UCI id 2). The probability surface turns out to be deeply interactive:
  global curves reproduce 72%, and it takes **four** splits — conditioned
  on `age`, `marital-status`, `capital-gain` and `education-num` — to
  reach 86%.

The dataset: 45,222 census records (after dropping rows with missing
values); the target is whether yearly income exceeds $50K. The model is a
`HistGradientBoostingClassifier`; effector sees only its
`predict_proba(...)[:, 1]` — a numpy → numpy function like any other.


```python
import effector
import numpy as np

np.random.seed(21)
```

## Load the data

`effector.datasets.AdultIncome` drops `fnlwgt` (a sampling weight) and
`education` (duplicated by `education-num`), encodes the categorical columns
to integer codes with the level names recorded, and buckets levels rarer
than 50 rows into `"Other"` — a level that rare can vanish from a train
split, leaving the schema promising a category the data never shows.


```python
data = effector.datasets.AdultIncome()

print(f"X_train: {data.x_train.shape}, X_test: {data.x_test.shape}")
print("-" * 66)
for i, name in enumerate(data.feature_names):
    cats = data.category_names[i]
    extra = f"{len(cats)} levels" if cats else "numeric"
    print(f"x_{i:<2} {name:16s} [{data.feature_types[i]:10s}] {extra}")
print("-" * 66)
print(f"target: {data.target_name}, base rate: {data.y_train.mean():.1%}")
```

    X_train: (36177, 12), X_test: (9045, 12)
    ------------------------------------------------------------------
    x_0  age              [continuous] numeric
    x_1  workclass        [nominal   ] 7 levels
    x_2  education-num    [ordinal   ] numeric
    x_3  marital-status   [nominal   ] 7 levels
    x_4  occupation       [nominal   ] 14 levels
    x_5  relationship     [nominal   ] 6 levels
    x_6  race             [nominal   ] 5 levels
    x_7  sex              [nominal   ] 2 levels
    x_8  capital-gain     [continuous] numeric
    x_9  capital-loss     [continuous] numeric
    x_10 hours-per-week   [continuous] numeric
    x_11 native-country   [nominal   ] 25 levels
    ------------------------------------------------------------------
    target: income>50K, base rate: 24.9%


## Fit a classifier


```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

clf = HistGradientBoostingClassifier(random_state=21).fit(data.x_train, data.y_train)
print(f"test AUC = {roc_auc_score(data.y_test, clf.predict_proba(data.x_test)[:, 1]):.3f}")
```

    test AUC = 0.925


## Explain

Class labels are not a regression surface — effector explains a per-class
probability. `effector.adapters.classifier_proba` wraps
`predict_proba(...)[:, class_]` into the numpy → numpy callable every engine
expects. From here on, everything — curves, heterogeneity, the ledger — is
in probability units.


```python
model_forward = effector.adapters.classifier_proba(clf, class_=1)
effector.adapters.check(model_forward, data.x_train)

schema = effector.Schema(
    feature_names=data.feature_names,
    feature_types=data.feature_types,
    category_names=data.category_names,
    target_name=data.target_name,
)
```

## The one-click report


```python
report = effector.explain(
    data=data.x_train,
    model=model_forward,
    schema=schema,
    method="pdp",
    nof_instances=5000,
)
report.show()
report.to_html("report_adult_income_pdp.html")  # open in browser
```

    [effector] global effects reproduce 72.5% of the model's variance; with subregions, 86.0%
    
    PDP report — target: income>50K
    ============================================================
    explained variance: global effects (GAM) 72.5%
      + split education-num (on age, capital-gain, marital-status) → 80.4% (+7.9 pts, heter 0.069→0.040)
      + split capital-gain (on age, education-num, marital-status) → 83.2% (+2.9 pts, heter 0.254→0.162)
      + split capital-loss (on age, capital-gain, marital-status) → 85.0% (+1.7 pts, heter 0.108→0.084)
      + split hours-per-week (on age, capital-gain, marital-status) → 86.0% (+1.0 pts, heter 0.064→0.045)
      rejected: age (on hours-per-week, marital-status) — +0.7 pts, below the 1.0-pt threshold
      rejected: relationship (on age, marital-status) — +0.3 pts, below the 1.0-pt threshold
    ------------------------------------------------------------
    feature                   importance     heter  #regions
    ------------------------------------------------------------
    capital-gain                  0.1096    0.1621         7
    age                           0.0709    0.0862         1
    education-num                 0.0699    0.0404         7
    capital-loss                  0.0659    0.0836         7
    relationship                  0.0525    0.0561         1
    hours-per-week                0.0351    0.0448         7
    ============================================================
    the plotted features carry 75% of the total importance mass
    
    
    Feature 8 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    capital-gain 🔹 [id: 0 | heter: 0.25 | inst: 5000 | w: 1.00]
        marital-status = 1.00 🔹 [id: 1 | heter: 0.23 | inst: 2319 | w: 0.46]
            education-num = 13.00 🔹 [id: 2 | heter: 0.17 | inst: 436 | w: 0.09]
            education-num ∈ {1.00, 2.00, 3.00, …} (15 levels) 🔹 [id: 3 | heter: 0.22 | inst: 1883 | w: 0.38]
        marital-status ∈ {0.00, 2.00, 3.00, …} (6 levels) 🔹 [id: 4 | heter: 0.21 | inst: 2681 | w: 0.54]
            age < 20.65 🔹 [id: 5 | heter: 0.08 | inst: 304 | w: 0.06]
            age ≥ 20.65 🔹 [id: 6 | heter: 0.12 | inst: 2377 | w: 0.48]
    --------------------------------------------------
    Feature 8 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.25
        Level 1🔹heter: 0.22 | 🔻0.04 (14.10%)
            Level 2🔹heter: 0.16 | 🔻0.06 (25.85%)
    
    
    
    
    Feature 2 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    education-num 🔹 [id: 0 | heter: 0.07 | inst: 5000 | w: 1.00]
        marital-status = 1.00 🔹 [id: 1 | heter: 0.07 | inst: 2319 | w: 0.46]
            capital-gain < 4999.95 🔹 [id: 2 | heter: 0.06 | inst: 2115 | w: 0.42]
            capital-gain ≥ 4999.95 🔹 [id: 3 | heter: 0.01 | inst: 204 | w: 0.04]
        marital-status ∈ {0.00, 2.00, 3.00, …} (6 levels) 🔹 [id: 4 | heter: 0.04 | inst: 2681 | w: 0.54]
            age < 31.60 🔹 [id: 5 | heter: 0.01 | inst: 1319 | w: 0.26]
            age ≥ 31.60 🔹 [id: 6 | heter: 0.05 | inst: 1362 | w: 0.27]
    --------------------------------------------------
    Feature 2 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.07
        Level 1🔹heter: 0.05 | 🔻0.02 (27.01%)
            Level 2🔹heter: 0.04 | 🔻0.01 (19.91%)
    
    
    
    
    Feature 9 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    capital-loss 🔹 [id: 0 | heter: 0.11 | inst: 5000 | w: 1.00]
        marital-status = 1.00 🔹 [id: 1 | heter: 0.10 | inst: 2319 | w: 0.46]
            capital-gain < 4999.95 🔹 [id: 2 | heter: 0.10 | inst: 2115 | w: 0.42]
            capital-gain ≥ 4999.95 🔹 [id: 3 | heter: 0.05 | inst: 204 | w: 0.04]
        marital-status ∈ {0.00, 2.00, 3.00, …} (6 levels) 🔹 [id: 4 | heter: 0.10 | inst: 2681 | w: 0.54]
            age < 24.30 🔹 [id: 5 | heter: 0.05 | inst: 733 | w: 0.15]
            age ≥ 24.30 🔹 [id: 6 | heter: 0.08 | inst: 1948 | w: 0.39]
    --------------------------------------------------
    Feature 9 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.11
        Level 1🔹heter: 0.10 | 🔻0.01 (7.28%)
            Level 2🔹heter: 0.08 | 🔻0.02 (16.62%)
    
    
    
    
    Feature 10 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    hours-per-week 🔹 [id: 0 | heter: 0.06 | inst: 5000 | w: 1.00]
        marital-status = 1.00 🔹 [id: 1 | heter: 0.07 | inst: 2319 | w: 0.46]
            capital-gain < 4999.95 🔹 [id: 2 | heter: 0.07 | inst: 2115 | w: 0.42]
            capital-gain ≥ 4999.95 🔹 [id: 3 | heter: 0.01 | inst: 204 | w: 0.04]
        marital-status ∈ {0.00, 2.00, 3.00, …} (6 levels) 🔹 [id: 4 | heter: 0.04 | inst: 2681 | w: 0.54]
            age < 24.30 🔹 [id: 5 | heter: 0.00 | inst: 733 | w: 0.15]
            age ≥ 24.30 🔹 [id: 6 | heter: 0.04 | inst: 1948 | w: 0.39]
    --------------------------------------------------
    Feature 10 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.06
        Level 1🔹heter: 0.05 | 🔻0.01 (20.00%)
            Level 2🔹heter: 0.04 | 🔻0.01 (12.24%)
    
    


    /home/givasile/github/packages/effector/effector/visualization.py:202: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
      fig, ax = plt.subplots()


## The decision sequence

Unlike the single-interaction stories (medical costs, airfoil), this
probability surface needs a *chain*: four splits, each with a real marginal
gain, and each conditioning on the same social cast — `age`,
`marital-status`, `education-num`.


```python
pdp = effector.PDP(data.x_train, model_forward, schema=schema, nof_instances=5000)
chain = pdp.select_regions()
chain.show()
```

    GAM: R2 = 0.725
      + education-num regions (on age, capital-gain, marital-status) -> R2 = 0.804 (+7.9 pts)
      + capital-gain regions (on age, education-num, marital-status) -> R2 = 0.832 (+2.9 pts)
      + capital-loss regions (on age, capital-gain, marital-status) -> R2 = 0.850 (+1.7 pts)
      + hours-per-week regions (on age, capital-gain, marital-status) -> R2 = 0.860 (+1.0 pts)
      x age regions skipped (below_threshold, +0.7 pts)
      x relationship regions skipped (below_threshold, +0.3 pts)


## Look at the headline split: `capital-gain`

The global `capital-gain` curve has the dataset's famous cliff: predicted
probability jumps sharply once reported capital gains pass a few thousand
dollars (almost nobody with meaningful capital gains earned under $50K in
the 1994 census). The split shows the cliff is *conditional* — its height
depends on who is standing at it.


```python
parts = pdp.find_regions("capital-gain", finder="best")
parts.show()
```

    
    
    Feature 8 - Full partition tree:
    🌳 Full Tree Structure:
    ───────────────────────
    capital-gain 🔹 [id: 0 | heter: 0.25 | inst: 5000 | w: 1.00]
        marital-status = Married-civ-spouse 🔹 [id: 1 | heter: 0.23 | inst: 2319 | w: 0.46]
            education-num = 13.00 🔹 [id: 2 | heter: 0.17 | inst: 436 | w: 0.09]
            education-num ∈ {1.00, 2.00, 3.00, …} (15 levels) 🔹 [id: 3 | heter: 0.22 | inst: 1883 | w: 0.38]
        marital-status ∈ {Divorced, Married-spouse-absent, Never-married, …} (6 levels) 🔹 [id: 4 | heter: 0.21 | inst: 2681 | w: 0.54]
            age < 20.65 🔹 [id: 5 | heter: 0.08 | inst: 304 | w: 0.06]
            age ≥ 20.65 🔹 [id: 6 | heter: 0.12 | inst: 2377 | w: 0.48]
    --------------------------------------------------
    Feature 8 - Statistics per tree level:
    🌳 Tree Summary:
    ─────────────────
    Level 0🔹heter: 0.25
        Level 1🔹heter: 0.22 | 🔻0.04 (14.10%)
            Level 2🔹heter: 0.16 | 🔻0.06 (25.85%)
    
    



```python
for r in parts:
    if r.level == 1:
        parts.plot(r.idx, centering=True)
```


    
![png](07_adult_income_files/07_adult_income_14_0.png)
    



    
![png](07_adult_income_files/07_adult_income_14_1.png)
    



```python
effector.plot_triage(pdp, partitions={"capital-gain": parts})
```


    
![png](07_adult_income_files/07_adult_income_15_0.png)
    


## Conclusion

- Classification needs no special machinery: wrap `predict_proba` with
  `adapters.classifier_proba` and every verb — curves, regions, the ledger —
  works in probability units.
- The ledger read is qualitatively different from the single-interaction
  datasets: **72% → 86% takes four stacked splits**, because the model's
  probability surface interacts along several axes at once
  (`education-num`, `capital-gain`, `capital-loss`, `hours-per-week` — all
  conditioned on `age` / `marital-status` / `capital-gain`).
- For a model like this, the final CALM — a small collection of
  region-conditional curves — is a far more faithful mental model than any
  single set of global curves.

## Cross-method sanity check

The one-liner `effector.explain` with every engine this notebook's model
supports. A gradient-boosted tree is piecewise-constant, so derivative-scale
methods (RHALE, DerPDP) have no meaningful gradients to work with and are out
of scope here; PDP, ALE and SHAP-DP cover the output-scale reads. Where
methods disagree — ranking, accepted splits, R² — that is a property of the
data/model worth a closer look, not an error.



```python
# === cross-method sweep: effector.explain on every applicable engine ======
sweep_reports = {}
for _m in ["pdp", "ale", "shapdp"]:
    _kw = {"nof_instances": 300} if _m == "shapdp" else {"nof_instances": 5000}
    print(f"--- {_m} " + "-" * 50)
    sweep_reports[_m] = effector.explain(
        data.x_train, model_forward, method=_m, schema=schema, **_kw
    )
    sweep_reports[_m].to_html(f"report_adult_income_{_m}.html")

print()
print(f"{'method':<8} {'ranking (plotted)':<52} {'GAM R2':>8} {'final R2':>9}  splits")
for _m, _r in sweep_reports.items():
    _rank = " > ".join(fr.name for fr in _r.features)
    _ev = _r.explained_variance
    _sp = "; ".join(f"{s['name']} on {s['on']}" for s in _ev["stages"]) or "none"
    print(f"{_m:<8} {_rank:<52} {_ev['gam_r2']:>7.1%} {_ev['regional_r2']:>8.1%}  {_sp}")

```

    --- pdp --------------------------------------------------


    [effector] global effects reproduce 72.5% of the model's variance; with subregions, 86.0%


    /home/givasile/github/packages/effector/effector/visualization.py:202: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
      fig, ax = plt.subplots()


    --- ale --------------------------------------------------


    [effector] global effects reproduce 75.3% of the model's variance; with subregions, 82.9%


    --- shapdp --------------------------------------------------


    PermutationExplainer explainer:  21%|██        | 62/300 [00:00<?, ?it/s]

    PermutationExplainer explainer:  22%|██▏       | 65/300 [00:10<00:10, 21.67it/s]

    PermutationExplainer explainer:  23%|██▎       | 68/300 [00:10<00:14, 15.64it/s]

    PermutationExplainer explainer:  23%|██▎       | 70/300 [00:10<00:16, 13.54it/s]

    PermutationExplainer explainer:  24%|██▍       | 72/300 [00:10<00:16, 13.85it/s]

    PermutationExplainer explainer:  25%|██▍       | 74/300 [00:10<00:16, 13.78it/s]

    PermutationExplainer explainer:  25%|██▌       | 76/300 [00:10<00:16, 13.94it/s]

    PermutationExplainer explainer:  26%|██▌       | 78/300 [00:11<00:15, 13.96it/s]

    PermutationExplainer explainer:  27%|██▋       | 80/300 [00:11<00:15, 13.86it/s]

    PermutationExplainer explainer:  27%|██▋       | 82/300 [00:11<00:15, 14.01it/s]

    PermutationExplainer explainer:  28%|██▊       | 84/300 [00:11<00:15, 13.76it/s]

    PermutationExplainer explainer:  29%|██▊       | 86/300 [00:11<00:15, 13.54it/s]

    PermutationExplainer explainer:  29%|██▉       | 88/300 [00:11<00:16, 12.80it/s]

    PermutationExplainer explainer:  30%|███       | 90/300 [00:12<00:15, 13.32it/s]

    PermutationExplainer explainer:  31%|███       | 92/300 [00:12<00:15, 13.77it/s]

    PermutationExplainer explainer:  31%|███▏      | 94/300 [00:12<00:15, 13.72it/s]

    PermutationExplainer explainer:  32%|███▏      | 96/300 [00:12<00:15, 13.23it/s]

    PermutationExplainer explainer:  33%|███▎      | 98/300 [00:12<00:14, 13.57it/s]

    PermutationExplainer explainer:  33%|███▎      | 100/300 [00:12<00:14, 13.55it/s]

    PermutationExplainer explainer:  34%|███▍      | 102/300 [00:12<00:14, 13.23it/s]

    PermutationExplainer explainer:  35%|███▍      | 104/300 [00:13<00:14, 13.32it/s]

    PermutationExplainer explainer:  35%|███▌      | 106/300 [00:13<00:14, 13.81it/s]

    PermutationExplainer explainer:  36%|███▌      | 108/300 [00:13<00:13, 13.96it/s]

    PermutationExplainer explainer:  37%|███▋      | 110/300 [00:13<00:13, 14.10it/s]

    PermutationExplainer explainer:  37%|███▋      | 112/300 [00:13<00:13, 13.44it/s]

    PermutationExplainer explainer:  38%|███▊      | 114/300 [00:13<00:14, 13.11it/s]

    PermutationExplainer explainer:  39%|███▊      | 116/300 [00:13<00:14, 13.08it/s]

    PermutationExplainer explainer:  39%|███▉      | 118/300 [00:14<00:13, 13.36it/s]

    PermutationExplainer explainer:  40%|████      | 120/300 [00:14<00:13, 13.51it/s]

    PermutationExplainer explainer:  41%|████      | 122/300 [00:14<00:13, 13.29it/s]

    PermutationExplainer explainer:  41%|████▏     | 124/300 [00:14<00:14, 12.33it/s]

    PermutationExplainer explainer:  42%|████▏     | 126/300 [00:14<00:13, 12.79it/s]

    PermutationExplainer explainer:  43%|████▎     | 128/300 [00:14<00:13, 12.86it/s]

    PermutationExplainer explainer:  43%|████▎     | 130/300 [00:15<00:13, 12.42it/s]

    PermutationExplainer explainer:  44%|████▍     | 132/300 [00:15<00:13, 12.63it/s]

    PermutationExplainer explainer:  45%|████▍     | 134/300 [00:15<00:12, 13.22it/s]

    PermutationExplainer explainer:  45%|████▌     | 136/300 [00:15<00:12, 13.43it/s]

    PermutationExplainer explainer:  46%|████▌     | 138/300 [00:15<00:11, 13.59it/s]

    PermutationExplainer explainer:  47%|████▋     | 140/300 [00:15<00:12, 13.22it/s]

    PermutationExplainer explainer:  47%|████▋     | 142/300 [00:15<00:11, 13.29it/s]

    PermutationExplainer explainer:  48%|████▊     | 144/300 [00:16<00:12, 12.47it/s]

    PermutationExplainer explainer:  49%|████▊     | 146/300 [00:16<00:12, 12.64it/s]

    PermutationExplainer explainer:  49%|████▉     | 148/300 [00:16<00:11, 12.71it/s]

    PermutationExplainer explainer:  50%|█████     | 150/300 [00:16<00:11, 13.18it/s]

    PermutationExplainer explainer:  51%|█████     | 152/300 [00:16<00:11, 13.39it/s]

    PermutationExplainer explainer:  51%|█████▏    | 154/300 [00:16<00:10, 13.30it/s]

    PermutationExplainer explainer:  52%|█████▏    | 156/300 [00:17<00:11, 12.95it/s]

    PermutationExplainer explainer:  53%|█████▎    | 158/300 [00:17<00:10, 13.53it/s]

    PermutationExplainer explainer:  53%|█████▎    | 160/300 [00:17<00:10, 13.70it/s]

    PermutationExplainer explainer:  54%|█████▍    | 162/300 [00:17<00:09, 13.84it/s]

    PermutationExplainer explainer:  55%|█████▍    | 164/300 [00:17<00:09, 14.25it/s]

    PermutationExplainer explainer:  55%|█████▌    | 166/300 [00:17<00:10, 13.34it/s]

    PermutationExplainer explainer:  56%|█████▌    | 168/300 [00:17<00:09, 13.70it/s]

    PermutationExplainer explainer:  57%|█████▋    | 170/300 [00:18<00:09, 13.77it/s]

    PermutationExplainer explainer:  57%|█████▋    | 172/300 [00:18<00:09, 13.55it/s]

    PermutationExplainer explainer:  58%|█████▊    | 174/300 [00:18<00:09, 13.52it/s]

    PermutationExplainer explainer:  59%|█████▊    | 176/300 [00:18<00:08, 13.90it/s]

    PermutationExplainer explainer:  59%|█████▉    | 178/300 [00:18<00:08, 14.36it/s]

    PermutationExplainer explainer:  60%|██████    | 180/300 [00:18<00:08, 13.50it/s]

    PermutationExplainer explainer:  61%|██████    | 182/300 [00:18<00:08, 13.76it/s]

    PermutationExplainer explainer:  61%|██████▏   | 184/300 [00:19<00:08, 13.17it/s]

    PermutationExplainer explainer:  62%|██████▏   | 186/300 [00:19<00:09, 12.66it/s]

    PermutationExplainer explainer:  63%|██████▎   | 188/300 [00:19<00:08, 13.09it/s]

    PermutationExplainer explainer:  63%|██████▎   | 190/300 [00:19<00:08, 13.53it/s]

    PermutationExplainer explainer:  64%|██████▍   | 192/300 [00:19<00:07, 13.94it/s]

    PermutationExplainer explainer:  65%|██████▍   | 194/300 [00:19<00:08, 12.96it/s]

    PermutationExplainer explainer:  65%|██████▌   | 196/300 [00:19<00:07, 13.51it/s]

    PermutationExplainer explainer:  66%|██████▌   | 198/300 [00:20<00:07, 13.62it/s]

    PermutationExplainer explainer:  67%|██████▋   | 200/300 [00:20<00:07, 13.39it/s]

    PermutationExplainer explainer:  67%|██████▋   | 202/300 [00:20<00:07, 13.66it/s]

    PermutationExplainer explainer:  68%|██████▊   | 204/300 [00:20<00:07, 13.46it/s]

    PermutationExplainer explainer:  69%|██████▊   | 206/300 [00:20<00:07, 13.12it/s]

    PermutationExplainer explainer:  69%|██████▉   | 208/300 [00:20<00:07, 12.80it/s]

    PermutationExplainer explainer:  70%|███████   | 210/300 [00:21<00:06, 12.92it/s]

    PermutationExplainer explainer:  71%|███████   | 212/300 [00:21<00:06, 12.59it/s]

    PermutationExplainer explainer:  71%|███████▏  | 214/300 [00:21<00:06, 12.64it/s]

    PermutationExplainer explainer:  72%|███████▏  | 216/300 [00:21<00:06, 12.94it/s]

    PermutationExplainer explainer:  73%|███████▎  | 218/300 [00:21<00:06, 12.90it/s]

    PermutationExplainer explainer:  73%|███████▎  | 220/300 [00:21<00:06, 12.99it/s]

    PermutationExplainer explainer:  74%|███████▍  | 222/300 [00:21<00:05, 13.18it/s]

    PermutationExplainer explainer:  75%|███████▍  | 224/300 [00:22<00:05, 13.03it/s]

    PermutationExplainer explainer:  75%|███████▌  | 226/300 [00:22<00:05, 13.14it/s]

    PermutationExplainer explainer:  76%|███████▌  | 228/300 [00:22<00:05, 13.26it/s]

    PermutationExplainer explainer:  77%|███████▋  | 230/300 [00:22<00:05, 13.31it/s]

    PermutationExplainer explainer:  77%|███████▋  | 232/300 [00:22<00:05, 12.91it/s]

    PermutationExplainer explainer:  78%|███████▊  | 234/300 [00:22<00:05, 12.86it/s]

    PermutationExplainer explainer:  79%|███████▊  | 236/300 [00:23<00:04, 13.35it/s]

    PermutationExplainer explainer:  79%|███████▉  | 238/300 [00:23<00:04, 13.20it/s]

    PermutationExplainer explainer:  80%|████████  | 240/300 [00:23<00:04, 12.57it/s]

    PermutationExplainer explainer:  81%|████████  | 242/300 [00:23<00:04, 13.16it/s]

    PermutationExplainer explainer:  81%|████████▏ | 244/300 [00:23<00:04, 13.79it/s]

    PermutationExplainer explainer:  82%|████████▏ | 246/300 [00:23<00:03, 13.81it/s]

    PermutationExplainer explainer:  83%|████████▎ | 248/300 [00:23<00:03, 13.55it/s]

    PermutationExplainer explainer:  83%|████████▎ | 250/300 [00:24<00:03, 13.21it/s]

    PermutationExplainer explainer:  84%|████████▍ | 252/300 [00:24<00:03, 12.66it/s]

    PermutationExplainer explainer:  85%|████████▍ | 254/300 [00:24<00:03, 11.96it/s]

    PermutationExplainer explainer:  85%|████████▌ | 256/300 [00:24<00:03, 12.66it/s]

    PermutationExplainer explainer:  86%|████████▌ | 258/300 [00:24<00:03, 12.27it/s]

    PermutationExplainer explainer:  87%|████████▋ | 260/300 [00:24<00:03, 12.59it/s]

    PermutationExplainer explainer:  87%|████████▋ | 262/300 [00:25<00:03, 12.44it/s]

    PermutationExplainer explainer:  88%|████████▊ | 264/300 [00:25<00:02, 12.70it/s]

    PermutationExplainer explainer:  89%|████████▊ | 266/300 [00:25<00:02, 12.40it/s]

    PermutationExplainer explainer:  89%|████████▉ | 268/300 [00:25<00:02, 12.61it/s]

    PermutationExplainer explainer:  90%|█████████ | 270/300 [00:25<00:02, 12.53it/s]

    PermutationExplainer explainer:  91%|█████████ | 272/300 [00:25<00:02, 12.45it/s]

    PermutationExplainer explainer:  91%|█████████▏| 274/300 [00:26<00:02, 12.72it/s]

    PermutationExplainer explainer:  92%|█████████▏| 276/300 [00:26<00:01, 13.20it/s]

    PermutationExplainer explainer:  93%|█████████▎| 278/300 [00:26<00:01, 12.77it/s]

    PermutationExplainer explainer:  93%|█████████▎| 280/300 [00:26<00:01, 12.82it/s]

    PermutationExplainer explainer:  94%|█████████▍| 282/300 [00:26<00:01, 12.69it/s]

    PermutationExplainer explainer:  95%|█████████▍| 284/300 [00:26<00:01, 12.20it/s]

    PermutationExplainer explainer:  95%|█████████▌| 286/300 [00:26<00:01, 12.39it/s]

    PermutationExplainer explainer:  96%|█████████▌| 288/300 [00:27<00:00, 12.70it/s]

    PermutationExplainer explainer:  97%|█████████▋| 290/300 [00:27<00:00, 12.49it/s]

    PermutationExplainer explainer:  97%|█████████▋| 292/300 [00:27<00:00, 12.98it/s]

    PermutationExplainer explainer:  98%|█████████▊| 294/300 [00:27<00:00, 13.31it/s]

    PermutationExplainer explainer:  99%|█████████▊| 296/300 [00:27<00:00, 13.25it/s]

    PermutationExplainer explainer:  99%|█████████▉| 298/300 [00:27<00:00, 13.56it/s]

    PermutationExplainer explainer: 100%|██████████| 300/300 [00:28<00:00, 13.61it/s]

    PermutationExplainer explainer: 301it [00:28,  8.51it/s]                         

    


    [effector] global effects reproduce 64.2% of the model's variance; with subregions, 73.5%


    
    method   ranking (plotted)                                      GAM R2  final R2  splits
    pdp      capital-gain > age > education-num > capital-loss > relationship > hours-per-week   72.5%    86.0%  education-num on age, capital-gain, marital-status; capital-gain on age, education-num, marital-status; capital-loss on age, capital-gain, marital-status; hours-per-week on age, capital-gain, marital-status
    ale      capital-gain > education-num > capital-loss > relationship > marital-status > hours-per-week   75.3%    82.9%  education-num on age, capital-gain, marital-status; hours-per-week on age, capital-gain, relationship
    shapdp   capital-gain > education-num > relationship > marital-status > age > capital-loss   64.2%    73.5%  capital-gain on marital-status, workclass; education-num on marital-status; capital-loss on education-num, relationship

