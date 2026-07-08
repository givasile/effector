"""Toy interaction example for the categorical (nominal) methods — LINEAR form.

Send it cell-by-cell to an inferior Python process (mark region, C-c C-c).
Cells are delimited by `# %%`. Run the SETUP + MODEL + DATA cells first; after
that every cell is independent and reuses the module-level state.

Story
-----
  gender in {male, female, non-binary}  -> NOMINAL   (codes 0,1,2)
  experience in [0, 40] years           -> continuous
  income (K) = 20 + experience + 1{experience<=20} * bias(gender)
  bias = {male:+5, female:-5, non-binary:0}

Everyone starts ~20K, +1K per year; a +/-5K gender bias exists only in the first
working half (<=20 yrs) and vanishes after. The model we EXPLAIN is the
deterministic mean function (effector queries it at counterfactual points); the
Gaussian noise lives only in the data `y`, for a possible trained surrogate later.

GT (balanced genders, experience ~ U[0,40]): centered global gender effect
[+2.5,-2.5,0]; regional-on-gender splits at experience~20 -> [+5,-5,0] for the
young half, [0,0,0] for the senior half.
"""

# %% setup ------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")              # headless-safe; switch to "QtAgg" for interactive cells
import matplotlib.pyplot as plt
plt.ion()                          # non-blocking windows; plot() pops and returns
import effector

GENDERS = ["male", "female", "non-binary"]   # codes 0, 1, 2
BIAS = np.array([+5.0, -5.0, 0.0])           # K, applies only while experience <= 20
NOISE_STD = 8.0                              # K, "quite high" -- data only
GENDER, EXPERIENCE = 0, 1
LEVELS = np.array([0.0, 1.0, 2.0])
fmt = lambda v: dict(zip(GENDERS, np.round(np.asarray(v, float), 2)))

# %% model + jac (the black box we explain) ---------------------------------
def model(X):
    """Deterministic mean function, in K."""
    g = X[:, GENDER].astype(int)
    exp = X[:, EXPERIENCE]
    return 20.0 + exp + (exp <= 20).astype(float) * BIAS[g]

def jac(X):
    """d/d(experience)=1 a.e. (the gender step at 20 is a discontinuity, invisible
    to the gradient); d/d(gender)=0. This is why RHALE is blind to the step."""
    out = np.zeros_like(X, dtype=float)
    out[:, EXPERIENCE] = 1.0
    return out

# %% data + schema ----------------------------------------------------------
rng = np.random.RandomState(0)
g = rng.randint(0, 3, 6000)                     # balanced genders
exp = rng.uniform(0, 40, 6000)
X = np.column_stack([g.astype(float), exp])
y = model(X) + rng.normal(0, NOISE_STD, 6000)   # data only (surrogate later)

schema = {
    "feature_names": ["gender", "experience"],
    "feature_types": ["nominal", "continuous"],
    "category_names": [GENDERS, None],   # show male/female/non-binary, not 0/1/2
    "target_name": "income (K)",
}

# %% ground truth -----------------------------------------------------------
mu = np.array([np.mean(20 + exp + (exp <= 20) * BIAS[k]) for k in range(3)])
w = np.array([np.mean(g == k) for k in range(3)])
mu_c = mu - np.sum(w * mu)                       # zero_integral, frequency-weighted
print("level freqs w     :", np.round(w, 3))
print("PDP uncentered mu :", fmt(mu))
print("PDP centered (0I) :", fmt(mu_c))

# %% GLOBAL — PDP on gender (nominal) ---------------------------------------
pdp = effector.PDP(X, model, schema=schema)
pdp.fit(GENDER, centering="zero_integral")
print("PDP  @levels:", fmt(pdp.eval(GENDER, LEVELS, centering="zero_integral")), "| GT:", fmt(mu_c))
pdp.plot(GENDER, centering="zero_integral", heterogeneity="ice")

# %% GLOBAL — ShapDP on gender ----------------------------------------------
shap = effector.ShapDP(X, model, schema=schema, nof_instances=1500)
shap.fit(GENDER, centering="zero_integral")
print("Shap @levels:", fmt(shap.eval(GENDER, LEVELS, centering="zero_integral")))
shap.plot(GENDER, centering="zero_integral")

# %% GLOBAL — ALE on gender (nominal, ascending order) ----------------------
ale = effector.ALE(X, model, schema=schema)
ale.fit(GENDER, centering="zero_integral")
print("ALE  @levels:", fmt(ale.eval(GENDER, LEVELS, centering="zero_integral")))
print("ALE  h(step-in):", np.round(ale._eval_unnorm(GENDER, LEVELS, heterogeneity=True)[1], 1))
ale.plot(GENDER, centering="zero_integral")

# %% GLOBAL — guards (capability matrix): both should raise ValueError -------
try:
    effector.RHALE(X, model, model_jac=jac, schema=schema).fit(GENDER)
    print("RHALE(nominal)  -> NO ERROR (unexpected!)")
except ValueError as e:
    print("RHALE(nominal)  -> ValueError OK:", str(e)[:70])

try:
    effector.DerPDP(X, model, model_jac=jac, schema=schema).fit(GENDER)
    print("DerPDP(nominal) -> NO ERROR (unexpected!)")
except ValueError as e:
    print("DerPDP(nominal) -> ValueError OK:", str(e)[:70])

# %% COMPARE — FeatureEffect facade on gender (overlays methods at the levels)
# nominal FOI: RHALE is auto-skipped (with a warning); PDP/ALE/ShapDP overlaid
fe = effector.FeatureEffect(X, model, model_jac=jac, schema=schema)
fe.plot(GENDER, methods=["PDP", "ALE", "ShapDP"], centering="zero_integral")

# %% IMPORTANCE + one-click report -----------------------------------------
# importance = dispersion of the mean effect (mu-twin of heterogeneity)
print("PDP importances:", fmt(pdp.importances()) if False else np.round(pdp.importances(), 2))
report = effector.explain(X, model, method="pdp", schema=schema, nof_instances="all")
report.show()

# %% REGIONAL — PDP on gender  (expect split on experience ~ 20) ------------
# Regional effects are now a query on the fitted global effect: find_regions -> Partition
rpdp = effector.PDP(X, model, schema=schema)
rpdp.fit(GENDER, centering="zero_integral")
part = rpdp.find_regions(GENDER, finder=effector.space_partitioning.Best(max_depth=2))
part.show()
part.plot(1, centering="zero_integral", heterogeneity="ice")   # young
part.plot(2, centering="zero_integral", heterogeneity="ice")   # senior

# %% REGIONAL — ALE on gender  (same split) --------------------------------
rale = effector.ALE(X, model, schema=schema)
rale.fit(GENDER, centering="zero_integral")
rale.find_regions(GENDER, finder=effector.space_partitioning.Best(max_depth=2)).show()

# %% REGIONAL — ShapDP on gender  (same split) -----------------------------
rshap = effector.ShapDP(X, model, schema=schema, nof_instances=1500)
rshap.fit(GENDER, centering="zero_integral")
rshap.find_regions(GENDER, finder=effector.space_partitioning.Best(max_depth=2)).show()

# %% REGIONAL — RHALE on gender => RHALE rejects nominal at fit (capability guard)
rrhale = effector.RHALE(X, model, model_jac=jac, schema=schema)
try:
    rrhale.fit(GENDER)
    rrhale.find_regions(GENDER, finder=effector.space_partitioning.Best(max_depth=2))
    print("RHALE(nominal) -> NO ERROR (unexpected!)")
except ValueError as e:
    print("RHALE(nominal) -> ValueError OK:", str(e)[:70])

# %% REVERSE — Regional PDP on experience  (expect split on gender) ---------
rpdp_x = effector.PDP(X, model, schema=schema)
rpdp_x.fit(EXPERIENCE, centering="zero_integral")
rpdp_x.find_regions(EXPERIENCE, finder=effector.space_partitioning.Best(max_depth=2)).show()

# %% REVERSE — Regional ALE on experience  (splits on gender: sees the step)
rale_x = effector.ALE(X, model, schema=schema)
rale_x.fit(EXPERIENCE, centering="zero_integral")
rale_x.find_regions(EXPERIENCE, finder=effector.space_partitioning.Best(max_depth=2)).show()

# %% REVERSE — Regional RHALE on experience  (NO split: gradient blind to step)
rrhale_x = effector.RHALE(X, model, model_jac=jac, schema=schema)
rrhale_x.fit(EXPERIENCE)
rrhale_x.find_regions(EXPERIENCE, finder=effector.space_partitioning.Best(max_depth=2)).show()

# %% ==========================================================================
# %% ORDINAL experience — same model, `experience` as whole years (1-yr step)
# %% treated ORDINAL instead of continuous. Does the discrete-kernel path
# %% reproduce the continuous story? (P2 investigation.)
# %%
# %% Verdict from the cells below:
# %%   PDP, ALE -> reproduce continuous (same split on gender, matching effect;
# %%               the ~0.2K offset is only the frequency-weighted centering).
# %%   RHALE    -> DIVERGES, and ordinal is the MORE correct one: continuous
# %%               RHALE differentiates via the analytic Jacobian (d/dexp=1
# %%               everywhere) so it is BLIND to the gender step at exp=20;
# %%               ordinal RHALE takes discrete adjacent-level output differences
# %%               f(k+1)-f(k), which straddle the step and SEE the interaction
# %%               -> it splits on gender where continuous RHALE cannot.
# %% ==========================================================================
exp_int = np.rint(exp)                                  # whole years, 0..40
X_oi = np.column_stack([g.astype(float), exp_int])      # same data, int experience
schema_oi = {**schema, "feature_types": ["nominal", "ordinal"]}
GRID = np.array([5.0, 10.0, 20.0, 30.0, 40.0])          # observed integer years

def _split(part):                                       # compact split readout
    child = next((r.name for r in part if r.level == 1), None)   # first-level condition
    return f"{len(part)} nodes | " + (child if child else "NO SPLIT")

# %% ORDINAL exp — GLOBAL effect (continuous vs ordinal at the same years) ----
pdp_c = effector.PDP(X_oi, model, schema=schema);    pdp_c.fit(EXPERIENCE, centering="zero_integral")
pdp_o = effector.PDP(X_oi, model, schema=schema_oi); pdp_o.fit(EXPERIENCE, centering="zero_integral")
ale_c = effector.ALE(X_oi, model, schema=schema);    ale_c.fit(EXPERIENCE, centering="zero_integral")
ale_o = effector.ALE(X_oi, model, schema=schema_oi); ale_o.fit(EXPERIENCE, centering="zero_integral")
print("exp @[5,10,20,30,40]")
print("  PDP cont :", np.round(pdp_c.eval(EXPERIENCE, GRID, centering="zero_integral"), 2))
print("  PDP ordi :", np.round(pdp_o.eval(EXPERIENCE, GRID, centering="zero_integral"), 2))
print("  ALE cont :", np.round(ale_c.eval(EXPERIENCE, GRID, centering="zero_integral"), 2))
print("  ALE ordi :", np.round(ale_o.eval(EXPERIENCE, GRID, centering="zero_integral"), 2))
pdp_o.plot(EXPERIENCE, centering="zero_integral", heterogeneity="ice")   # one bar per year

# %% ORDINAL exp — REGIONAL (continuous vs ordinal; PDP/ALE agree, RHALE differs)
for name, ctor, needs_jac in [("PDP", effector.PDP, False),
                              ("ALE", effector.ALE, False),
                              ("RHALE", effector.RHALE, True)]:
    kw = dict(model_jac=jac) if needs_jac else {}
    finder = lambda: effector.space_partitioning.Best(max_depth=2)
    rc = ctor(X_oi, model, schema=schema, **kw);    rc.fit(EXPERIENCE)
    ro = ctor(X_oi, model, schema=schema_oi, **kw); ro.fit(EXPERIENCE)
    print(f"{name:6s} cont: {_split(rc.find_regions(EXPERIENCE, finder=finder()))}")
    print(f"{name:6s} ordi: {_split(ro.find_regions(EXPERIENCE, finder=finder()))}")
