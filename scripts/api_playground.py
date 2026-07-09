# =============================================================================
# effector — subregion-splitting playground
# =============================================================================
# A flat, sequential tour of ONE corner of the API: splitting a feature's
# effect into subregions (find_regions -> Partition) and how the conditional
# feature names land everywhere — rule strings in, region labels out.
# NO for-loops: run it block-by-block (IPython `%run -i`, VS Code cells).
#
# The synthetic model is built so the splits fall on NAMED things:
#   hr         (idx 0, continuous): the feature of interest. Its daily sine
#                                   curve flips SIGN with season and doubles
#                                   AMPLITUDE with temp -> 4 hidden regimes.
#   workingday (idx 1, nominal, no/yes): purely additive -> never splits hr.
#   temp       (idx 2, continuous): amplitude x2 when temp > 15  -> splitter.
#   season     (idx 3, nominal, winter/spring/summer/fall):
#              sign +1 in {summer, fall}, -1 in {winter, spring} -> splitter,
#              and a PAIR of levels, so the categorical proposers differ.
# =============================================================================

import numpy as np
import effector

# -----------------------------------------------------------------------------
# 0. Data + model — numpy-only, names live in the Schema
# -----------------------------------------------------------------------------
rng = np.random.default_rng(21)
N = 5_000

hr = rng.uniform(0, 24, N)
workingday = rng.integers(0, 2, N).astype(float)      # levels 0/1  -> no/yes
temp = rng.uniform(-5, 30, N)
season = rng.integers(0, 4, N).astype(float)          # 0/1/2/3 -> winter..fall

X = np.column_stack([hr, workingday, temp, season])

def predict(X):
    sign = np.where(np.isin(X[:, 3], (2.0, 3.0)), 1.0, -1.0)   # summer/fall vs rest
    amp = 1.0 + (X[:, 2] > 15)                                  # temp doubles it
    return sign * amp * 3 * np.sin(2 * np.pi * X[:, 0] / 24) + 2 * X[:, 1]

# Declare names/types/level-names ONCE. Everything downstream — rule strings,
# tree prints, region labels, plot titles — speaks these names.
schema = effector.Schema(
    feature_names=["hr", "workingday", "temp", "season"],
    feature_types=["continuous", "nominal", "continuous", "nominal"],
    category_names=[None, ["no", "yes"], None, ["winter", "spring", "summer", "fall"]],
    target_name="rentals",
)

pdp = effector.PDP(X, predict, schema=schema)


# -----------------------------------------------------------------------------
# 1. The itch: hr's global effect is a lie
# -----------------------------------------------------------------------------
# The mean effect of hr is ~flat (the +/- regimes cancel), but the spread is
# enormous. heter_score is the scalar the whole splitting story minimizes.
xs = np.linspace(0, 24, 100)
pdp.plot(feature=0, heterogeneity=True)          # flat mean, huge band
h_full = pdp.heter_score(0)
print("heter_score(hr) global:", round(h_full, 3))


# -----------------------------------------------------------------------------
# 2. Conditioning by hand — rule STRINGS, in feature/level names
# -----------------------------------------------------------------------------
# Every query (eval / heter_score / importance / plot) takes rule= — a string
# parsed against the schema. Categorical clauses use the LEVEL NAMES, not codes.
h_hot = pdp.heter_score(0, rule="temp > 15")            # RISES (~17): hot = double
h_summerish = pdp.heter_score(0, rule="season in {summer, fall}")   # amplitude, both
h_pure = pdp.heter_score(0, rule="season in {summer, fall} and temp > 15")  # signs
print("heter | temp > 15               :", round(h_hot, 3))
print("heter | season in {summer,fall} :", round(h_summerish, 3))
print("heter | both                    :", round(h_pure, 3), " <- a clean regime")

# same sugar on eval; workingday is additive, so conditioning on it does nothing
y_yes = pdp.eval(0, xs, rule="workingday == 'yes'")
y_no = pdp.eval(0, xs, rule="workingday == 'no'")
print("workingday is inert:", np.allclose(y_yes, y_no, atol=0.2))

# a rule is also a first-class value if you'd rather not write strings
r = effector.Rule.parse(
    "season == winter and temp <= 15",
    feature_names=pdp.feature_names,
    feature_types=pdp.feature_types,
    category_names=pdp.feature_metadata.category_names,
)
r                                   # Rule(season = 3.0 and temp ≤ 15.0)-ish repr
r.contains(X).sum()                 # rules apply themselves to raw numpy
print("formatted back with names:", r.format(
    pdp.feature_names, category_names=pdp.feature_metadata.category_names))


# -----------------------------------------------------------------------------
# 3. find_regions — the search that does step 2 for you
# -----------------------------------------------------------------------------
# A pure QUERY on the fitted effect (nothing stored on pdp): it searches for
# rules that drop heter_score and returns a Partition VALUE.
partition = pdp.find_regions(feature=0)

partition.show()          # tree print: every node's condition in real names,
                          # heter + instance count + weight, and per-level drop

len(partition)            # regions incl. root
partition.leaves          # the terminal regions

# a Region is a plain frozen record; its identity is its Rule (rule-primary).
# NB: region.name / the Rule repr are raw (codes, x_i); label() below is the
# schema-aware renderer — use that for anything human-facing.
root = partition[0]
leaf = partition.leaves[0]
print("root:", partition.label(0), "| heter:", round(root.heterogeneity, 3))
print("leaf:", partition.label(leaf.idx), "| heter:", round(leaf.heterogeneity, 3),
      "| n:", leaf.nof_instances, "| rule:", leaf.rule)


# -----------------------------------------------------------------------------
# 4. Where the names land — labels, masks, region-level eval/plot
# -----------------------------------------------------------------------------
# label(idx): "<feature> | <conditions>" with level names substituted, e.g.
#   "hr | season = winter and temp ≤ 15.0"   — never "x_3 = 0.0".
partition.label(1)
partition.label(leaf.idx)

sub_mask = partition.mask(leaf.idx)     # boolean (N,), a COPY — safe to mutate
sub_mask.sum()

# the partition is BOUND to pdp: eval/plot on a region index re-query the
# effect with that region's mask; the plot title carries the region label.
y_leaf = partition.eval(leaf.idx, xs)
partition.plot(leaf.idx)                 # a clean sine — regime isolated
partition.plot(0)                        # root = the muddled global, for contrast


# -----------------------------------------------------------------------------
# 5. show_axes — the grid view (works when leaves form an axis/grid)
# -----------------------------------------------------------------------------
# With the default one_vs_rest proposer the tree may be lopsided; the SUBSETS
# proposer can put {winter, spring} vs {summer, fall} in ONE split, giving a
# clean season x temp grid. Finder knobs travel as a finder INSTANCE:
finder = effector.space_partitioning.Best(categorical_proposer="subsets")
p_grid = pdp.find_regions(0, finder=finder)
p_grid.show()             # note the level-SET labels: "season ∈ {winter, spring}"
p_grid.show_axes()        # 1 axis -> partition print; 2 axes -> a labeled grid


# -----------------------------------------------------------------------------
# 6. The proposer menu — how categorical/continuous candidates are enumerated
# -----------------------------------------------------------------------------
# categorical_proposer: "one_vs_rest" (default) | "subsets" | "ordered" | "multiway"
# continuous_proposer:  "threshold" (default)  | "quantiles"
# ...or any instance with .propose(ctx, foc), e.g. effector.proposers.CategoricalOrdered(order=[...])

# multiway: one k-way candidate, one child per level -> four "season = <name>" children
p_multi = pdp.find_regions(0, finder=effector.space_partitioning.Best(
    categorical_proposer="multiway", max_depth=1))
p_multi.show()

# quantiles: k-way continuous splits at marginal quantiles (instead of binary thresholds)
p_quant = pdp.find_regions(0, finder=effector.space_partitioning.Best(
    continuous_proposer="quantiles", categorical_proposer="subsets"))
p_quant.show()

# other search knobs live on the same instance:
#   max_depth, min_samples_leaf, min_heterogeneity_decrease_pcg,
#   numerical_features_grid_size, finder="best_level_wise" (level-wise variant)
# and candidate_conditioning_features restricts WHO may split:
p_temp_only = pdp.find_regions(0, candidate_conditioning_features=[2])
p_temp_only.show()        # temp alone can't cut the sign-flip heterogeneity, so
                          # NO split passes the drop threshold -> just the root
                          # (the search declines rather than returning junk)


# -----------------------------------------------------------------------------
# 7. User-authored partitions — your rules, their names
# -----------------------------------------------------------------------------
# from_rules: hand rule strings (level names welcome); they must partition the
# data (disjoint + covering — checked). Stats are stamped model-free.
p_mine = effector.Partition.from_rules(
    [
        "season in {winter, spring}",
        "season in {summer, fall} and temp <= 15",
        "season in {summer, fall} and temp > 15",
    ],
    effect=pdp,
    feature=0,
)
p_mine.show()
p_mine.plot(3)            # the summer/fall + hot regime, labeled from YOUR rule


# -----------------------------------------------------------------------------
# 8. Partition is a value — names survive serialization
# -----------------------------------------------------------------------------
d = p_grid.to_dict()                          # rules + feature names, NO masks/effect
restored = effector.Partition.from_dict(d)
restored.show()   # feature names survive; LEVEL names show as codes ("season ∈
                  # {0.00, 1.00}") — they live on the effect, not in the dict.
# bind re-attaches: re-derives every mask from its rule and VERIFIES it against
# the data (drift tripwire), and the level names come back:
restored.bind(pdp)
restored.show()                               # "season ∈ {winter, spring}" again
restored.plot(1)

print("\n=== playground loaded — poke at any object above ===")
