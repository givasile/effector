"""Side-by-side prototype: OLD node-object regional plots vs NEW masked-global
regional plots (PR feat/regional-masked-delegation), plus the binning_scope
global-vs-effective comparison for RHALE.

Run twice:
    PYTHONPATH=<old-worktree> python scripts/proto_masked_regional.py old <outdir>
    PYTHONPATH=<repo>         python scripts/proto_masked_regional.py new <outdir>

Part A: synthetic gated model (fast gate).
Part B: bike sharing + lightgbm (once, the expensive real-data look).
Part C (new only): RHALE binning_scope "global" vs "effective".
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import effector

MODE = sys.argv[1]  # "old" | "new"
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
print("effector from:", effector.__file__)


def save(ret, fname):
    if ret is None:
        return
    fig = ret[0]
    fig.suptitle(fname.replace(".png", f" [{MODE}]"), fontsize=9)
    fig.savefig(os.path.join(OUT, f"{MODE}_{fname}"), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Part A ----
def gated(x):
    y = np.zeros_like(x[:, 0])
    ind = np.logical_and(x[:, 1] > 0, x[:, 2] == 0)
    y[ind] = 5 * x[ind, 0]
    return y


def gated_jac(x):
    j = np.zeros_like(x)
    ind = np.logical_and(x[:, 1] > 0, x[:, 2] == 0)
    j[ind, 0] = 5
    return j


rng = np.random.default_rng(21)
N = 1000
X = np.stack(
    [
        rng.uniform(-1, 1, N),
        rng.uniform(-1, 1, N),
        rng.integers(0, 2, N).astype(float),
    ],
    axis=1,
)

# small noise keeps DP binning away from the single-bin/zero-variance corner,
# where ShapDP's spline goes nan (pre-existing kernel wart, logged separately)
phi = rng.normal(0, 0.05, X.shape)
ind = np.logical_and(X[:, 1] > 0, X[:, 2] == 0)
phi[ind, 0] += 5 * X[ind, 0]

REGIONALS = {
    "pdp": lambda: effector.RegionalPDP(X, gated, nof_instances="all"),
    "ale": lambda: effector.RegionalALE(X, gated, nof_instances="all"),
    "rhale": lambda: effector.RegionalRHALE(
        X, gated, gated_jac, nof_instances="all"
    ),
    "shapdp": lambda: effector.RegionalShapDP(
        X, gated, nof_instances="all", shap_values=phi
    ),
}

for name, make in REGIONALS.items():
    reg = make()
    reg.fit(0, space_partitioner=effector.space_partitioning.Best(max_depth=2))
    tree = reg.tree["feature_0"]
    print(f"[A:{name}] nodes={len(tree.nodes)}")
    for node_idx in range(1, min(len(tree.nodes), 5)):
        try:
            ret = reg.plot(feature=0, node_idx=node_idx, show_plot=False)
        except Exception as e:  # noqa: BLE001 — comparison probe, keep going
            print(f"  node {node_idx} CRASH: {type(e).__name__}: {e}")
            continue
        save(ret, f"A_{name}_node{node_idx}.png")

# ---------------------------------------------------------------- Part C ----
if MODE == "new":
    for scope in ["global", "effective"]:
        # condition on x0 itself via a correlated conditioning feature: build
        # data where x1 correlates with x0, so subregions restrict the x0 range
        rng2 = np.random.default_rng(7)
        Xc = np.stack(
            [rng2.uniform(-1, 1, N), np.zeros(N), rng2.uniform(-1, 1, N)], axis=1
        )
        Xc[:, 1] = np.clip(Xc[:, 0] + rng2.normal(0, 0.3, N), -1, 1)

        def corr_model(x):
            return x[:, 0] ** 2 * (x[:, 1] > 0) + x[:, 2]

        def corr_jac(x):
            j = np.zeros_like(x)
            j[:, 0] = 2 * x[:, 0] * (x[:, 1] > 0)
            j[:, 2] = 1.0
            return j

        reg = effector.RegionalRHALE(Xc, corr_model, corr_jac, nof_instances="all")
        reg.fit(0, binning_scope=scope)
        tree = reg.tree["feature_0"]
        print(f"[C:rhale:{scope}] nodes={len(tree.nodes)}")
        for node_idx in range(1, min(len(tree.nodes), 3)):
            ret = reg.plot(feature=0, node_idx=node_idx, show_plot=False)
            save(ret, f"C_rhale_scope-{scope}_node{node_idx}.png")

# ---------------------------------------------------------------- Part B ----
if os.environ.get("PROTO_REAL", "0") == "1":
    from lightgbm import LGBMRegressor

    bike = effector.datasets.BikeSharing(pcg_train=0.8)
    Xtr, ytr = bike.x_train, bike.y_train
    model_lgbm = LGBMRegressor(n_estimators=100, verbose=-1, random_state=21)
    model_lgbm.fit(Xtr, ytr)

    def bike_model(x):
        return model_lgbm.predict(x)

    schema = {
        "feature_names": bike.feature_names,
        "target_name": bike.target_name,
        "scale_x_list": [
            {"mean": mu, "std": std}
            for mu, std in zip(bike.x_train_mu, bike.x_train_std)
        ],
        "scale_y": {"mean": bike.y_train_mu, "std": bike.y_train_std},
    }
    HR = 3  # the classic bike-sharing FOI: hour

    for name, ctor in [
        ("pdp", effector.RegionalPDP),
        ("ale", effector.RegionalALE),
    ]:
        reg = ctor(Xtr, bike_model, nof_instances=5000, schema=schema)
        reg.fit(
            HR,
            space_partitioner=effector.space_partitioning.Best(max_depth=1),
            candidate_conditioning_features=[6, 8],  # workingday, temp
        )
        tree = reg.tree[f"feature_{HR}"]
        print(f"[B:{name}] nodes={len(tree.nodes)}")
        for node_idx in range(1, min(len(tree.nodes), 3)):
            ret = reg.plot(
                feature=HR,
                node_idx=node_idx,
                show_plot=False,
                scale_x_list=schema["scale_x_list"],
            )
            save(ret, f"B_bike_{name}_hr_node{node_idx}.png")

print("done ->", OUT)
