from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
import effector

X, y = make_classification(n_samples=1000)
m = GradientBoostingClassifier().fit(X, y)

regional_fe = effector.RegionalPDP(
    data=X,
    model=m.predict,
    nof_instances="all",
)
regional_fe.fit(
    features="all",
    candidate_conditioning_features="all",
)

partitioners = list(regional_fe.partitioners.values())
print(all(p is partitioners[0] for p in partitioners))
