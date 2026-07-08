API DOCS:

- [Adapters — model wrappers, the explicit final pass](./api_docs/api_adapters.md)
    - [`effector.adapters.from_sklearn`](./api_docs/api_adapters.md/#effector.adapters.from_sklearn)
    - [`effector.adapters.classifier_proba`](./api_docs/api_adapters.md/#effector.adapters.classifier_proba)
    - [`effector.adapters.from_torch`](./api_docs/api_adapters.md/#effector.adapters.from_torch)
    - [`effector.adapters.check`](./api_docs/api_adapters.md/#effector.adapters.check)

- [Global effect](./api_docs/api_global.md):
    - [`effector.PDP`](./api_docs/api_global.md/#effector.global_effect_pdp.PDP)
    - [`effector.RHALE`](./api_docs/api_global.md/#effector.global_effect_ale.RHALE)
    - [`effector.ShapDP`](./api_docs/api_global.md/#effector.global_effect_shap.ShapDP)
    - [`effector.ALE`](./api_docs/api_global.md/#effector.global_effect_ale.RHALE)
    - [`effector.DerPDP`](./api_docs/api_global.md/#effector.global_effect_pdp.DerPDP)

- [Regional effects — find_regions & Partition](./api_docs/api_partition.md)
    - [`effector.GlobalEffectBase.find_regions`](./api_docs/api_partition.md/#effector.global_effect.GlobalEffectBase.find_regions)
      — singular `feature=` → `Partition`, plural `features=` (list / `"all"` /
      `"heterogeneous"`) → `{name: Partition}`
    - [`effector.Partition`](./api_docs/api_partition.md/#effector.partition.Partition)
    - [`effector.Region`](./api_docs/api_partition.md/#effector.partition.Region)

- [Triage & comparison](./api_docs/api_visualization.md)
    - [`effector.plot_triage`](./api_docs/api_visualization.md/#effector.visualization.plot_triage)
      — importance × heterogeneity plane, before/after arrows with `partitions=`
    - [`effector.compare`](./api_docs/api_visualization.md/#effector.visualization.compare)
      — overlay fitted engines on one feature

- [One-click report — explain & Report](./api_docs/api_report.md)
    - [`effector.explain`](./api_docs/api_report.md/#effector.report.explain)
    - [`effector.Report`](./api_docs/api_report.md/#effector.report.Report)


- [`effector.axis_partitioning`](./api_docs/api_axis_partitioning.md)
- [`effector.space_partitioning`](./api_docs/api_space_partitioning.md)
- [`effector.proposers`](./api_docs/api_proposers.md)
