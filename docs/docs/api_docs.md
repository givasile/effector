API DOCS:

- [Input layer — Schema & from_dataframe](./api_docs/api_ingestion.md)
    - [`effector.Schema`](./api_docs/api_ingestion.md/#effector.ingestion.Schema)
    - [`effector.from_dataframe`](./api_docs/api_ingestion.md/#effector.ingestion.from_dataframe)

- [Adapters — model wrappers, the explicit final pass](./api_docs/api_adapters.md)
    - [`effector.adapters.from_sklearn`](./api_docs/api_adapters.md/#effector.adapters.from_sklearn)
    - [`effector.adapters.classifier_proba`](./api_docs/api_adapters.md/#effector.adapters.classifier_proba)
    - [`effector.adapters.from_torch`](./api_docs/api_adapters.md/#effector.adapters.from_torch)
    - [`effector.adapters.check`](./api_docs/api_adapters.md/#effector.adapters.check)

- [Global effect](./api_docs/api_global.md):
    - [`effector.PDP`](./api_docs/api_global.md/#effector.global_effect_pdp.PDP)
    - [`effector.RHALE`](./api_docs/api_global.md/#effector.global_effect_ale.RHALE)
    - [`effector.ShapDP`](./api_docs/api_global.md/#effector.global_effect_shap.ShapDP)
    - [`effector.ALE`](./api_docs/api_global.md/#effector.global_effect_ale.ALE)
    - [`effector.DerPDP`](./api_docs/api_global.md/#effector.global_effect_pdp.DerPDP)

- [Regional effects — find_regions & Partition](./api_docs/api_partition.md)
    - [`effector.GlobalEffectBase.find_regions`](./api_docs/api_partition.md/#effector.global_effect.GlobalEffectBase.find_regions)
      — singular `feature=` → `Partition`, plural `features=` (list / `"all"` /
      `"heterogeneous"`) → `{name: Partition}`
    - [`effector.Partition`](./api_docs/api_partition.md/#effector.partition.Partition)
    - [`effector.Region`](./api_docs/api_partition.md/#effector.partition.Region)

- [Explained variance — select_regions & the CALM chain](./api_docs/api_calm.md)
    - [`effector.GlobalEffectBase.select_regions`](./api_docs/api_calm.md/#effector.global_effect.GlobalEffectBase.select_regions)
      — which found splits earn their keep, as a `CalmSequence`
    - [`effector.CalmSequence`](./api_docs/api_calm.md/#effector.calm.CalmSequence)
    - [`effector.CALM`](./api_docs/api_calm.md/#effector.calm.CALM)

- [Triage, comparison & theme](./api_docs/api_visualization.md)
    - [`effector.plot_triage`](./api_docs/api_visualization.md/#effector.visualization.plot_triage)
      — importance × heterogeneity plane, before/after arrows with `partitions=`
    - [`effector.compare`](./api_docs/api_visualization.md/#effector.visualization.compare)
      — overlay fitted engines on one feature
    - [`effector.FeatureEffect`](./api_docs/api_visualization.md/#effector.feature_effect.FeatureEffect)
      — the single-model comparison facade
    - [`effector.set_theme`](./api_docs/api_visualization.md/#effector.theme.set_theme)

- [One-click report — explain & Report](./api_docs/api_report.md)
    - [`effector.explain`](./api_docs/api_report.md/#effector.report.explain)
    - [`effector.GlobalEffectBase.explain`](./api_docs/api_report.md/#effector.global_effect.GlobalEffectBase.explain)
      — the one-liner on an already-constructed engine
    - [`effector.Report`](./api_docs/api_report.md/#effector.report.Report)


- [`effector.axis_partitioning`](./api_docs/api_axis_partitioning.md)
- [`effector.space_partitioning`](./api_docs/api_space_partitioning.md)
- [`effector.proposers`](./api_docs/api_proposers.md)
- [Extras — models, datasets, benchmarks](./api_docs/api_extras.md)
