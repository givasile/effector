## Summary

Regional effects are found on a **global** effect object: `find_regions(feature)`
searches for heterogeneity-reducing subregions of a feature and returns a
`Partition` — a value object, not stored state (design contract R12).

1. create and `.fit()` a global effect method (`PDP`, `RHALE`, `ALE`, `ShapDP`, `DerPDP`)
2. `partition = effect.find_regions(feature)` to search for subregions
3. `partition.show()` to print the partition tree
4. `partition.plot(idx)` to plot the effect within region `idx`
5. `partition.eval(idx, xs)` / `partition.eval_heter(idx, xs)` to evaluate it

---

## Usage

```python
# set up the input
X = ...        # input data
predict = ...  # model to be explained
jacobian = ... # jacobian of the model (RHALE / DerPDP only)
```

1. **Create and fit a global effect method**:

    === "PDP"

        ```python
        effect = effector.PDP(data=X, model=predict)
        effect.fit(features=[0, 1])
        ```

    === "RHALE"

        ```python
        effect = effector.RHALE(data=X, model=predict, model_jac=jacobian)
        effect.fit(features=[0, 1])
        ```

    === "ShapDP"

        ```python
        effect = effector.ShapDP(data=X, model=predict, nof_instances=500)
        effect.fit(features=[0, 1])
        ```

    === "ALE"

        ```python
        effect = effector.ALE(data=X, model=predict)
        effect.fit(features=[0, 1])
        ```

    === "DerPDP"

        ```python
        effect = effector.DerPDP(data=X, model=predict, model_jac=jacobian)
        effect.fit(features=[0, 1])
        ```

2. **Search for subregions of a feature** (returns a `Partition`):

    `find_regions(feature, *, finder="best", candidate_conditioning_features="all")`

    ??? Tip "Customize the search"

        `finder` accepts a name (`"best"` / `"best_level_wise"`) or a configured
        partitioner instance:

        ```python
        finder = effector.space_partitioning.Best(
            min_heterogeneity_decrease_pcg=0.3,  # drop threshold (default: 0.1)
            max_depth=1,                         # max split levels (default: 2)
        )
        partition = effect.find_regions(0, finder=finder)
        ```

3. **Print the partition tree**:

    `partition.show()`

    ??? Example "Example Output"

        ```python
        Feature 3 - Full partition tree:
        🌳 Full Tree Structure:
        ───────────────────────
        hr 🔹 [id: 0 | heter: 0.43 | inst: 3476 | w: 1.00]
            workingday = 0.00 🔹 [id: 1 | heter: 0.36 | inst: 1129 | w: 0.32]
                temp ≤ 6.50 🔹 [id: 3 | heter: 0.17 | inst: 568 | w: 0.16]
                temp > 6.50 🔹 [id: 4 | heter: 0.21 | inst: 561 | w: 0.16]
            workingday ≠ 0.00 🔹 [id: 2 | heter: 0.28 | inst: 2347 | w: 0.68]
                temp ≤ 6.50 🔹 [id: 5 | heter: 0.19 | inst: 953 | w: 0.27]
                temp > 6.50 🔹 [id: 6 | heter: 0.20 | inst: 1394 | w: 0.40]
        --------------------------------------------------
        Feature 3 - Statistics per tree level:
        🌳 Tree Summary:
        ─────────────────
        Level 0🔹heter: 0.43
            Level 1🔹heter: 0.31 | 🔻0.12 (28.15%)
                Level 2🔹heter: 0.19 | 🔻0.11 (37.10%)
        ```

4. **Plot the effect within a region**:

    `partition.plot(idx)` — `idx` is a region id from the tree (`0` is the full data).

5. **Evaluate the effect / heterogeneity within a region**:

    ```python
    y = partition.eval(idx, xs)             # mean effect within region idx
    h = partition.eval_heter(idx, xs)       # heterogeneity within region idx
    ```

## API

### ::: effector.global_effect.GlobalEffectBase.find_regions
       options:
         show_root_heading: True
         show_symbol_type_toc: True

### ::: effector.partition.Partition
       options:
         show_root_heading: True
         show_symbol_type_toc: True
         members:
           - leaves
           - mask
           - label
           - show
           - eval
           - eval_heter
           - plot
           - to_dict

### ::: effector.partition.Region
       options:
         show_root_heading: True
         show_symbol_type_toc: True
