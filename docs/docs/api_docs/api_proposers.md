# `effector.proposers`

## The proposer protocol

Inside the built-in finders, candidate enumeration is its own seam: a
**proposer** is any object exposing

```python
propose(ctx: SearchContext, foc: int) -> list[CandidateSplit]
```

where each `CandidateSplit` is an ordered tuple of disjoint, jointly-covering
`Condition`s on the conditioning feature `foc` — a split expressed as
*parent rule → child conditions*. Child `i` of a parent region is
`parent_rule.refine(conditions[i])` with mask
`parent_mask & conditions[i].contains(data)`; candidates are
parent-independent (a level-wise finder applies one candidate to every node
of a level) and k-way by construction (`len(conditions) >= 2`).

The defaults reproduce the classic search — `ContinuousThreshold` (binary
`x < t / x >= t` on an interior grid) and `CategoricalOneVsRest`
(`{v}` vs the explicit complement over the observed levels). The richer
built-ins are selected by name from a finder's constructor:

```python
effector.space_partitioning.Best(
    categorical_proposer="ordered",   # "one_vs_rest" | "subsets" | "ordered" | "multiway"
    continuous_proposer="quantiles",  # "threshold" | "quantiles"
)
```

Both kwargs also accept a proposer instance (for non-default parameters,
e.g. `CategoricalOrdered(order=[2.0, 0.0, 1.0])`), and any custom object
exposing `propose`. The raw seam underneath is the finder's
`proposer_factory` attribute (`feature type -> proposer`), which
`make_proposer_factory` builds from the two kwargs.

## API

### ::: effector.proposers.SearchContext
           options:
             show_root_heading: True
             show_symbol_type_toc: True

### ::: effector.proposers.CandidateSplit
           options:
             show_root_heading: True
             show_symbol_type_toc: True

### ::: effector.proposers.ContinuousThreshold
           options:
             show_root_heading: True
             show_symbol_type_toc: True
             members:
               - propose

### ::: effector.proposers.CategoricalOneVsRest
           options:
             show_root_heading: True
             show_symbol_type_toc: True
             members:
               - propose

### ::: effector.proposers.CategoricalSubsets
           options:
             show_root_heading: True
             show_symbol_type_toc: True
             members:
               - propose

### ::: effector.proposers.CategoricalOrdered
           options:
             show_root_heading: True
             show_symbol_type_toc: True
             members:
               - propose

### ::: effector.proposers.CategoricalMultiway
           options:
             show_root_heading: True
             show_symbol_type_toc: True
             members:
               - propose

### ::: effector.proposers.ContinuousQuantiles
           options:
             show_root_heading: True
             show_symbol_type_toc: True
             members:
               - propose

### ::: effector.proposers.default_proposer
           options:
             show_root_heading: True

### ::: effector.proposers.make_proposer_factory
           options:
             show_root_heading: True
