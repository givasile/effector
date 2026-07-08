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
(`{v}` vs the explicit complement over the observed levels). Custom
proposers plug in via a finder's `proposer_factory` attribute
(`feature type -> proposer`).

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

### ::: effector.proposers.default_proposer
           options:
             show_root_heading: True
