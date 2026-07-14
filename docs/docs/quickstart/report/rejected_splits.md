---
title: The rejected splits
---

???+ success "Description"

    The third component of [effector's report](./../report.md): the
    **REJECTED SPLITS** table. Every split that was considered and refused,
    with the reason. The report shows its work.

???+ note "Reading time"

	Approx. 4' to read.

## What you see

In `report.show()`, right under the
[explained variance ledger](./explained_variance.md):

```text
  REJECTED SPLITS                                            min gain 1.0%
  ────────────────────────────────────────────────────────────────────────
    feature      split on                 solo     ΔR²    reason
    ──────────────────────────────────────────────────────────────────────
  ✗ yr           hr, workingday          +2.6%   -0.8%    redundant
  ✗ temp         hum, workingday         +1.6%   +0.3%    below threshold
  ✗ hum          hr, temp                +1.2%   +0.7%    below threshold
  ✗ mnth         hum, season, temp       +0.7%   +0.4%    below threshold
  ✗ workingday   hr, yr                  +5.9%   -4.9%    redundant

    ✗ redundant: it would explain variance on its own (see solo),
      but the accepted splits already account for it.
```

On the HTML page the same rows appear dimmed inside the **decision sequence**
table, one `rejected · <feature> (on ...)` row each, with the candidate's
region count and its heterogeneity before → after.

The section exists only when something was rejected; a run where every
candidate is accepted (or none is proposed) prints nothing here.

## Why show refusals at all

Because "why is `workingday` not split?" deserves a row, not a shrug. The
selection is greedy and it stops; everything it left behind is listed with
the number that condemned it. Nothing is silently dropped, so the report can
be *audited*, not just believed.

## The two reasons

Both columns come from the same round: **the moment the selection stopped**,
every remaining candidate was measured once more, on top of everything
already accepted.

- **`below threshold`**: the split still adds real variance (`ΔR²` positive),
  just less than `min gain` (the header repeats the cut: `min gain 1.0%`,
  from `min_r2_gain=0.01`). Lower the threshold and these are the rows that
  would enter the ledger next.
- **`redundant`**: the split adds nothing, or even *hurts* (`ΔR²` zero or
  negative), because the accepted splits already explain its variance.

???+ warning "Read `solo` against `ΔR²`"

    `workingday` would have been worth **+5.9% on its own**: the second
    strongest split in the model. But once `hr` is split (and `hr`'s split
    already conditions on `workingday`), it adds **−4.9%**. It is not
    useless; it is **redundant**. The `solo` column is what separates the
    two verdicts, and it is why the column exists.

???+ question "How can ΔR² be negative?"

    Two splits explaining the *same* variance double count it: the summed
    surrogate moves twice for one underlying pattern, and its R² drops. A
    negative `ΔR²` is the selection catching that overlap, not a numerical
    accident.

## Rejected does not mean fake

Every rejected split is a *real* split: `find_regions` proposed it because it
genuinely reduces its feature's heterogeneity. The lesson of this table is
that **a heterogeneity drop and an explained-variance gain are different
currencies**: a split can resolve real spread inside its own feature while
adding nothing to what the accepted splits already explain of the model.

In [the regional analysis](./regional_analysis.md) section of the HTML page,
a rejected feature stays **global**, with a note pointing back at this table;
its found partition is not drawn as if it had been accepted.

---

## Where to next

- [The ranked features](./ranked_features.md): the next component
- [effector's report](./../report.md): back to the guide's map
- [The explained variance ledger](./explained_variance.md): the accepted rows
