# Contributing to effector

Thanks for wanting to make `effector` better! 🎉

Our workflow fits in two simple concept:

> **`main` is always shippable. For a commit to be accepted in main, it must be reviewed and pass tests.**
> by taging (`vX.Y.Z`) a commit on `main`, we automatically run another round of tests and, if they pass, publish to PyPI.

---

## The mental model

`main` is the **stable integration branch**; we always keep it publishing-ready. 
Therefore, we **never** commit to `main` directly. 
All changes come through a PR, which is reviewed and tested before merging.

Every change gets its own **small branch**: 
you spin one up for a single idea, do the work, get it merged, and throw the branch away. 
Next idea starts with a fresh new branch from an up-to-date `main`.
The concept is one idea = one branch = one PR. 
"one idea" is normally a change that you can describe in a single sentence without saying *"and"*.

A **release** is a separate, deliberate step: we tag a commit on `main`, and that tag is what
ships to PyPI and what people `pip install`. 
The idea is `main` moves often, but releases happen less frequently.


---

## Pull Requests

Use a **Pull Request (PR)** to join `main`. 

1. **The diff**: shows exactly what you changed, for anyone to read.
2. **The tests**: runs the the full CI suite against your change automatically.
3. **The review**:a maintainer looks it over and approves or asks for tweaks.

---

## The workflow, step by step

```bash
# 1. Start fresh from main
git checkout main && git pull

# 2. Branch for your one idea
git checkout -b fix/short-description

# 3. Do the work, commit as you go
git commit -am "wip"        # use as many commits as you like, they will be squashed later

# 4. Push your branch
git push -u origin fix/short-description

# 5. Open the PR
gh pr create --fill         # or use the GitHub website

# 6. Let CI run, address review, then it merges (squashed) and the branch is deleted
```

---

## Branch names

Name the branch after the *intent* of the change, in `kebab-case`:

| Prefix      | Use it for                          | Example                       |
|-------------|-------------------------------------|-------------------------------|
| `feat/`     | a new feature                       | `feat/regional-importance`    |
| `fix/`      | a bug fix                           | `fix/centering-default`       |
| `chore/`    | tooling / deps / CI                 | `chore/switch-to-uv`          |
| `docs/`     | documentation                       | `docs/contributing`           |
| `refactor/` | internal change, no behavior change | `refactor/split-partitioning` |

---

## Commits & the changelog

We **squash-merge**, so all the little commits on your branch collapse into a
single commit on `main`; the **PR title** becomes that commit. 
So, 

- make your PR title a good one, this is what goes into history.
- be more descriptive in **`CHANGELOG.md`**, that's where we keep the descriptive
  record of what changed and why, not in commit messages.

---

## Two protection layers

They guard two different actions:

- **Merging to `main`.** A PR can't merge until CI is green **and** a maintainer approved it.
- **Releasing to PyPI.** You can't publish to PyPI without a tag. If you add a tag, the publish workflow runs tests again on that commit. If they pass, it publishes to PyPI.

---

## Versioning & releases

We follow [Semantic Versioning](https://semver.org/) and release **early and often**:

- Bug fixes / docs / cleanups → a **patch** (`0.2.1`, `0.2.2`, …)
- New backward-compatible features → a **minor** (`0.3.0`, …)

A release is just a normal PR merged to `main`, followed by a `vX.Y.Z` tag.
The tag triggers the publish to PyPI automatically. 

---

## Running things locally

We use [uv](https://docs.astral.sh/uv/) to manage the environment. Install it once
([instructions](https://docs.astral.sh/uv/getting-started/installation/)), then:

```bash
uv sync          # create .venv and install all dev dependencies
make test        # run the fast test suite (the merge gate)
make test-all    # run the full suite, including slow tests
make format      # format the code (ruff)
make lint        # lint the code (ruff)
make docs-serve  # preview the docs locally
```

---

Questions? Open an [issue](https://github.com/givasile/effector/issues) or start a
draft PR and ask there.
