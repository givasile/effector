# Getting Started

Feature effect plots in single-line commands

```python
# for PDP
PDP(data=X, model=ml_model).plot(feature=0)

# for ALE
ALE(data=X, model=ml_model, model_jac=ml_model_jac).plot(feature=0)
```

--- 

Regional Effect plots in single-line commands

```python
# for RHALE
```

---








## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
