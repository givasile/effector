# mdale

# Methods supported

* Global  Effects:
  - ALE: [Accumulated Local Effects](https://arxiv.org/abs/1612.08468)
  - RHALE: ALE + heterogeneity
  - PDP: [Partial Dependence Plot](https://christophm.github.io/interpretable-ml-book/pdp.html)
  - ICE: [Individual Conditional Expectation](https://arxiv.org/abs/1309.6392)

* Regional Effects:
  - REPID: [Regional Effect Plots with implicit Interaction Detection](https://arxiv.org/abs/2202.07254)

* Feature Importance:
  - PFI: [Permutation Feature Importance](https://arxiv.org/abs/1801.01489)

* Feature Interaction:
  - TODO: add methods

# Repo Structure

* https://docs.python-guide.org/writing/structure/
* https://packaging.python.org/en/latest/tutorials/packaging-projects/

# Packaging

* toml: https://realpython.com/python-toml/

# Testing

with pytest https://docs.pytest.org/en/6.2.x/contents.html

* for running all tests `pytest tests` from inside the root directory `./`

# Documentation

with Sphinx: https://www.sphinx-doc.org/en/master/index.html

* for building html documentation `sphinx-build -b html docs/source/ docs/build/html` (or from inside `./docs` -> `make html`)  
* for building html documentation `sphinx-build -b latex docs/source/ docs/build/latex` (or from inside `./docs` -> `make latex`)

# CI-CD
* github actions


# TODO:

* remove example_models
  * keep only one/two models as synthetic examples
* create two examples with real datasets: 
  * Risk Factors for Cervical Cancer (Classification)
  * Bike Sharing Dataset (Regression)
* Refactor tests based on the two synthetic examples
* refactor `src`:
  * scan code to remove functions used only for research purposes
  * aggree on the final API
  * check on syntetic and real examples
* add new methods:
  * REPID
  * PFI
