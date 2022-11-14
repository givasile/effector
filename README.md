# mdale
Multiscale Differential Aggregate Local Effects (MDALE)


# Repo's Structure
the repo follows the guidelines described in:

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
