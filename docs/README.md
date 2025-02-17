# Documentation

---

## Copy changelog

```bash
cp ../CHANGELOG.md docs/changelog.md
```


If you want to clean everything first:

```bash
rm -rf docs/notebooks
mkdir docs/notebooks
```

To update the docs with the latest tutorials:

```bash
jupyter nbconvert --to markdown ./../notebooks/real-examples/* --output-dir docs/notebooks/real-examples/
jupyter nbconvert --to markdown ./../notebooks/synthetic-examples/* --output-dir docs/notebooks/synthetic-examples/
jupyter nbconvert --to markdown ./../notebooks/quickstart/* --output-dir docs/notebooks/quickstart/
jupyter nbconvert --to markdown ./../notebooks/guides/* --output-dir docs/notebooks/guides/
```

Then copy some on the static folder:

First create dir, if it does not exist:

```bash
mkdir docs/static/quickstart/
mkdir docs/static/real-examples/
mkdir docs/static/quickstart/simple_api_files/
mkdir docs/static/quickstart/flexible_api_files/
mkdir docs/static/real-examples/01_bike_sharing_dataset_files/
```

Then copy the files:

```bash
cp docs/notebooks/quickstart/simple_api_files/* docs/static/quickstart/simple_api_files/
cp docs/notebooks/quickstart/flexible_api_files/* docs/static/quickstart/flexible_api_files/
cp docs/notebooks/real-examples/01_bike_sharing_dataset_files/* docs/static/real-examples/01_bike_sharing_dataset_files/
```

To launch the documentation locally:

```zsh
mkdocs serve
```


