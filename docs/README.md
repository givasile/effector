# Documentation

---

## TODO before publishing

### Copy changelog

```bash
cp ../CHANGELOG.md docs/changelog.md
```

---

### Update the notebooks

#### (Optional) Remove the old notebooks

```bash
rm -rf docs/notebooks
mkdir docs/notebooks
```

#### Convert the notebooks to markdown

```bash
jupyter nbconvert --to markdown ./../notebooks/real-examples/* --output-dir docs/notebooks/real-examples/
jupyter nbconvert --to markdown ./../notebooks/synthetic-examples/* --output-dir docs/notebooks/synthetic-examples/
jupyter nbconvert --to markdown ./../notebooks/quickstart/* --output-dir docs/notebooks/quickstart/
jupyter nbconvert --to markdown ./../notebooks/guides/* --output-dir docs/notebooks/guides/
```
---

### Update the images on the `./static` folder

#### Create the folders if they don't exist

```bash
mkdir docs/static/quickstart/
mkdir docs/static/real-examples/
```

#### Copy the images

```bash
cp -r docs/notebooks/quickstart/simple_api_files/ docs/static/quickstart/
cp -r docs/notebooks/quickstart/flexible_api_files/ docs/static/quickstart/
cp -r docs/notebooks/quickstart/readme_example_files/ docs/static/quickstart/
cp -r docs/notebooks/real-examples/01_bike_sharing_dataset_files/ docs/static/real-examples/
```

---

## Run the documentation locally

```zsh
mkdocs serve
```



## Connection between files

`index.md` -> `/quickstart/readme_example.ipynb`
`docs/docs/quickstart/global_and_regional_effects.ipynb` -> `notebooks/real-examples/01_bike_sharing_dataset.ipynb`
