# Documentation

To update the docs with the latest tutorials:

```bash
jupyter nbconvert --to markdown ./../notebooks/real-examples/* --output-dir docs/Tutorials/real-examples/
jupyter nbconvert --to markdown ./../notebooks/synthetic-examples/* --output-dir docs/Tutorials/synthetic-examples/
jupyter nbconvert --to markdown ./../notebooks/getting-started/* --output-dir docs/Tutorials/getting-started/
```

To launch the documentation locally:

```zsh
mkdocs serve
```


