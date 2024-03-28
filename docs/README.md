# Documentation

To update the docs with the latest tutorials:

```bash
jupyter nbconvert --to markdown ./../real-examples/* --output-dir docs/Tutorials/real-examples/
jupyter nbconvert --to markdown ./../synthetic-examples/* --output-dir docs/Tutorials/synthetic-examples/
```

To launch the documentation locally:

```zsh
mkdocs serve
```


