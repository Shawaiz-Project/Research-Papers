# Contributing

1. Create a focused branch and keep model logic under `src/topokd/`.
2. Add or update a YAML configuration for every methodological change.
3. Never modify the frozen test manifest during model comparison.
4. Add tests for tensor shapes, metrics, and any new topology representation.
5. Run `pytest`, `python -m compileall -q src scripts`, and `mkdocs build --strict`.
6. Do not commit datasets, checkpoints, caches, generated predictions, or medical identifiers.
7. Describe whether a change is a confirmed method, reconstruction default, optional extension, or negative result.
