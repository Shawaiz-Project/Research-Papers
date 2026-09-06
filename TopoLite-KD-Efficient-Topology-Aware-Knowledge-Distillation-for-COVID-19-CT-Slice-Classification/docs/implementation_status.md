# Implementation Status

See the repository-root `CODEBASE_AUDIT.md` for the complete provenance matrix.

The package distinguishes three things:

1. **Executable implementation** — source code that can be trained and evaluated.
2. **Historical record** — author-supplied metrics retained under `research_records/`.
3. **Exact reproduction artifact** — a run that also has the original manifest, checkpoints, resolved configuration, topology cache signature, and environment capture.

A runnable reconstruction is not automatically an exact reproduction. Generated results always come from model predictions; historical metrics are never copied into a run directory.
