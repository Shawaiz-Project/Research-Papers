# Reproducibility Protocol

1. Archive the raw dataset identity and directory listing.
2. Create `split_manifest.csv` once and never regenerate it while comparing methods.
3. Store SHA-256 hashes for every image.
4. Remove exact same-label duplicates and stop on cross-label hash conflicts.
5. Use patient-level grouping when patient identifiers are available. Otherwise label the evaluation as image-level.
6. Precompute topology with a configuration signature.
7. Fit descriptor standardization using training samples only.
8. Train the teacher using train/validation data only.
9. Select the student checkpoint using validation MCC or another preregistered metric.
10. Fit temperature and decision threshold on validation predictions only.
11. Evaluate the frozen test set once per finalized seed/model.
12. Report at least three seeds; five are preferred.
13. Save per-sample predictions and exact hashes for paired comparison.
14. Record package versions, GPU, parameter count, memory, and latency.

## Exact historical reproduction

The public project repository did not expose the original implementation or resolved training metadata. Exact historical reproduction therefore requires these archived artifacts:

```text
[ARCHIVED_RESOLVED_CONFIG_PATH]
[ARCHIVED_SPLIT_MANIFEST_PATH]
[ARCHIVED_TEACHER_CHECKPOINT_PATH]
[ARCHIVED_STUDENT_CHECKPOINT_PATH]
[ARCHIVED_TDA_CACHE_PATH]
```

Without those files, new runs are protocol-consistent reconstructions, not byte-identical reruns.
