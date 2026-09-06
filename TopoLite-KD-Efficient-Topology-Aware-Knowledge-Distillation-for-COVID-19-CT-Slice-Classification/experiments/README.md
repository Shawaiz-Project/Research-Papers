# Experiment Registry

Use this directory for immutable paper-level experiment records. Model code belongs under `src/topokd/`; generated artifacts belong under `results/`.

Recommended record:

```text
experiments/[EXPERIMENT_ID]/
├── hypothesis.md
├── config.yaml
├── seeds.txt
├── command.sh
├── manifest_sha256.txt
├── teacher_checkpoint_sha256.txt
├── environment.txt
└── artifact_checksums.sha256
```

Copy the exact resolved run config—not the editable source config—into a finalized experiment record.
