# TopoFM-Slice-v1

Run with:

```bash
python scripts/train.py --config configs/topofm_slice_v1.yaml
```

For offline Kaggle execution, attach a local clone of `facebookresearch/dinov2` and its pretrained checkpoint, then override:

```bash
python scripts/train.py \
  --config configs/topofm_slice_v1.yaml \
  --set model.dinov2.repository_path=/kaggle/input/[DINOV2_REPOSITORY_DIRECTORY] \
  --set model.dinov2.checkpoint=/kaggle/input/[DINOV2_CHECKPOINT_FILE]
```

The default config deliberately records the untested domain-generalization options as disabled. Do not rename the resulting experiment as the complete TopoFM-DG method.
