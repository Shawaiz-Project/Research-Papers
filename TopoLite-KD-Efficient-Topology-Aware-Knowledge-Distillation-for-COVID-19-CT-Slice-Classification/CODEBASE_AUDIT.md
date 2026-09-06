# Codebase Audit and Provenance

## Confidence labels

- **Confirmed architecture**: directly preserved from the project’s authoritative research specification.
- **Reconstruction default**: a transparent, executable choice used because the exact archived value was unavailable.
- **Optional extension**: implemented for controlled experimentation but not claimed as part of the successful main run.
- **Not implemented/tested historically**: intentionally disabled and never presented as completed evidence.

## Technique matrix

| Technique | Implementation status | Historical status | Main files |
|---|---|---|---|
| Standard Custom CNN + TDA | Complete | Confirmed baseline family | `models/topolite.py`, `baseline_cnn_tda.yaml` |
| TopoLite-KD | Complete, paper-aligned reconstruction | Confirmed successful main method | `models/topolite.py`, `topolite_kd.yaml` |
| TopoLite-MSF-KD | Complete | Confirmed negative/extended experiment | `models/msf.py`, `topolite_msf_kd.yaml` |
| TopoLite-FKD-SAM | Complete framework | Confirmed attempted extension; exact archived weights unavailable | `topolite_fkd_sam.yaml`, `optim/sam.py` |
| TopoFM-Slice-v1 | Component-faithful reconstruction | Confirmed slice-level experiment | `models/topofm_slice.py`, `topofm_slice_v1.yaml` |
| A0–A10 ablations | Complete | Required ablation matrix | `configs/ablations/` |
| Patient-level 2.5D TopoFM-DG | Not claimed/disabled | Not tested in archived run | documented only |
| VREx/domain adversarial/soft masks | Not claimed/disabled | Not tested in archived run | `domain_generalization` flags in config |

## Confirmed TopoLite-KD implementation

- Input: one grayscale `1×224×224` CT slice.
- Visual channels: `24→32→48→96→160`.
- Blocks: depthwise-separable residual operations with GroupNorm and SiLU.
- Coordinate Attention: after visual stages 2 and 3.
- Visual embedding: 64 dimensions.
- Topology: cubical persistence on `64×64` images, sublevel and superlevel filtrations, H0 and H1.
- Descriptor: 134 dimensions, built as `4×(10 lifetime statistics + 4 persistence counts + 16 Betti values) + 14 binary component/hole counts`.
- TDA MLP: `134→128→64`.
- Fusion: per-feature reliability gate, weighted blend, and multiplicative interaction.
- Teacher: EfficientNet-B0.
- Main loss: `0.60 supervised BCE + 0.30 response KD + 0.05 visual auxiliary BCE + 0.05 topology auxiliary BCE` with `T=3`.
- Current trainable parameter count: 196,773, close to the archived approximate count without non-functional padding.

## Reconstruction defaults that are not exact historical claims

The exact archived optimizer schedule, every augmentation value, some FKD-SAM weights, and the internal dimensions of TopoFM-Slice-v1 were unavailable. The repository provides explicit, editable defaults and labels them as reconstruction defaults. Replace them with the archived `resolved_config.yaml` before making an exact-reproduction claim.

## Verification performed in this package build

- Python source compilation completed.
- MkDocs documentation built successfully with strict mode.
- All six unit tests passed, including GUDHI-backed descriptor and 144-token shape tests.
- CPU forward passes passed for TopoLite-KD, the baseline, FKD-SAM, visual/TDA-only models, the full two-layer MSF model, and the ablation variants.
- The full MSF reconstruction has 1,131,654 trainable parameters, close to the historical approximately 1.12M record.
- A 24-image synthetic integration dataset was used only for software verification: manifest creation, hashing, stratified splitting, cubical persistence caching, train-only descriptor standardization, dataloading, one-epoch training, validation calibration, frozen-test evaluation, all quantitative figures/CSV exports, Grad-CAM, and run-artifact auditing completed successfully. Synthetic verification metrics are not included in the repository or presented as research results.
- GPU execution and exact dataset/checkpoint reproduction were not available in the packaging environment.
- DINOv2 weights were not downloaded in the packaging environment. The adapter uses the official torch.hub model or a local clone/checkpoint and fails explicitly if neither is accessible.
