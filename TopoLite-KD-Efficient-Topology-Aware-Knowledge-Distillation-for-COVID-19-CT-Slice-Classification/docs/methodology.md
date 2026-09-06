# Methodology

## TopoLite-KD

A normalized grayscale CT slice `x` is encoded by a custom lightweight visual branch:

```text
1×224×224
  → Conv 3×3/s2 + GroupNorm + SiLU (24 channels)
  → depthwise-separable residual stages: 32 → 48 → 96 → 160
  → Coordinate Attention after stages 2 and 3
  → global average pooling
  → projection to v ∈ R64
```

The topology pipeline resizes the unaugmented grayscale slice to `64×64`, constructs sublevel `I` and superlevel `1-I` cubical filtrations, and computes H0/H1 persistence. It produces `z∈R134`, which is standardized using training data only and encoded by:

```text
134 → 128 → 64
```

The fusion path is:

```text
v' = Wv(v)
t' = Wt(t)
g  = sigmoid(MLP([v || t]))
blend = g ⊙ v' + (1-g) ⊙ t'
interaction = v' ⊙ t'
h = MLP([blend || interaction])
logit = Linear(h)
```

The main reconstruction uses:

```text
L = 0.60 L_BCE(fused)
  + 0.30 L_response-KD(T=3)
  + 0.05 L_BCE(visual auxiliary)
  + 0.05 L_BCE(topology auxiliary)
```

Binary response distillation is implemented as temperature-scaled soft-target BCE:

```text
qT = sigmoid(teacher_logit / T)
L_KD = T² × BCEWithLogits(student_logit / T, qT)
```

## Fixed 134-D topology descriptor

The representation has fixed width across all ablations. Disabled topology groups are zero-masked rather than changing the MLP size.

For each of four canonical groups—sublevel-H0, sublevel-H1, superlevel-H0, superlevel-H1—the code records:

- 10 lifetime statistics;
- 4 counts above persistence thresholds;
- a 16-point Betti curve.

This yields `4×30=120` values. Seven binary intensity cuts provide seven connected-component counts and seven hole counts, producing 14 more values and a total of 134.

## TopoLite-MSF-KD

The multi-scale extension uses three resolutions, three filtrations, two homology dimensions, and the eight most persistent intervals per group:

```text
3 scales × 3 filtrations × 2 dimensions × 8 pairs = 144 tokens
```

Each token stores birth, death, lifetime, and normalized rank. H0/H1 token streams have separate Transformer encoders and bidirectional cross-attention. The resulting topology embedding conditions the last three visual stages through FiLM. Gated, concatenation, and interaction experts are combined by a learned router.

## TopoLite-FKD-SAM

This extension adds normalized feature matching from a student projection to the EfficientNet-B0 pooled representation and uses Sharpness-Aware Minimization. SAM performs two forward/backward passes per update; AMP is disabled by default in its config for simpler numerical behavior.

## TopoFM-Slice-v1

The implemented reconstruction contains only the components confirmed for the completed slice-level run:

- DINOv2 visual patch and class tokens;
- learnable multi-scale topology tokens;
- bidirectional cross-attention;
- supervised contrastive loss;
- validation-only temperature scaling and threshold selection.

The exact hidden dimensions and historical loss weights were unavailable, so the config labels them as reconstruction defaults. Patient-level 2.5D aggregation, VREx, domain-adversarial learning, soft masks, external validation, and ensembling remain disabled and are not claimed as tested.
