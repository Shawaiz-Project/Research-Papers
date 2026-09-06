# Ablation Matrix

| ID | Configuration | Question |
|---|---|---|
| A0 | Visual student only | How much does the visual branch learn alone? |
| A1 | TDA MLP only | Is explicit topology independently predictive? |
| A2 | Visual + TDA concatenation | Does simple feature aggregation help? |
| A3 | Fixed 0.5/0.5 fusion | Is equal blending sufficient? |
| A4 | Input-conditioned gate, no KD | What is the contribution of adaptive fusion? |
| A5 | Visual student + KD, no TDA | How much improvement comes from KD alone? |
| A6 | Full TopoLite-KD | Combined proposed method. |
| A7 | Full model without Coordinate Attention | Contribution of coordinate attention. |
| A8 | H0 only | Contribution of connected-component persistence. |
| A9 | H1 only | Contribution of loop persistence. |
| A10a | Sublevel only | Contribution of low-intensity topology. |
| A10b | Superlevel only | Contribution of high-intensity topology. |

Use the same manifest, teacher, seeds, epoch budget, validation rule, and test protocol across all rows.
