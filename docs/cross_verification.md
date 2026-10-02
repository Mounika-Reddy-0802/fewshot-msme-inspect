# Cross-Verification and Benchmark Summary

## 1. Overall Image AUROC Performance Across Datasets & Backbones

| Dataset | Backbone | 1-shot | 5-shot | 10-shot |
|---|---|---|---|---|
| **MVTec AD** | ViT-B-16-quickgelu | 0.856 | 0.888 | 0.904 |
| **MVTec AD** | ViT-B-32-quickgelu | 0.846 | 0.886 | 0.904 |
| **VisA** | ViT-B-16-quickgelu | 0.773 | 0.811 | 0.831 |
| **VisA** | ViT-B-32-quickgelu | 0.681 | 0.748 | 0.765 |
| **NEU-DET** | ViT-B-16-quickgelu | 0.810 | 0.854 | 0.870 |
| **NEU-DET** | ViT-B-32-quickgelu | 0.818 | 0.867 | 0.893 |

---

## 2. NEU-DET Class Breakdown

| Class | ViT-B/16 (1/5/10-shot) | ViT-B/32 (1/5/10-shot) |
|---|---|---|
| **crazing** | 0.991 / 0.993 / 0.994 | 0.876 / 0.917 / 0.931 |
| **inclusion** | 0.658 / 0.780 / 0.831 | 0.871 / 0.901 / 0.946 |
| **patches** | 0.704 / 0.779 / 0.828 | 0.803 / 0.874 / 0.901 |
| **pitted_surface** | 0.878 / 0.921 / 0.917 | 0.709 / 0.834 / 0.864 |
| **rolled-in_scale** | 0.974 / 0.979 / 0.982 | 0.938 / 0.992 / 0.994 |
| **scratches** | 0.724 / 0.757 / 0.768 | 0.709 / 0.683 / 0.723 |
| **MEAN** | **0.810 / 0.854 / 0.870** | **0.818 / 0.867 / 0.893** |

---

## 3. Incremental Class Addition Evaluation (5-shot, ViT-B-16)

- **Total Classes Added:** 15
- **Total Time Taken:** 21.7 ms
- **Memory Overhead:** 43.07 MiB (14,700 prototypes)
- **Mean Image AUROC:** 0.8863
- **Append-Only Isolation:** PASSED (bit-identical scores for prior classes across all steps)
