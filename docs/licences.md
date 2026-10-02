# Dataset licences

We ship download scripts only (`scripts/download_mvtec_full.py`,
`scripts/download_visa.py`, `scripts/reorganize_neu_det.py`); no dataset files
are committed to this repo (`data/` is gitignored). Anyone reproducing this
work downloads directly from the original sources below.

## MVTec AD

- **Source:** https://www.mvtec.com/company/research/datasets/mvtec-ad
- **Licence:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  International (CC BY-NC-SA 4.0)
- **Permits:** Research / non-commercial use only, with attribution and
  share-alike. Explicitly **not** licensed for commercial use.
- **Citation:** Bergmann et al., "The MVTec Anomaly Detection Dataset: A
  Comprehensive Real-World Dataset for Unsupervised Anomaly Detection",
  IJCV 2021.

## VisA (Visual Anomaly)

- **Source:** https://registry.opendata.aws/visa/ (official AWS Open Data
  listing, published by Amazon Science)
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Permits:** Any use, including commercial, with attribution.
- **Citation:** Zou et al., "SPot-the-Difference Self-Supervised Pre-training
  for Anomaly Detection and Segmentation", ECCV 2022.
- **Note on a discrepancy:** Amazon's own AWS Open Data listing and GitHub
  (`amazon-science/spot-diff`) both state CC BY 4.0. One secondary source
  (anomalib's documentation) instead states CC BY-NC-SA 4.0. We treat
  Amazon's own listing as authoritative since they are the dataset's
  publisher, but this is worth re-checking before any commercial use of
  results derived from VisA, given the conflicting secondary claim.

## NEU-DET

- **Source:** Northeastern University surface defect database (Song & Yan,
  2013); original host: http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/
  Downloaded here via a Kaggle mirror
  (kaustubhdikshit/neu-surface-defect-database), since the original site's
  automated/API access was unreliable (403/401/404 across Kaggle CLI,
  Hugging Face and GitHub mirror attempts -- see commit history).
- **Licence: unclear.** The original Northeastern University page does not
  state a machine-readable licence; the dataset is cited by academic
  convention (cite the paper, not a licence grant). One third-party re-host
  (IEEE Dataport, uploaded by an unrelated party, not the original authors)
  claims CC BY 4.0, but this is not authoritative for the original dataset
  and should not be relied on without independent verification.
- **Permits:** Treat as research/academic use only unless and until an
  explicit licence grant from the original authors is found. **Do not** use
  for any commercial purpose or redistribution without checking with
  Northeastern University directly first.
- **Citation:** Song, K. and Yan, Y., "A noise robust method based on
  completed local binary patterns for hot-rolled steel strip surface
  defects", Applied Surface Science, 2013.
