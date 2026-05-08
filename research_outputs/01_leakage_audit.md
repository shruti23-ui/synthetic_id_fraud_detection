# Data-leakage audit

## 1. Source-template leakage

- Total records: 2222
- Total unique source templates: 1000
- Templates appearing in TRAIN: 797
- Templates appearing in TEST:  374
- **Templates shared train ∩ test:** 249
- **TEST images whose template also appears in TRAIN:** 289 / 444 (65.1 %)
- **Cross-class template leaks (real of template T in train, fake of T in test or vice versa):** 244 (55.0 %)

Sample of shared templates: alb_id_01, alb_id_02, alb_id_05, alb_id_19, alb_id_21, alb_id_22, alb_id_24, alb_id_25, alb_id_26, alb_id_38

## 2. Near-duplicate (pHash) leakage

- Hamming threshold for 'near-duplicate': 8 bits out of 64
- Median nearest-train distance: 1.0
- 10th percentile: 0.0
- Test images with a near-duplicate in train: 440 / 444 (99.1 %)
- Of those, **cross-class** near-duplicates: 280 (63.1 %)

## Interpretation

> **CRITICAL:** more than half of the test set sits on a template that the model already saw during training. The 0.99 ROC-AUC almost certainly includes a substantial 'template-memorisation' component. The headline number is *not* a clean measurement of the model's ability to detect forgery patterns on unseen IDs.

## Recommended fix

Re-split the dataset by **template ID** rather than by image: assign each unique template to exactly one of train / val / test, then take every real and every fake derived from that template into the same split. This eliminates the channel by which the model can score via template memorisation.

See `research/02_template_split_retrain.py` for the implementation.
