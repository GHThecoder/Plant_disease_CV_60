# Ultimate Plant Dataset v1

Leakage-free merge of PlantVillage-family sources (`2/`, `valid/`, `PlantVillage/`)
and the crop pest/disease set (`Train/`, `Validation/`).

## Rules

- Offline augmentations dropped (suffix: `_180deg`, `_flip*`, `_newGRR`, `_new*deg*`, `_newPixel*`, …; prefix: `contrast_`, `rotation_`, …).
- Nested `PlantVillage/PlantVillage` ignored.
- Images grouped by PlantVillage identity stem or crop basename, plus SHA-256.
- pHash near-duplicates (Hamming ≤ 6) share a split unit (group split).
- Each split unit assigned to exactly one of train / val / test.
- Split: 70% / 15% / 15% (seed=42).
- Maize semantic merges: Common rust, Gray leaf spot, healthy.
- Post-filter: drop classes with total < 100; if a crop's healthy class is below 100, drop all classes for that crop.
- Always exclude: denylisted single-crop classes + legacy Train/Validation rice.
- Rice only from ``Rice_Leaf_AUG`` **non-augmented** originals (``aug_*`` dropped).

## Summary

- Classes: **60**
- Images kept: **46677** (from 124736 scanned; dropped 45460 augs; collapsed 32599 exact/identity dupes; pHash split-links=3846)
- Splits: train=32657, val=6985, test=7035
- Leakage audit (identity/SHA): **PASS**
- Re-check near-dups: `python audit_phash_stratification.py`

## Files

- `train|val|test/<class>/` — images
- `labels.json` — class index, crop, disease, sources
- `manifest.csv` — per-image provenance
- `stats.json` — counts and audit
- `script1.py` — training (preserved across rebuilds)
- `audit_leakage.py` / `audit_phash_stratification.py` — auditors
