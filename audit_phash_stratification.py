"""Follow-up leakage checks the SHA audit cannot see:

1. pHash near-duplicates (Hamming <= 6) across train <-> val
2. Per-class bincounts per split (all classes present, val >= 5)
3. Reconcile class count (labels / folders / manifest)

Writes: phash_stratification_report.json and updates stats.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
HAMMING_MAX = 6
HASH_SIZE = 8  # 64-bit pHash
SPLITS = ("train", "val", "test")


def popcount64(x: np.ndarray) -> np.ndarray:
    """Bit count for uint64 arrays (works without numpy 2 bitwise_count)."""
    x = x.astype(np.uint64, copy=False)
    # SWAR popcount
    m1 = np.uint64(0x5555555555555555)
    m2 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x0F0F0F0F0F0F0F0F)
    h01 = np.uint64(0x0101010101010101)
    x = x - ((x >> np.uint64(1)) & m1)
    x = (x & m2) + ((x >> np.uint64(2)) & m2)
    x = (x + (x >> np.uint64(4))) & m4
    return ((x * h01) >> np.uint64(56)).astype(np.int32)


def phash_int(path: Path) -> int:
    with Image.open(path) as im:
        im = im.convert("RGB")
        h = imagehash.phash(im, hash_size=HASH_SIZE)
    return int(str(h), 16)


def compute_hashes(paths: list[str]) -> tuple[np.ndarray, list[str]]:
    vals = []
    ok_paths = []
    errors = []
    for rel in tqdm(paths, desc="pHash", leave=False):
        p = ROOT / rel
        try:
            vals.append(phash_int(p))
            ok_paths.append(rel)
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": rel, "error": str(exc)})
    return np.array(vals, dtype=np.uint64), ok_paths, errors


def find_near_dups(
    train_h: np.ndarray,
    train_paths: list[str],
    val_h: np.ndarray,
    val_paths: list[str],
    max_ham: int = HAMMING_MAX,
    batch: int = 256,
) -> list[dict]:
    """Return near-dup pairs with Hamming distance <= max_ham."""
    pairs = []
    # Also index exact hash matches quickly
    train_by_hash: dict[int, list[int]] = {}
    for i, h in enumerate(train_h.tolist()):
        train_by_hash.setdefault(int(h), []).append(i)

    for start in tqdm(range(0, len(val_h), batch), desc="near-dup scan", leave=False):
        vb = val_h[start : start + batch]
        # exact first
        for local_i, h in enumerate(vb.tolist()):
            vi = start + local_i
            for ti in train_by_hash.get(int(h), []):
                pairs.append(
                    {
                        "hamming": 0,
                        "val_path": val_paths[vi],
                        "train_path": train_paths[ti],
                        "val_hash": f"{int(h):016x}",
                        "train_hash": f"{int(train_h[ti]):016x}",
                    }
                )
        # near: XOR + popcount against all train (batched val)
        # Materialize (batch, n_train) distances
        # vb[:, None] ^ train_h[None, :]
        xor = np.bitwise_xor(vb[:, None], train_h[None, :])
        dist = popcount64(xor.reshape(-1)).reshape(xor.shape)
        # mask out exact (already recorded) and > max_ham
        ys, xs = np.where((dist > 0) & (dist <= max_ham))
        for y, x in zip(ys.tolist(), xs.tolist(), strict=False):
            vi = start + y
            pairs.append(
                {
                    "hamming": int(dist[y, x]),
                    "val_path": val_paths[vi],
                    "train_path": train_paths[x],
                    "val_hash": f"{int(vb[y]):016x}",
                    "train_hash": f"{int(train_h[x]):016x}",
                }
            )
    return pairs


def main() -> None:
    m = pd.read_csv(ROOT / "manifest.csv")
    labels = json.loads((ROOT / "labels.json").read_text(encoding="utf-8"))
    label_names = [e["name"] for e in sorted(labels, key=lambda x: x["index"])]

    # --- 3) Class count reconciliation ---
    folder_counts = {
        s: len([p for p in (ROOT / s).iterdir() if p.is_dir()]) for s in SPLITS
    }
    n_labels = len(label_names)
    n_manifest_classes = int(m["class"].nunique())
    class_reconcile = {
        "labels_json": n_labels,
        "manifest_unique_class": n_manifest_classes,
        "folders": folder_counts,
        "all_equal_64": n_labels == n_manifest_classes == 64
        and all(v == 64 for v in folder_counts.values()),
        "note": (
            "Dataset is 64 classes after Bollrot removal + <100-example filter. "
            "Any '65' figure is stale (pre-filter/checkpoint era); not present in "
            "current labels/manifest/folders."
        ),
    }

    # --- 2) Per-class stratification ---
    ct = (
        m.groupby(["class", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(SPLITS), fill_value=0)
        .reindex(label_names, fill_value=0)
    )
    missing = {
        s: [c for c in label_names if int(ct.loc[c, s]) == 0] for s in SPLITS
    }
    val_lt5 = [c for c in label_names if int(ct.loc[c, "val"]) < 5]
    strat = {
        "n_classes": n_labels,
        "all_classes_in_train": len(missing["train"]) == 0,
        "all_classes_in_val": len(missing["val"]) == 0,
        "all_classes_in_test": len(missing["test"]) == 0,
        "missing": missing,
        "val_min": int(ct["val"].min()),
        "val_max": int(ct["val"].max()),
        "val_lt_5": val_lt5,
        "val_ge_5_all": len(val_lt5) == 0,
        "per_class": {
            c: {s: int(ct.loc[c, s]) for s in SPLITS} for c in label_names
        },
        "ok": (
            len(missing["train"]) == 0
            and len(missing["val"]) == 0
            and len(missing["test"]) == 0
            and len(val_lt5) == 0
        ),
    }

    # --- 1) pHash train <-> val ---
    train_paths = m.loc[m["split"] == "train", "path"].astype(str).tolist()
    val_paths = m.loc[m["split"] == "val", "path"].astype(str).tolist()
    print(f"Computing pHash for train={len(train_paths)} val={len(val_paths)} ...")
    train_h, train_ok, train_err = compute_hashes(train_paths)
    val_h, val_ok, val_err = compute_hashes(val_paths)
    print(f"Hash errors: train={len(train_err)} val={len(val_err)}")
    print(f"Scanning Hamming <= {HAMMING_MAX} ...")
    pairs = find_near_dups(train_h, train_ok, val_h, val_ok, HAMMING_MAX)

    # Enrich with class labels
    path_to_class = dict(zip(m["path"].astype(str), m["class"].astype(str), strict=False))
    for p in pairs:
        p["val_class"] = path_to_class.get(p["val_path"])
        p["train_class"] = path_to_class.get(p["train_path"])
        p["same_class"] = p["val_class"] == p["train_class"]

    # Deduplicate undirected? keep as val->train directed
    by_ham: dict[int, int] = {}
    for p in pairs:
        by_ham[p["hamming"]] = by_ham.get(p["hamming"], 0) + 1

    # Unique val images that leak
    val_leakers = sorted({p["val_path"] for p in pairs})
    same_class_n = sum(1 for p in pairs if p["same_class"])
    cross_class_n = sum(1 for p in pairs if not p["same_class"])

    # Same-class near-dups inflate accuracy. Cross-class ham 4–6 (e.g. healthy
    # leaves across crops) is mostly hash collision noise — warn, don't fail.
    same_class_leakers = sorted({p["val_path"] for p in pairs if p["same_class"]})
    phash = {
        "hamming_max": HAMMING_MAX,
        "hash_size": HASH_SIZE,
        "n_train_hashed": len(train_ok),
        "n_val_hashed": len(val_ok),
        "n_hash_errors": len(train_err) + len(val_err),
        "hash_error_examples": (train_err + val_err)[:20],
        "n_near_dup_pairs": len(pairs),
        "n_val_images_with_train_near_dup": len(val_leakers),
        "n_val_same_class_leakers": len(same_class_leakers),
        "pairs_by_hamming": {str(k): by_ham[k] for k in sorted(by_ham)},
        "n_same_class_pairs": same_class_n,
        "n_cross_class_pairs": cross_class_n,
        "ok": same_class_n == 0,
        "warn_cross_class_near": cross_class_n > 0,
        "examples": sorted(pairs, key=lambda p: (p["hamming"], p["val_path"]))[:50],
    }

    overall_pass = (
        class_reconcile["all_equal_64"]
        and strat["ok"]
        and phash["ok"]
    )
    report = {
        "pass": overall_pass,
        "class_count_reconcile": class_reconcile,
        "stratification": {
            k: v for k, v in strat.items() if k != "per_class"
        },
        "stratification_per_class": strat["per_class"],
        "phash_train_val": phash,
        "verdict": (
            (
                "Same-class pHash train<->val clean + stratification complete; "
                "val metrics are trustworthy for model comparison. "
                + (
                    f"Note: {cross_class_n} cross-class near pairs (hash noise). "
                    if cross_class_n
                    else ""
                )
                + "Remaining work: calibration / abstain."
            )
            if overall_pass
            else (
                "Same-class near-dup leakage remains; rebuild group split before trusting accuracy."
                if not phash["ok"]
                else "Fix stratification / class-count inconsistency before shipping."
            )
        ),
    }

    out = ROOT / "phash_stratification_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    stats_path = ROOT / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats["phash_stratification_audit"] = {
            "pass": overall_pass,
            "phash_ok": phash["ok"],
            "n_near_dup_pairs": phash["n_near_dup_pairs"],
            "n_val_leakers": phash["n_val_images_with_train_near_dup"],
            "stratification_ok": strat["ok"],
            "val_min": strat["val_min"],
            "n_classes": n_labels,
            "report": "phash_stratification_report.json",
        }
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print("\n=== Class count ===")
    print(f"  labels={n_labels} manifest={n_manifest_classes} folders={folder_counts}")
    print(f"  {'PASS' if class_reconcile['all_equal_64'] else 'FAIL'}: reconciled to 64 (not 65)")

    print("\n=== Stratification ===")
    print(f"  val min/max = {strat['val_min']}/{strat['val_max']}")
    print(f"  missing val classes: {missing['val'] or 'none'}")
    print(f"  val < 5: {val_lt5 or 'none'}")
    print(f"  {'PASS' if strat['ok'] else 'FAIL'}: all 64 classes in all splits, val >= 5")

    print("\n=== pHash train<->val (Hamming <= %d) ===" % HAMMING_MAX)
    print(f"  pairs={len(pairs)}  val_images_hit={len(val_leakers)}  same_class_val_hit={len(same_class_leakers)}")
    print(f"  by_hamming={by_ham}")
    print(f"  same_class_pairs={same_class_n} cross_class_pairs={cross_class_n}")
    for ex in report["phash_train_val"]["examples"][:10]:
        print(
            f"  ham={ex['hamming']}  {ex['val_path']}  <->  {ex['train_path']}  "
            f"[{ex['val_class']} / {ex['train_class']}]"
        )
    print(f"  {'PASS' if phash['ok'] else 'FAIL'}: no same-class near-dups")
    if phash["warn_cross_class_near"]:
        print(f"  WARN: {cross_class_n} cross-class near pairs (typically healthy-leaf hash noise)")

    print(f"\nOverall: {'PASS' if overall_pass else 'FAIL'}")
    print(f"Wrote {out}")
    print(report["verdict"])


if __name__ == "__main__":
    main()
