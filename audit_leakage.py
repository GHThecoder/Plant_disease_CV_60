"""Data leakage / integrity audit for ultimate_plant_dataset_v1.

Checks:
  1. identity_id / sha256 / original_path never shared across train/val/test
  2. Duplicate SHA within a split (esp. cross-class = label conflict)
  3. Manifest paths exist on disk; disk images appear in manifest
  4. On-disk SHA-256 matches manifest (full rehash)
  5. labels.json vs folder names vs class_index consistency

Writes:
  - leakage_audit_report.json
  - updates stats.json["leakage_audit"]
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".webp", ".bmp"}
SPLITS = ("train", "val", "test")


def file_sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def list_disk_images() -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {s: [] for s in SPLITS}
    for split in SPLITS:
        split_dir = ROOT / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for p in class_dir.iterdir():
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    out[split].append(p)
    return out


def main() -> None:
    manifest_path = ROOT / "manifest.csv"
    labels_path = ROOT / "labels.json"
    stats_path = ROOT / "stats.json"

    m = pd.read_csv(manifest_path)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    label_names = [e["name"] for e in sorted(labels, key=lambda x: x["index"])]
    name_to_idx = {e["name"]: e["index"] for e in labels}

    report: dict = {
        "dataset_root": str(ROOT),
        "n_manifest": int(len(m)),
        "n_labels": len(label_names),
        "split_counts_manifest": m["split"].value_counts().to_dict(),
        "checks": {},
        "pass": True,
    }

    # --- 1) Cross-split key overlaps ---
    cross: dict = {}
    for key in ("identity_id", "sha256", "original_path"):
        by_key = m.groupby(key)["split"].agg(lambda s: sorted(set(s)))
        leaks = {str(k): v for k, v in by_key.items() if len(v) > 1}
        cross[key] = {
            "ok": len(leaks) == 0,
            "n_leaking_keys": len(leaks),
            "examples": dict(list(leaks.items())[:10]),
        }
        if leaks:
            report["pass"] = False
    report["checks"]["cross_split"] = cross

    # --- 2) Duplicate SHA (global / cross-class) ---
    sha_groups = m.groupby("sha256")
    dup_rows = []
    for sha, g in sha_groups:
        if len(g) < 2:
            continue
        classes = sorted(g["class"].unique())
        splits = sorted(g["split"].unique())
        dup_rows.append(
            {
                "sha256": sha,
                "n": int(len(g)),
                "classes": classes,
                "splits": splits,
                "cross_class": len(classes) > 1,
                "cross_split": len(splits) > 1,
                "paths": g["path"].tolist(),
            }
        )
    cross_class_dups = [d for d in dup_rows if d["cross_class"]]
    cross_split_dups = [d for d in dup_rows if d["cross_split"]]
    report["checks"]["duplicate_sha"] = {
        "ok": len(cross_split_dups) == 0,
        # Cross-class same-SHA is a data-quality / soft-leakage issue (same pixels, two labels)
        "n_duplicate_sha_groups": len(dup_rows),
        "n_cross_class": len(cross_class_dups),
        "n_cross_split": len(cross_split_dups),
        "cross_class_examples": cross_class_dups[:20],
        "warn_cross_class": len(cross_class_dups) > 0,
    }
    if cross_split_dups:
        report["pass"] = False

    # Duplicate identity_id
    id_dups = int(m["identity_id"].duplicated().sum())
    report["checks"]["duplicate_identity"] = {
        "ok": id_dups == 0,
        "n_duplicate_rows": id_dups,
    }
    if id_dups:
        report["pass"] = False

    # --- 3) Manifest vs disk ---
    disk = list_disk_images()
    disk_rel = {}
    for split, paths in disk.items():
        for p in paths:
            rel = p.relative_to(ROOT).as_posix()
            disk_rel[rel] = p

    man_paths = set(m["path"].astype(str).str.replace("\\", "/"))
    missing_on_disk = sorted(man_paths - set(disk_rel))
    orphan_on_disk = sorted(set(disk_rel) - man_paths)
    report["checks"]["manifest_vs_disk"] = {
        "ok": len(missing_on_disk) == 0 and len(orphan_on_disk) == 0,
        "n_disk": len(disk_rel),
        "n_missing_on_disk": len(missing_on_disk),
        "n_orphan_on_disk": len(orphan_on_disk),
        "missing_examples": missing_on_disk[:20],
        "orphan_examples": orphan_on_disk[:20],
    }
    if missing_on_disk or orphan_on_disk:
        report["pass"] = False

    # --- 4) Labels / folders / class_index ---
    folder_issues = []
    for split in SPLITS:
        folders = {p.name for p in (ROOT / split).iterdir() if p.is_dir()} if (ROOT / split).is_dir() else set()
        for name in label_names:
            if name not in folders:
                folder_issues.append(f"missing folder {split}/{name}")
        for name in sorted(folders - set(label_names)):
            folder_issues.append(f"extra folder {split}/{name}")

    bad_idx = m[m.apply(lambda r: name_to_idx.get(r["class"]) != r["class_index"], axis=1)]
    report["checks"]["labels_consistency"] = {
        "ok": len(folder_issues) == 0 and len(bad_idx) == 0,
        "n_folder_issues": len(folder_issues),
        "folder_issues": folder_issues[:30],
        "n_bad_class_index": int(len(bad_idx)),
    }
    if folder_issues or len(bad_idx):
        report["pass"] = False

    # --- 5) Full on-disk rehash vs manifest ---
    print("Rehashing on-disk files (this may take a few minutes)...")
    sha_mismatch = []
    hash_fail = []
    for i, row in enumerate(m.itertuples(index=False), 1):
        rel = str(row.path).replace("\\", "/")
        p = ROOT / rel
        if not p.is_file():
            continue
        try:
            got = file_sha256(p)
        except OSError as exc:
            hash_fail.append({"path": rel, "error": str(exc)})
            continue
        if got != row.sha256:
            sha_mismatch.append({"path": rel, "manifest": row.sha256, "disk": got})
        if i % 10000 == 0:
            print(f"  hashed {i}/{len(m)}")

    report["checks"]["rehash"] = {
        "ok": len(sha_mismatch) == 0 and len(hash_fail) == 0,
        "n_checked": int(len(m) - len([1 for r in m.itertuples(index=False) if not (ROOT / str(r.path).replace('\\', '/')).is_file()])),
        "n_mismatch": len(sha_mismatch),
        "n_hash_errors": len(hash_fail),
        "mismatch_examples": sha_mismatch[:20],
        "error_examples": hash_fail[:20],
    }
    if sha_mismatch or hash_fail:
        report["pass"] = False

    # Summary flags (compat with stats.json)
    report["identity_ok"] = cross["identity_id"]["ok"] and report["checks"]["duplicate_identity"]["ok"]
    report["sha_ok"] = cross["sha256"]["ok"] and report["checks"]["duplicate_sha"]["ok"]
    report["original_path_ok"] = cross["original_path"]["ok"]
    report["disk_ok"] = report["checks"]["manifest_vs_disk"]["ok"]
    report["rehash_ok"] = report["checks"]["rehash"]["ok"]
    report["labels_ok"] = report["checks"]["labels_consistency"]["ok"]

    out_path = ROOT / "leakage_audit_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats["leakage_audit"] = {
            "pass": report["pass"],
            "identity_ok": report["identity_ok"],
            "sha_ok": report["sha_ok"],
            "original_path_ok": report["original_path_ok"],
            "disk_ok": report["disk_ok"],
            "rehash_ok": report["rehash_ok"],
            "labels_ok": report["labels_ok"],
            "n_cross_class_duplicate_sha": len(cross_class_dups),
            "report": "leakage_audit_report.json",
        }
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print("\n=== Leakage audit ===")
    for name, ok in [
        ("identity cross-split", report["identity_ok"]),
        ("sha cross-split", report["sha_ok"]),
        ("original_path cross-split", report["original_path_ok"]),
        ("manifest vs disk", report["disk_ok"]),
        ("rehash match", report["rehash_ok"]),
        ("labels/folders", report["labels_ok"]),
    ]:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    if report["checks"]["duplicate_sha"]["warn_cross_class"]:
        print(
            f"  WARN: {report['checks']['duplicate_sha']['n_cross_class']} "
            "cross-class duplicate SHA group(s) (same pixels, different labels)"
        )
        for ex in report["checks"]["duplicate_sha"]["cross_class_examples"][:5]:
            print(f"    classes={ex['classes']} paths={ex['paths']}")
    print(f"\nOverall: {'PASS' if report['pass'] else 'FAIL'}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
