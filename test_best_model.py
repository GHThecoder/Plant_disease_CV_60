"""Test the best EfficientNet (by val macro-F1) on the held-out test split.

Computes a wide multiclass metric suite and writes JSON/CSV artifacts under
``runs/test_best/``.

Usage:
  python test_best_model.py
  python test_best_model.py --backbone efficientnet_b0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    hamming_loss,
    hinge_loss,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
    zero_one_loss,
)
from tqdm import tqdm

import script1 as S

BACKBONES = ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"]
ECE_BINS = 15


# ---------------------------------------------------------------------------
# Checkpoint selection
# ---------------------------------------------------------------------------

def candidate_ckpt_paths(backbone: str) -> list[Path]:
    base = S.RUNS_DIR / backbone
    return [base / "best_overall_f1.pt", base / "fixed" / "best_f1.pt"]


def find_checkpoint(backbone: str) -> Path | None:
    for p in candidate_ckpt_paths(backbone):
        if p.is_file():
            return p
    return None


def load_ckpt_info(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    metrics = ckpt.get("metrics") or {}
    return {
        "path": path,
        "ckpt": ckpt,
        "backbone": ckpt.get("backbone", path.parent.name),
        "epoch": ckpt.get("epoch"),
        "val_f1_macro": float(metrics.get("f1_macro", -1.0)),
        "val_acc": metrics.get("acc"),
        "val_ece": metrics.get("ece"),
        "hparams": ckpt.get("hparams") or dict(S.DEFAULT_HPARAMS),
    }


def select_best(backbone: str | None = None) -> dict:
    if backbone is not None:
        path = find_checkpoint(backbone)
        if path is None:
            raise FileNotFoundError(f"No checkpoint for {backbone} under {S.RUNS_DIR}")
        return load_ckpt_info(path)

    best: dict | None = None
    missing: list[str] = []
    for bb in BACKBONES:
        path = find_checkpoint(bb)
        if path is None:
            missing.append(bb)
            continue
        info = load_ckpt_info(path)
        print(
            f"  candidate {bb}: val_f1_macro={info['val_f1_macro']:.4f} "
            f"epoch={info['epoch']}  ({info['path'].relative_to(S.RUNS_DIR)})"
        )
        if best is None or info["val_f1_macro"] > best["val_f1_macro"]:
            best = info
    if best is None:
        raise FileNotFoundError(f"No EfficientNet checkpoints under {S.RUNS_DIR}")
    if missing:
        print(f"  (no ckpt yet: {', '.join(missing)})")
    return best


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_proba(model: torch.nn.Module, loader: torch.utils.data.DataLoader):
    model.eval()
    all_logits, all_y = [], []
    for x, y in tqdm(loader, desc="test", leave=False):
        x = x.to(S.DEVICE, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.float().cpu().numpy())
        all_y.append(y.numpy())
    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return y_true, logits, probs


# ---------------------------------------------------------------------------
# Full metric suite
# ---------------------------------------------------------------------------

def _safe(fn, default=float("nan")):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def calibration_metrics(y_true: np.ndarray, probs: np.ndarray, n_bins: int = ECE_BINS) -> dict:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)
    bounds = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bins = []
    for i in range(n_bins):
        lo, hi = bounds[i], bounds[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        frac = float(mask.mean())
        if frac == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "frac": 0.0, "acc": 0.0, "conf": 0.0, "gap": 0.0})
            continue
        acc = float(correct[mask].mean())
        c = float(conf[mask].mean())
        gap = abs(acc - c)
        ece += frac * gap
        mce = max(mce, gap)
        bins.append({"lo": float(lo), "hi": float(hi), "frac": frac, "acc": acc, "conf": c, "gap": float(gap)})

    # Adaptive ECE (equal-mass bins)
    order = np.argsort(conf)
    conf_s, corr_s = conf[order], correct[order]
    edges = np.linspace(0, len(conf), n_bins + 1).astype(int)
    ace = 0.0
    for i in range(n_bins):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            continue
        ace += abs(corr_s[a:b].mean() - conf_s[a:b].mean()) * ((b - a) / len(conf))

    return {
        "ece": float(ece),
        "mce": float(mce),
        "ace_equal_mass": float(ace),
        "mean_confidence": float(conf.mean()),
        "mean_confidence_correct": float(conf[correct == 1].mean()) if correct.any() else float("nan"),
        "mean_confidence_incorrect": float(conf[correct == 0].mean()) if (correct == 0).any() else float("nan"),
        "reliability_bins": bins,
    }


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    onehot = np.eye(n_classes, dtype=np.float64)[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def per_class_binary_rates(y_true: np.ndarray, preds: np.ndarray, n_classes: int) -> pd.DataFrame:
    """One-vs-rest rates per class: specificity, NPV, FPR, FNR, FDR, FOR, threat score."""
    rows = []
    for c in range(n_classes):
        yt = y_true == c
        yp = preds == c
        tp = int(np.sum(yt & yp))
        tn = int(np.sum(~yt & ~yp))
        fp = int(np.sum(~yt & yp))
        fn = int(np.sum(yt & ~yp))
        support = int(yt.sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        fdr = fp / (fp + tp) if (fp + tp) else 0.0
        forg = fn / (fn + tn) if (fn + tn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        threat = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0  # critical success index / Jaccard
        rows.append(
            {
                "class_index": c,
                "class": S.CLASS_NAMES[c],
                "support": support,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "precision": prec,
                "recall_tpr_sensitivity": rec,
                "specificity_tnr": spec,
                "npv": npv,
                "fpr": fpr,
                "fnr": fnr,
                "fdr": fdr,
                "for": forg,
                "f1": f1,
                "threat_score_jaccard": threat,
            }
        )
    return pd.DataFrame(rows)


def compute_all_metrics(y_true: np.ndarray, logits: np.ndarray, probs: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    n_classes = probs.shape[1]
    labels_all = list(range(n_classes))
    present = np.unique(y_true)

    m: dict = {
        "n_test": int(len(y_true)),
        "n_classes": int(n_classes),
        "n_classes_present_in_test": int(len(present)),
    }

    # --- Accuracy family ---
    m["accuracy"] = float(accuracy_score(y_true, preds))
    m["balanced_accuracy"] = float(balanced_accuracy_score(y_true, preds))
    m["error_rate"] = float(1.0 - m["accuracy"])
    m["zero_one_loss"] = float(zero_one_loss(y_true, preds))
    for k in (1, 2, 3, 5, 10):
        kk = min(k, n_classes)
        m[f"top{kk}_accuracy"] = float(
            top_k_accuracy_score(y_true, probs, k=kk, labels=labels_all)
        )

    # --- Precision / recall / F-scores ---
    for avg in ("micro", "macro", "weighted"):
        m[f"precision_{avg}"] = float(precision_score(y_true, preds, average=avg, zero_division=0))
        m[f"recall_{avg}"] = float(recall_score(y_true, preds, average=avg, zero_division=0))
        m[f"f1_{avg}"] = float(f1_score(y_true, preds, average=avg, zero_division=0))
        m[f"fbeta0.5_{avg}"] = float(fbeta_score(y_true, preds, beta=0.5, average=avg, zero_division=0))
        m[f"fbeta2_{avg}"] = float(fbeta_score(y_true, preds, beta=2.0, average=avg, zero_division=0))
        m[f"jaccard_{avg}"] = float(jaccard_score(y_true, preds, average=avg, zero_division=0))

    # --- Agreement / correlation ---
    m["cohen_kappa"] = float(cohen_kappa_score(y_true, preds))
    m["cohen_kappa_linear"] = float(cohen_kappa_score(y_true, preds, weights="linear"))
    m["cohen_kappa_quadratic"] = float(cohen_kappa_score(y_true, preds, weights="quadratic"))
    m["mcc"] = float(matthews_corrcoef(y_true, preds))
    m["hamming_loss"] = float(hamming_loss(y_true, preds))

    # --- Probabilistic losses ---
    m["log_loss"] = float(_safe(lambda: log_loss(y_true, probs, labels=labels_all)))
    m["brier_multiclass"] = multiclass_brier(y_true, probs, n_classes)
    # Multiclass hinge needs label encoding as -1/+1 margins via decision function ≈ logits
    m["hinge_loss"] = float(
        _safe(lambda: hinge_loss(y_true, logits, labels=labels_all))
    )

    # --- Ranking / AUC ---
    def _roc(average: str, multi: str):
        return float(
            roc_auc_score(
                y_true,
                probs[:, present],
                multi_class=multi,
                average=average,
                labels=present,
            )
        )

    if len(present) >= 2:
        m["roc_auc_ovr_macro"] = float(_safe(lambda: _roc("macro", "ovr")))
        m["roc_auc_ovr_weighted"] = float(_safe(lambda: _roc("weighted", "ovr")))
        m["roc_auc_ovo_macro"] = float(_safe(lambda: _roc("macro", "ovo")))
        m["roc_auc_ovo_weighted"] = float(_safe(lambda: _roc("weighted", "ovo")))
    else:
        for k in (
            "roc_auc_ovr_macro",
            "roc_auc_ovr_weighted",
            "roc_auc_ovo_macro",
            "roc_auc_ovo_weighted",
        ):
            m[k] = float("nan")

    # PR-AUC (average precision) one-vs-rest
    def _ap(average: str):
        yt = np.eye(n_classes, dtype=np.int32)[y_true][:, present]
        return float(average_precision_score(yt, probs[:, present], average=average))

    m["pr_auc_ovr_macro"] = float(_safe(lambda: _ap("macro")))
    m["pr_auc_ovr_weighted"] = float(_safe(lambda: _ap("weighted")))
    m["pr_auc_ovr_micro"] = float(_safe(lambda: _ap("micro")))

    # --- Calibration ---
    cal = calibration_metrics(y_true, probs, ECE_BINS)
    m.update({k: v for k, v in cal.items() if k != "reliability_bins"})
    m["_reliability_bins"] = cal["reliability_bins"]

    # --- Confusion / report artifacts (not scalar) ---
    cm = confusion_matrix(y_true, preds, labels=labels_all)
    m["_confusion_matrix"] = cm
    m["_classification_report"] = classification_report(
        y_true,
        preds,
        labels=labels_all,
        target_names=S.CLASS_NAMES,
        zero_division=0,
        digits=4,
        output_dict=True,
    )
    m["_classification_report_text"] = classification_report(
        y_true,
        preds,
        labels=labels_all,
        target_names=S.CLASS_NAMES,
        zero_division=0,
        digits=4,
    )

    p, r, f1, support = precision_recall_fscore_support(
        y_true, preds, labels=labels_all, zero_division=0
    )
    m["_per_class_precision"] = p
    m["_per_class_recall"] = r
    m["_per_class_f1"] = f1
    m["_per_class_support"] = support

    # Most confused pairs (off-diagonal)
    pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                continue
            if cm[i, j] > 0:
                pairs.append(
                    {
                        "true": S.CLASS_NAMES[i],
                        "pred": S.CLASS_NAMES[j],
                        "count": int(cm[i, j]),
                    }
                )
    pairs.sort(key=lambda d: -d["count"])
    m["_top_confusions"] = pairs[:30]

    return m


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def print_section(title: str, keys: list[str], m: dict) -> None:
    print(f"\n=== {title} ===")
    width = max(len(k) for k in keys)
    for k in keys:
        v = m.get(k, float("nan"))
        if isinstance(v, float):
            print(f"  {k:<{width}}  {v:.6f}")
        else:
            print(f"  {k:<{width}}  {v}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=BACKBONES, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    print("DATA_ROOT", S.DATA_ROOT)
    print("classes", S.NUM_CLASSES)
    print("Selecting best EfficientNet checkpoint by val macro-F1...")
    info = select_best(args.backbone)
    ckpt = info["ckpt"]
    backbone = info["backbone"]
    hparams = info["hparams"]
    batch_size = int(args.batch_size or hparams.get("batch_size", 64))

    print("\n=== Selected ===")
    print(f"  backbone     : {backbone}")
    print(f"  checkpoint   : {info['path']}")
    print(f"  best epoch   : {info['epoch']}")
    print(f"  val f1_macro : {info['val_f1_macro']:.4f}")
    print(f"  val acc/ece  : {info['val_acc']} / {info['val_ece']}")
    print(f"  hparams      : {hparams}")

    ckpt_n = int(ckpt.get("num_classes", -1))
    if ckpt_n != S.NUM_CLASSES:
        print(
            f"ERROR: checkpoint num_classes={ckpt_n} != labels.json ({S.NUM_CLASSES})",
            file=sys.stderr,
        )
        sys.exit(1)

    model = S.create_model(
        backbone,
        hidden_mult=float(hparams.get("hidden_mult", 0.5)),
        drop_rate=float(hparams.get("drop_rate", 0.1)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, _, _, _, _, test_loader = S.make_loaders(batch_size)
    print(f"\nRunning inference on TEST loader (batch_size={batch_size})...")
    y_true, logits, probs = predict_proba(model, test_loader)
    preds = probs.argmax(axis=1)

    print("Computing full metric suite...")
    m = compute_all_metrics(y_true, logits, probs)
    per_df = per_class_binary_rates(y_true, preds, S.NUM_CLASSES)

    print_section(
        "Accuracy / top-k",
        [
            "n_test",
            "accuracy",
            "balanced_accuracy",
            "error_rate",
            "zero_one_loss",
            "top1_accuracy",
            "top2_accuracy",
            "top3_accuracy",
            "top5_accuracy",
            "top10_accuracy",
        ],
        m,
    )
    print_section(
        "Precision / Recall / F-scores / Jaccard",
        [
            "precision_micro",
            "precision_macro",
            "precision_weighted",
            "recall_micro",
            "recall_macro",
            "recall_weighted",
            "f1_micro",
            "f1_macro",
            "f1_weighted",
            "fbeta0.5_macro",
            "fbeta2_macro",
            "jaccard_macro",
            "jaccard_weighted",
        ],
        m,
    )
    print_section(
        "Agreement",
        [
            "cohen_kappa",
            "cohen_kappa_linear",
            "cohen_kappa_quadratic",
            "mcc",
            "hamming_loss",
        ],
        m,
    )
    print_section(
        "Probabilistic / ranking",
        [
            "log_loss",
            "brier_multiclass",
            "hinge_loss",
            "roc_auc_ovr_macro",
            "roc_auc_ovr_weighted",
            "roc_auc_ovo_macro",
            "roc_auc_ovo_weighted",
            "pr_auc_ovr_macro",
            "pr_auc_ovr_weighted",
            "pr_auc_ovr_micro",
        ],
        m,
    )
    print_section(
        "Calibration",
        [
            "ece",
            "mce",
            "ace_equal_mass",
            "mean_confidence",
            "mean_confidence_correct",
            "mean_confidence_incorrect",
        ],
        m,
    )

    order = np.argsort(m["_per_class_f1"])
    print("\n=== Worst-15 test F1 classes ===")
    for i in order[:15]:
        print(f"  {S.CLASS_NAMES[i]}: f1={m['_per_class_f1'][i]:.3f}  n={int(m['_per_class_support'][i])}")
    print("\n=== Best-15 test F1 classes ===")
    for i in order[-15:][::-1]:
        print(f"  {S.CLASS_NAMES[i]}: f1={m['_per_class_f1'][i]:.3f}  n={int(m['_per_class_support'][i])}")

    print("\n=== Top confused pairs (true -> pred) ===")
    for row in m["_top_confusions"][:15]:
        print(f"  {row['count']:4d}  {row['true']}  ->  {row['pred']}")

    # ---- Artifacts ----
    out_dir = S.RUNS_DIR / "test_best"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "selected_backbone": backbone,
        "checkpoint": str(info["path"]),
        "ckpt_epoch": info["epoch"],
        "val_f1_macro": info["val_f1_macro"],
        "val_acc": info["val_acc"],
        "val_ece": info["val_ece"],
        "hparams": hparams,
        "num_classes": S.NUM_CLASSES,
        "test_metrics": _jsonable(m),
        "reliability_bins": m["_reliability_bins"],
        "top_confusions": m["_top_confusions"],
        "classification_report": m["_classification_report"],
    }
    report_path = out_dir / "test_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    (out_dir / "classification_report.txt").write_text(
        m["_classification_report_text"], encoding="utf-8"
    )

    # Scalar metrics CSV (one row)
    scalar = {k: v for k, v in m.items() if not str(k).startswith("_")}
    pd.DataFrame([{**{"backbone": backbone, "ckpt_epoch": info["epoch"]}, **scalar}]).to_csv(
        out_dir / "test_metrics.csv", index=False
    )

    # Per-class rich table
    per_df["precision_sklearn"] = m["_per_class_precision"]
    per_df["recall_sklearn"] = m["_per_class_recall"]
    per_df["f1_sklearn"] = m["_per_class_f1"]
    per_df.to_csv(out_dir / "test_per_class.csv", index=False)

    # Confusion matrix
    cm = m["_confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=S.CLASS_NAMES, columns=S.CLASS_NAMES)
    cm_df.to_csv(out_dir / "confusion_matrix.csv")
    np.save(out_dir / "confusion_matrix.npy", cm)

    # Predictions dump
    pred_df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": preds,
            "y_true_name": [S.CLASS_NAMES[i] for i in y_true],
            "y_pred_name": [S.CLASS_NAMES[i] for i in preds],
            "confidence": probs.max(axis=1),
            "correct": preds == y_true,
        }
    )
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    print(f"\nWrote {out_dir / 'test_report.json'}")
    print(f"Wrote {out_dir / 'test_metrics.csv'}")
    print(f"Wrote {out_dir / 'test_per_class.csv'}")
    print(f"Wrote {out_dir / 'confusion_matrix.csv'}")
    print(f"Wrote {out_dir / 'classification_report.txt'}")
    print(f"Wrote {out_dir / 'test_predictions.csv'}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
