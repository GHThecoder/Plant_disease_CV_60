# %% [markdown]
# # Ultimate Plant Dataset v1 — EfficientNet B0–B2
#
# Group-split dataset (identity + SHA + pHash near-dups). ImageNet norm on all splits;
# geometric/color augmentations on **train only** (online — offline augs already dropped at build).
#
# - Backbones: `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2` (timm) + MLP head
# - Fixed hparams: **25 epochs** per backbone (Optuna off)
# - Metrics: accuracy, top-k, P/R/F1, kappa, MCC, log-loss, Brier, ROC-AUC, **ECE/MCE**
# - Best checkpoints by val macro-F1; test once at the end
# - Re-audit after rebuild: `audit_leakage.py`, `audit_phash_stratification.py`

# %%
from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

try:
    from IPython.display import display
except ImportError:  # plain `python script1.py`
    def display(obj):  # type: ignore[misc]
        print(obj)

# Optional Hugging Face auth (timm pretrained weights). Avoids rate-limit warning.
# Set HF_TOKEN in the environment, or run:  huggingface-cli login
_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _hf_token:
    try:
        from huggingface_hub import login as hf_login

        hf_login(token=_hf_token, add_to_git_credential=False)
        print("HF Hub: authenticated via HF_TOKEN")
    except Exception as exc:
        print("HF Hub: login failed:", exc)
else:
    print(
        "HF Hub: no HF_TOKEN set — unauthenticated downloads still work, "
        "but may be slower / rate-limited. Optional: huggingface-cli login"
    )

print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA required (RTX 5090 / cu128)"
DEVICE = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))

# %%
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Resolve dataset root whether kernel cwd is project root or this folder
_candidates = [
    Path.cwd(),
    Path.cwd() / "ultimate_plant_dataset_v1",
    Path(r"c:\Users\guray\Desktop\archive (45)\ultimate_plant_dataset_v1"),
]
DATA_ROOT = None
for candidate in _candidates:
    if (candidate / "train").is_dir() and (candidate / "labels.json").is_file():
        DATA_ROOT = candidate.resolve()
        break
if DATA_ROOT is None:
    raise FileNotFoundError("Could not find ultimate_plant_dataset_v1 (train/ + labels.json)")

RUNS_DIR = DATA_ROOT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR = DATA_ROOT / "mlruns"
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 25
# Windows spawn requires `if __name__ == "__main__"` when num_workers > 0
NUM_WORKERS = 4
IMG_SIZE = 224
ECE_BINS = 15
BACKBONES = ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"]
EXPERIMENT_NAME = "ultimate_plant_efficientnet_v1"

# Fixed hyperparameters (Optuna off)
DEFAULT_HPARAMS = {
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "hidden_mult": 0.5,
    "drop_rate": 0.1,
    "batch_size": 64,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LABELS = json.loads((DATA_ROOT / "labels.json").read_text(encoding="utf-8"))
CLASS_NAMES = [e["name"] for e in sorted(LABELS, key=lambda x: x["index"])]
NUM_CLASSES = len(CLASS_NAMES)
NAME_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

print("DATA_ROOT", DATA_ROOT)
print("classes", NUM_CLASSES, "epochs", EPOCHS, "backbones", len(BACKBONES))


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

# %% [markdown]
# ## Dataset, transforms, dataloaders

# %%
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".webp", ".bmp"}


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class PlantFolderDataset(Dataset):
    def __init__(self, split: str, transform=None):
        self.split = split
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        split_dir = DATA_ROOT / split
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue
            idx = NAME_TO_IDX[class_name]
            for p in class_dir.iterdir():
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    self.samples.append((p, idx))
        if not self.samples:
            raise FileNotFoundError(f"No images in {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y


def class_counts(dataset: PlantFolderDataset) -> np.ndarray:
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for _, y in dataset.samples:
        counts[y] += 1
    return counts


def make_weighted_sampler(dataset: PlantFolderDataset) -> WeightedRandomSampler:
    counts = class_counts(dataset)
    counts = np.maximum(counts, 1.0)
    class_w = 1.0 / counts
    sample_w = [class_w[y] for _, y in dataset.samples]
    return WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)


def make_loaders(batch_size: int):
    train_ds = PlantFolderDataset("train", transform=build_transforms(True))
    val_ds = PlantFolderDataset("val", transform=build_transforms(False))
    test_ds = PlantFolderDataset("test", transform=build_transforms(False))
    loader_kw = dict(
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    if NUM_WORKERS > 0:
        loader_kw["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=make_weighted_sampler(train_ds),
        drop_last=False,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kw,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kw,
    )
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


CLASS_WEIGHTS: torch.Tensor | None = None


def init_class_weights() -> torch.Tensor:
    """Build inverse-frequency weights (call from main process only)."""
    global CLASS_WEIGHTS
    train_ds_probe = PlantFolderDataset("train", transform=build_transforms(False))
    train_counts = class_counts(train_ds_probe)
    w = torch.tensor(1.0 / np.maximum(train_counts, 1.0), dtype=torch.float32)
    w = w / w.mean()
    CLASS_WEIGHTS = w.to(DEVICE)
    print(
        f"train={len(train_ds_probe)} "
        f"val={len(PlantFolderDataset('val', None))} "
        f"test={len(PlantFolderDataset('test', None))}"
    )
    return CLASS_WEIGHTS

# %% [markdown]
# ## Metrics + calibration (ECE / MCE)

# %%
def expected_calibration_error(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = ECE_BINS
) -> tuple[float, float, dict]:
    """Multiclass ECE/MCE using max-probability confidence."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float64)
    bin_bounds = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_acc, bin_conf, bin_frac = [], [], []
    for i in range(n_bins):
        lo, hi = bin_bounds[i], bin_bounds[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        frac = mask.mean()
        if frac == 0:
            bin_acc.append(0.0)
            bin_conf.append(0.0)
            bin_frac.append(0.0)
            continue
        acc = accuracies[mask].mean()
        conf = confidences[mask].mean()
        gap = abs(acc - conf)
        ece += frac * gap
        mce = max(mce, gap)
        bin_acc.append(float(acc))
        bin_conf.append(float(conf))
        bin_frac.append(float(frac))
    return float(ece), float(mce), {"bin_acc": bin_acc, "bin_conf": bin_conf, "bin_frac": bin_frac}


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    onehot = np.eye(n_classes, dtype=np.float64)[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def compute_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    prefix: str = "",
) -> dict:
    # numerically stable softmax
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    probs = exp / exp.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    n_classes = probs.shape[1]
    labels_all = list(range(n_classes))

    metrics: dict = {}
    metrics["acc"] = float(accuracy_score(y_true, preds))
    k3 = min(3, n_classes)
    k5 = min(5, n_classes)
    metrics["top3"] = float(top_k_accuracy_score(y_true, probs, k=k3, labels=labels_all))
    metrics["top5"] = float(top_k_accuracy_score(y_true, probs, k=k5, labels=labels_all))
    for avg in ("micro", "macro", "weighted"):
        metrics[f"precision_{avg}"] = float(
            precision_score(y_true, preds, average=avg, zero_division=0)
        )
        metrics[f"recall_{avg}"] = float(
            recall_score(y_true, preds, average=avg, zero_division=0)
        )
        metrics[f"f1_{avg}"] = float(f1_score(y_true, preds, average=avg, zero_division=0))
    metrics["kappa"] = float(cohen_kappa_score(y_true, preds))
    metrics["mcc"] = float(matthews_corrcoef(y_true, preds))
    try:
        metrics["log_loss"] = float(log_loss(y_true, probs, labels=labels_all))
    except ValueError:
        metrics["log_loss"] = float("nan")
    metrics["brier"] = multiclass_brier(y_true, probs, n_classes)
    # OVR AUC is undefined for classes absent from y_true.
    # Score only present classes; column order of y_score must match `labels`.
    try:
        present = np.unique(y_true)
        if len(present) >= 2:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(
                    y_true,
                    probs[:, present],
                    multi_class="ovr",
                    average="macro",
                    labels=present,
                )
            )
        else:
            metrics["roc_auc_ovr_macro"] = float("nan")
    except ValueError:
        metrics["roc_auc_ovr_macro"] = float("nan")

    ece, mce, rel = expected_calibration_error(y_true, probs, ECE_BINS)
    metrics["ece"] = ece
    metrics["mce"] = mce
    metrics["_reliability"] = rel

    per_f1 = f1_score(y_true, preds, average=None, labels=labels_all, zero_division=0)
    metrics["_per_class_f1"] = per_f1

    if prefix:
        return {f"{prefix}_{k}" if not k.startswith("_") else k: v for k, v in metrics.items()}
    return metrics


def format_metrics_row(m: dict, keys: list[str]) -> str:
    parts = []
    for k in keys:
        v = m.get(k, float("nan"))
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)


COMPARE_KEYS = [
    "acc",
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "recall_macro",
    "top3",
    "top5",
    "kappa",
    "mcc",
    "log_loss",
    "brier",
    "roc_auc_ovr_macro",
    "ece",
    "mce",
]


def print_train_val_comparison(epoch: int, train_m: dict, val_m: dict) -> None:
    print(f"\n=== Epoch {epoch} — train vs val (over/underfit check) ===")
    print("TRAIN:", format_metrics_row(train_m, COMPARE_KEYS))
    print("VAL:  ", format_metrics_row(val_m, COMPARE_KEYS))
    gap = train_m["f1_macro"] - val_m["f1_macro"]
    print(f"F1_macro gap (train-val)={gap:+.4f}  |  ECE_val={val_m['ece']:.4f}")
    per = val_m.get("_per_class_f1")
    if per is not None:
        order = np.argsort(per)[:10]
        print("Worst-10 val F1 classes:")
        for i in order:
            print(f"  {CLASS_NAMES[i]}: {per[i]:.3f}")

# %% [markdown]
# ## Train / eval loop + checkpointing

# %%
class FourLayerMLPHead(nn.Module):
    """4-layer MLP classifier: Linear → BN → SiLU → Dropout → Linear → BN → SiLU → Dropout → Linear → BN → SiLU → Dropout → Linear."""

    def __init__(self, in_features: int, num_classes: int, hidden_mult: float = 0.5, drop_rate: float = 0.1):
        super().__init__()
        h1 = max(int(in_features * hidden_mult), num_classes * 2)
        h2 = max(h1 // 2, num_classes)
        d = float(drop_rate)
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.SiLU(inplace=True),
            nn.Dropout(d),
            nn.Linear(in_features, h1),
            nn.BatchNorm1d(h1),
            nn.SiLU(inplace=True),
            nn.Dropout(d),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.SiLU(inplace=True),
            nn.Dropout(d),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EfficientNetMLP(nn.Module):
    """timm EfficientNet backbone + 4-layer MLP head (SiLU, BatchNorm, Dropout)."""

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        hidden_mult: float = 0.5,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            num_classes=0,  # feature extractor
            global_pool="avg",
        )
        in_features = int(self.backbone.num_features)
        self.head = FourLayerMLPHead(
            in_features, num_classes, hidden_mult=hidden_mult, drop_rate=drop_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)


def create_model(
    backbone: str, hidden_mult: float = 0.5, drop_rate: float = 0.1
) -> nn.Module:
    model = EfficientNetMLP(
        backbone, NUM_CLASSES, hidden_mult=hidden_mult, drop_rate=drop_rate
    )
    return model.to(DEVICE)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_y = [], []
    for x, y in tqdm(loader, desc="val/test", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.float().cpu().numpy())
        all_y.append(y.cpu().numpy())
    logits_np = np.concatenate(all_logits, axis=0)
    y_np = np.concatenate(all_y, axis=0)
    metrics = compute_metrics(y_np, logits_np)
    return metrics, y_np, logits_np


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
) -> tuple[dict, float]:
    model.train()
    all_logits, all_y = [], []
    running_loss = 0.0
    n = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += float(loss.item()) * y.size(0)
        n += y.size(0)
        all_logits.append(logits.detach().float().cpu().numpy())
        all_y.append(y.cpu().numpy())
    logits_np = np.concatenate(all_logits, axis=0)
    y_np = np.concatenate(all_y, axis=0)
    metrics = compute_metrics(y_np, logits_np)
    return metrics, running_loss / max(n, 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    metrics: dict,
    hparams: dict,
    backbone: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # strip private arrays for smaller files
    metrics_clean = {k: v for k, v in metrics.items() if not k.startswith("_")}
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics_clean,
            "hparams": hparams,
            "backbone": backbone,
            "class_names": CLASS_NAMES,
            "num_classes": NUM_CLASSES,
            "imagenet_mean": IMAGENET_MEAN,
            "imagenet_std": IMAGENET_STD,
        },
        path,
    )


def run_training(backbone: str, hparams: dict) -> float:
    """Train one backbone for EPOCHS; save best-by-val-F1 checkpoint."""
    set_seed(SEED)
    batch_size = int(hparams["batch_size"])
    _, _, _, train_loader, val_loader, _ = make_loaders(batch_size)

    model = create_model(
        backbone,
        hidden_mult=float(hparams["hidden_mult"]),
        drop_rate=float(hparams["drop_rate"]),
    )
    if CLASS_WEIGHTS is None:
        init_class_weights()
    criterion = nn.CrossEntropyLoss(
        weight=CLASS_WEIGHTS,
        label_smoothing=float(hparams["label_smoothing"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hparams["lr"]),
        weight_decay=float(hparams["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda")

    run_dir = RUNS_DIR / backbone / "fixed"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_val_metrics: dict | None = None
    best_epoch = -1

    mlflow.log_params({**hparams, "backbone": backbone, "epochs": EPOCHS})

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_m, train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        val_m, _, _ = evaluate(model, val_loader)
        dt = time.time() - t0

        print_train_val_comparison(epoch, train_m, val_m)
        print(f"train_loss={train_loss:.4f}  time={dt:.1f}s")

        log_payload = {
            "train_loss": train_loss,
            **{f"train_{k}": train_m[k] for k in COMPARE_KEYS},
            **{f"val_{k}": val_m[k] for k in COMPARE_KEYS},
        }
        mlflow.log_metrics(log_payload, step=epoch)

        if val_m["f1_macro"] > best_f1:
            best_f1 = float(val_m["f1_macro"])
            best_val_metrics = {k: v for k, v in val_m.items() if not k.startswith("_")}
            best_epoch = epoch
            save_checkpoint(
                run_dir / "best_f1.pt",
                model,
                epoch,
                val_m,
                hparams,
                backbone,
            )
            print(f"* saved best_f1.pt (val f1_macro={best_f1:.4f})")

    overall_path = RUNS_DIR / backbone / "best_overall_f1.pt"
    if (run_dir / "best_f1.pt").is_file():
        ckpt = torch.load(run_dir / "best_f1.pt", map_location="cpu", weights_only=False)
        torch.save(ckpt, overall_path)
        print(f"* wrote {overall_path} (f1_macro={best_f1:.4f} @ epoch {best_epoch})")

    meta = {
        "backbone": backbone,
        "best_epoch": best_epoch,
        "best_val_f1_macro": best_f1,
        "best_val_metrics": best_val_metrics,
        "hparams": hparams,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    mlflow.log_metrics({"best_val_f1_macro": best_f1, "best_epoch": best_epoch})

    del model
    torch.cuda.empty_cache()
    return best_f1


def run_all_backbones() -> dict:
    """Train each backbone once with DEFAULT_HPARAMS for EPOCHS."""
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    results: dict = {}
    hparams = dict(DEFAULT_HPARAMS)

    for backbone in BACKBONES:
        print("\n" + "=" * 72)
        print(f"Train: {backbone}  ({EPOCHS} epochs, fixed hparams)")
        print("=" * 72)
        print("hparams:", hparams)

        with mlflow.start_run(run_name=f"{backbone}_fixed"):
            try:
                best_f1 = run_training(backbone, hparams)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"OOM on {backbone}; skipping.")
                    continue
                raise

        results[backbone] = {
            "backbone": backbone,
            "best_val_f1_macro": best_f1,
            "hparams": hparams,
        }
        out_json = RUNS_DIR / backbone / f"run_{backbone}.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(results[backbone], indent=2), encoding="utf-8")
        print(f"Best {backbone}: f1={best_f1:.4f}")
        print(f"Wrote {out_json}")

    print("\nAll backbones done.")
    print(
        pd.DataFrame(
            [
                {
                    "backbone": b,
                    "best_val_f1_macro": results[b]["best_val_f1_macro"],
                    **results[b]["hparams"],
                }
                for b in BACKBONES
                if b in results
            ]
        )
    )
    return results


def run_test_summary() -> None:
    summary_rows = []
    test_rows = []

    for backbone in BACKBONES:
        ckpt_path = RUNS_DIR / backbone / "best_overall_f1.pt"
        if not ckpt_path.is_file():
            print(f"Missing checkpoint for {backbone}: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hparams = ckpt["hparams"]
        val_metrics = ckpt.get("metrics", {})

        row = {
            "backbone": backbone,
            "ckpt_epoch": ckpt.get("epoch"),
            "val_f1_macro": val_metrics.get("f1_macro"),
            "val_acc": val_metrics.get("acc"),
            "val_ece": val_metrics.get("ece"),
            "val_mce": val_metrics.get("mce"),
            "path": str(ckpt_path),
        }
        summary_rows.append(row)

        model = create_model(
            backbone,
            hidden_mult=float(hparams.get("hidden_mult", 0.5)),
            drop_rate=float(hparams.get("drop_rate", 0.1)),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        batch_size = int(hparams.get("batch_size", 64))
        _, _, _, _, _, test_loader = make_loaders(batch_size)
        test_m, _, _ = evaluate(model, test_loader)
        print(f"\nTEST [{backbone}]:", format_metrics_row(test_m, COMPARE_KEYS))
        test_rows.append({"backbone": backbone, **{k: test_m[k] for k in COMPARE_KEYS}})
        del model
        torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_rows)
    test_df = pd.DataFrame(test_rows)
    print("\n=== Val summary (best checkpoints) ===")
    display(summary_df)
    print("\n=== Test metrics (held-out, no tuning) ===")
    display(test_df)

    summary_df.to_csv(RUNS_DIR / "summary_val.csv", index=False)
    test_df.to_csv(RUNS_DIR / "summary_test.csv", index=False)
    print("Wrote", RUNS_DIR / "summary_val.csv", "and", RUNS_DIR / "summary_test.csv")


if __name__ == "__main__":
    # Required on Windows when DataLoader num_workers > 0 (spawn)
    import multiprocessing as mp

    mp.freeze_support()
    run_all_backbones()
    run_test_summary()


