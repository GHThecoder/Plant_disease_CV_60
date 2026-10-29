"""Fit temperature scaling on VAL (NLL), report ECE before/after, save T.

Temperature scaling (Guo et al.): keep argmax the same, only rescale confidence:
    probs = softmax(logits / T)

Fit T on **validation only** (never on test). Then apply at inference.

Usage:
  python temperature_scaling.py
  python temperature_scaling.py --backbone efficientnet_b0
  python temperature_scaling.py --backbone efficientnet_b0 --eval-test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import script1 as S

BACKBONES = ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"]


def find_ckpt(backbone: str) -> Path:
    for p in (
        S.RUNS_DIR / backbone / "best_overall_f1.pt",
        S.RUNS_DIR / backbone / "fixed" / "best_f1.pt",
    ):
        if p.is_file():
            return p
    raise FileNotFoundError(f"No checkpoint for {backbone}")


def load_model(backbone: str, ckpt_path: Path) -> tuple[nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hparams") or dict(S.DEFAULT_HPARAMS)
    n_cls = int(ckpt.get("num_classes", S.NUM_CLASSES))
    if n_cls != S.NUM_CLASSES:
        raise RuntimeError(f"num_classes mismatch: ckpt={n_cls} labels={S.NUM_CLASSES}")
    model = S.EfficientNetMLP(
        backbone,
        n_cls,
        hidden_mult=float(hparams.get("hidden_mult", 0.5)),
        drop_rate=float(hparams.get("drop_rate", 0.1)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(S.DEVICE).eval()
    return model, ckpt


@torch.no_grad()
def collect_logits(model: nn.Module, loader) -> tuple[torch.Tensor, torch.Tensor]:
    all_logits, all_y = [], []
    for x, y in tqdm(loader, desc="logits", leave=False):
        x = x.to(S.DEVICE, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.float().cpu())
        all_y.append(y)
    return torch.cat(all_logits, 0), torch.cat(all_y, 0)


def softmax_np(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll_np(logits: np.ndarray, y: np.ndarray, T: float = 1.0) -> float:
    probs = softmax_np(logits, T)
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.log(p).mean())


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Minimize val NLL w.r.t. scalar T > 0 (optimize log_T)."""
    # logits: [N, C], labels: [N]
    log_T = nn.Parameter(torch.zeros(1))  # T = exp(0) = 1
    optimizer = torch.optim.LBFGS([log_T], lr=0.01, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        T = log_T.exp().clamp(min=1e-3, max=100.0)
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = float(log_T.exp().clamp(min=1e-3, max=100.0).item())
    return T


def summarize(name: str, y: np.ndarray, logits: np.ndarray, T: float) -> dict:
    probs = softmax_np(logits, T)
    preds = probs.argmax(1)
    acc = float((preds == y).mean())
    ece, mce, _ = S.expected_calibration_error(y, probs, S.ECE_BINS)
    nll = nll_np(logits, y, T)
    conf = float(probs.max(1).mean())
    return {
        "split": name,
        "T": T,
        "acc": acc,
        "nll": nll,
        "ece": ece,
        "mce": mce,
        "mean_confidence": conf,
    }


def run_backbone(backbone: str, eval_test: bool) -> dict:
    ckpt_path = find_ckpt(backbone)
    print(f"\n=== {backbone} ===")
    print(f"  ckpt: {ckpt_path}")
    model, ckpt = load_model(backbone, ckpt_path)
    hparams = ckpt.get("hparams") or {}
    bs = int(hparams.get("batch_size", 64))

    _, _, _, _, val_loader, test_loader = S.make_loaders(bs)
    val_logits, val_y = collect_logits(model, val_loader)
    val_logits_np = val_logits.numpy()
    val_y_np = val_y.numpy()

    before = summarize("val", val_y_np, val_logits_np, T=1.0)
    T = fit_temperature(val_logits, val_y)
    after = summarize("val", val_y_np, val_logits_np, T=T)

    print(f"  fitted T = {T:.4f}")
    print(
        f"  VAL  before: acc={before['acc']:.4f} nll={before['nll']:.4f} "
        f"ece={before['ece']:.4f} conf={before['mean_confidence']:.4f}"
    )
    print(
        f"  VAL  after:  acc={after['acc']:.4f} nll={after['nll']:.4f} "
        f"ece={after['ece']:.4f} conf={after['mean_confidence']:.4f}"
    )

    result = {
        "backbone": backbone,
        "checkpoint": str(ckpt_path),
        "temperature": T,
        "val_before": before,
        "val_after": after,
        "note": "Apply as softmax(logits / T). Argmax / accuracy unchanged.",
    }

    if eval_test:
        test_logits, test_y = collect_logits(model, test_loader)
        tb = summarize("test", test_y.numpy(), test_logits.numpy(), T=1.0)
        ta = summarize("test", test_y.numpy(), test_logits.numpy(), T=T)
        print(
            f"  TEST before: acc={tb['acc']:.4f} nll={tb['nll']:.4f} "
            f"ece={tb['ece']:.4f} conf={tb['mean_confidence']:.4f}"
        )
        print(
            f"  TEST after:  acc={ta['acc']:.4f} nll={ta['nll']:.4f} "
            f"ece={ta['ece']:.4f} conf={ta['mean_confidence']:.4f}"
        )
        result["test_before"] = tb
        result["test_after"] = ta

    out_dir = S.RUNS_DIR / backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "temperature.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  wrote {out_json}")

    # Also patch exported meta if present
    exp_meta = S.RUNS_DIR / "exported" / backbone / f"{backbone}_export_meta.json"
    if exp_meta.is_file():
        meta = json.loads(exp_meta.read_text(encoding="utf-8"))
        meta["temperature"] = T
        meta["temperature_fit"] = {
            "val_ece_before": before["ece"],
            "val_ece_after": after["ece"],
            "val_nll_before": before["nll"],
            "val_nll_after": after["nll"],
        }
        exp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"  updated {exp_meta}")

    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=BACKBONES, default=None)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fit T for every backbone that has a checkpoint",
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Also report calibrated metrics on held-out test (T still fit on val only)",
    )
    args = parser.parse_args()

    print("DATA_ROOT", S.DATA_ROOT)
    print("classes", S.NUM_CLASSES)

    if args.all or args.backbone is None:
        backs = []
        for bb in BACKBONES:
            try:
                find_ckpt(bb)
                backs.append(bb)
            except FileNotFoundError:
                print(f"skip {bb}: no ckpt")
        if args.backbone is not None:
            backs = [args.backbone]
    else:
        backs = [args.backbone]

    results = [run_backbone(bb, args.eval_test) for bb in backs]
    summary = S.RUNS_DIR / "temperature_summary.json"
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {summary}")

    print(
        "\nInference snippet:\n"
        "  T = json.load(open('runs/<bb>/temperature.json'))['temperature']\n"
        "  probs = softmax(logits / T, dim=-1)  # argmax unchanged"
    )


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
