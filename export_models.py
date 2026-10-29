"""Export EfficientNet B0/B1/B2 with temperature baked into the graph.

Each backbone uses its own T from ``runs/temperature_summary.json`` (or
``runs/<bb>/temperature.json``). Exported graphs output **calibrated logits**
(= raw_logits / T). Deploy: ``probs = softmax(output)`` — no extra T step.

Formats: .pth, .pt (TorchScript), .onnx, .tflite (+ float16 tflite).

Usage:
  python export_models.py
  python export_models.py --backbones efficientnet_b0 efficientnet_b2
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import script1 as S

BACKBONES_DEFAULT = ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"]
IMG_SIZE = S.IMG_SIZE


class TemperatureScaledModel(nn.Module):
    """Wraps a classifier; forward returns logits / T (T as buffer, baked in graph)."""

    def __init__(self, model: nn.Module, temperature: float):
        super().__init__()
        self.model = model
        t = float(temperature)
        if t <= 0:
            raise ValueError(f"temperature must be > 0, got {t}")
        self.register_buffer("temperature", torch.tensor(t, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) / self.temperature


def find_ckpt(backbone: str) -> Path:
    for p in (
        S.RUNS_DIR / backbone / "best_overall_f1.pt",
        S.RUNS_DIR / backbone / "fixed" / "best_f1.pt",
    ):
        if p.is_file():
            return p
    raise FileNotFoundError(f"No checkpoint for {backbone}")


def load_temperature(backbone: str) -> float:
    # Prefer per-backbone file, then summary
    per = S.RUNS_DIR / backbone / "temperature.json"
    if per.is_file():
        return float(json.loads(per.read_text(encoding="utf-8"))["temperature"])
    summary = S.RUNS_DIR / "temperature_summary.json"
    if summary.is_file():
        for row in json.loads(summary.read_text(encoding="utf-8")):
            if row.get("backbone") == backbone:
                return float(row["temperature"])
    raise FileNotFoundError(
        f"No temperature for {backbone}. Run: python temperature_scaling.py --all"
    )


def load_base_model(backbone: str, ckpt_path: Path) -> tuple[nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hparams") or dict(S.DEFAULT_HPARAMS)
    n_cls = int(ckpt.get("num_classes", S.NUM_CLASSES))
    if n_cls != S.NUM_CLASSES:
        raise RuntimeError(
            f"{backbone}: ckpt num_classes={n_cls} != labels.json {S.NUM_CLASSES}"
        )
    model = S.EfficientNetMLP(
        backbone,
        n_cls,
        hidden_mult=float(hparams.get("hidden_mult", 0.5)),
        drop_rate=float(hparams.get("drop_rate", 0.1)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def export_tflite(onnx_path: Path, tflite_out: Path) -> Path:
    try:
        from onnx2tf import convert as onnx2tf_convert
    except ImportError as e:
        raise RuntimeError("onnx2tf not installed. Run: pip install onnx2tf") from e

    with tempfile.TemporaryDirectory(prefix="onnx2tf_") as tmp:
        out_folder = Path(tmp) / "tfout"
        onnx2tf_convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(out_folder),
            non_verbose=True,
            verbosity="error",
            output_signaturedefs=True,
        )
        candidates = sorted(out_folder.rglob("*float32*.tflite"))
        if not candidates:
            candidates = sorted(out_folder.rglob("*.tflite"))
        if not candidates:
            raise FileNotFoundError(f"onnx2tf produced no .tflite under {out_folder}")
        shutil.copy2(candidates[0], tflite_out)
        f16 = sorted(out_folder.rglob("*float16*.tflite"))
        if f16:
            shutil.copy2(
                f16[0],
                tflite_out.with_name(tflite_out.stem + "_float16.tflite"),
            )
    return tflite_out


def export_one(backbone: str, out_root: Path) -> dict:
    ckpt_path = find_ckpt(backbone)
    T = load_temperature(backbone)
    print(f"\n=== {backbone}  (T={T:.6f} baked) ===")
    print(f"  source: {ckpt_path}")

    base, ckpt = load_base_model(backbone, ckpt_path)
    model = TemperatureScaledModel(base, T)
    model.eval()
    hparams = ckpt.get("hparams") or {}

    out_dir = out_root / backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str | None] = {}

    meta = {
        "backbone": backbone,
        "source_checkpoint": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "val_metrics": {
            k: v
            for k, v in (ckpt.get("metrics") or {}).items()
            if not str(k).startswith("_")
        },
        "hparams": hparams,
        "num_classes": S.NUM_CLASSES,
        "class_names": S.CLASS_NAMES,
        "img_size": IMG_SIZE,
        "imagenet_mean": list(S.IMAGENET_MEAN),
        "imagenet_std": list(S.IMAGENET_STD),
        "input_layout": "NCHW",
        "input_dtype": "float32",
        "temperature": T,
        "temperature_baked": True,
        "output": "calibrated_logits (= raw_logits / T); apply softmax for probs",
    }

    example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    # --- .pth ---
    pth_path = out_dir / f"{backbone}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "base_model_state_dict": base.state_dict(),
            "backbone": backbone,
            "num_classes": S.NUM_CLASSES,
            "hparams": hparams,
            "class_names": S.CLASS_NAMES,
            "img_size": IMG_SIZE,
            "imagenet_mean": S.IMAGENET_MEAN,
            "imagenet_std": S.IMAGENET_STD,
            "epoch": ckpt.get("epoch"),
            "metrics": meta["val_metrics"],
            "temperature": T,
            "temperature_baked": True,
        },
        pth_path,
    )
    paths["pth"] = str(pth_path)
    print(f"  wrote {pth_path}")

    # --- .pt TorchScript ---
    pt_path = out_dir / f"{backbone}.pt"
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
        traced = torch.jit.freeze(traced)
    traced.save(str(pt_path))
    paths["pt"] = str(pt_path)
    print(f"  wrote {pt_path} (TorchScript, T baked)")

    ckpt_copy = out_dir / f"{backbone}_train_ckpt.pt"
    shutil.copy2(ckpt_path, ckpt_copy)
    paths["train_ckpt"] = str(ckpt_copy)

    # --- .onnx ---
    onnx_path = out_dir / f"{backbone}.onnx"
    torch.onnx.export(
        model,
        example,
        str(onnx_path),
        input_names=["input"],
        output_names=["calibrated_logits"],
        dynamo=False,
        opset_version=18,
        do_constant_folding=True,
    )
    paths["onnx"] = str(onnx_path)
    print(f"  wrote {onnx_path}")

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"input": example.numpy()})[0]
        with torch.no_grad():
            pt_out = model(example).numpy()
        max_diff = float(np.max(np.abs(ort_out - pt_out)))
        print(f"  ONNX vs PyTorch max|diff|={max_diff:.6g}")
        meta["onnx_vs_pytorch_max_abs_diff"] = max_diff
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN onnxruntime check failed: {exc}")

    # Verify T effect: calibrated ≈ raw / T
    with torch.no_grad():
        raw = base(example)
        cal = model(example)
        ratio = (raw / cal).mean().item()
        print(f"  check mean(raw/cal)≈T: {ratio:.6f} (target {T:.6f})")
        meta["raw_over_calibrated_mean"] = float(ratio)

    # --- .tflite ---
    tflite_path = out_dir / f"{backbone}.tflite"
    try:
        export_tflite(onnx_path, tflite_path)
        paths["tflite"] = str(tflite_path)
        f16 = tflite_path.with_name(tflite_path.stem + "_float16.tflite")
        if f16.is_file():
            paths["tflite_float16"] = str(f16)
        print(f"  wrote {tflite_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR TFLite export failed: {exc}")
        paths["tflite"] = None
        meta["tflite_error"] = str(exc)

    # Sidecar T (redundant but handy)
    (out_dir / "temperature.json").write_text(
        json.dumps(
            {
                "backbone": backbone,
                "temperature": T,
                "temperature_baked_in_exports": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    meta["artifacts"] = paths
    meta_path = out_dir / f"{backbone}_export_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  wrote {meta_path}")

    del model, base
    torch.cuda.empty_cache()
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=BACKBONES_DEFAULT,
        choices=["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"],
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    out_root = args.out_dir or (S.RUNS_DIR / "exported")
    out_root.mkdir(parents=True, exist_ok=True)

    print("DATA_ROOT", S.DATA_ROOT)
    print("NUM_CLASSES", S.NUM_CLASSES)
    print("out", out_root)
    print("Temperature baked into all graph exports (logits/T).")

    results = []
    for bb in args.backbones:
        results.append(export_one(bb, out_root))

    summary = out_root / "export_summary.json"
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {summary}")
    print("Deploy: probs = softmax(model(x))  # T already inside")


if __name__ == "__main__":
    main()
