"""Re-test every exported format for EfficientNet B0/B1/B2 on the held-out test set.

Runtimes:
  - PyTorch .pth / TorchScript .pt  → CUDA (torch)
  - ONNX (.onnx)                     → onnxruntime CUDAExecutionProvider (GPU)
  - TFLite (.tflite / float16)       → TensorFlow Lite Interpreter on CPU

Exports already bake temperature (calibrated_logits = raw / T). Metrics use
softmax(output) — no extra T step.

Usage:
  python test_exported_formats.py
  python test_exported_formats.py --backbones efficientnet_b0 --formats onnx tflite
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import export_models as E
import script1 as S
import test_best_model as T

BACKBONES = ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"]
FORMATS_DEFAULT = ["pth", "pt", "onnx", "tflite", "tflite_float16"]
EXPORT_ROOT = S.RUNS_DIR / "exported"
OUT_ROOT = S.RUNS_DIR / "test_exported"


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z, dtype=np.float64)
    return exp / exp.sum(axis=1, keepdims=True)


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    return softmax_np(logits.astype(np.float64))


# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------

def resolve_artifacts(backbone: str) -> dict[str, Path | None]:
    d = EXPORT_ROOT / backbone
    mapping = {
        "pth": d / f"{backbone}.pth",
        "pt": d / f"{backbone}.pt",
        "onnx": d / f"{backbone}.onnx",
        "tflite": d / f"{backbone}.tflite",
        "tflite_float16": d / f"{backbone}_float16.tflite",
    }
    return {k: (p if p.is_file() else None) for k, p in mapping.items()}


# ---------------------------------------------------------------------------
# Predictors (each yields logits on a batch of NCHW float32 numpy / torch)
# ---------------------------------------------------------------------------

class Predictor:
    name: str
    device_label: str

    def predict_batch(self, x_nchw: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class PthPredictor(Predictor):
    """TemperatureScaledModel loaded from export .pth, run on CUDA."""

    def __init__(self, backbone: str, path: Path):
        self.name = "pth"
        self.device_label = "cuda"
        blob = torch.load(path, map_location="cpu", weights_only=False)
        T_val = float(blob.get("temperature") or E.load_temperature(backbone))
        hparams = blob.get("hparams") or dict(S.DEFAULT_HPARAMS)
        base = S.EfficientNetMLP(
            backbone,
            S.NUM_CLASSES,
            hidden_mult=float(hparams.get("hidden_mult", 0.5)),
            drop_rate=float(hparams.get("drop_rate", 0.1)),
        )
        # Prefer wrapped state; fall back to base + wrap
        sd = blob.get("model_state_dict") or {}
        model: torch.nn.Module
        if any(k.startswith("model.") for k in sd):
            model = E.TemperatureScaledModel(base, T_val)
            model.load_state_dict(sd)
        else:
            base.load_state_dict(blob.get("base_model_state_dict") or sd)
            model = E.TemperatureScaledModel(base, T_val)
        self.model = model.to(S.DEVICE).eval()

    @torch.no_grad()
    def predict_batch(self, x_nchw: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x_nchw).to(S.DEVICE, non_blocking=True)
        return self.model(x).float().cpu().numpy()

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()


class TorchScriptPredictor(Predictor):
    def __init__(self, path: Path):
        self.name = "pt"
        self.device_label = "cuda"
        self.model = torch.jit.load(str(path), map_location=S.DEVICE)
        self.model.eval()

    @torch.no_grad()
    def predict_batch(self, x_nchw: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x_nchw).to(S.DEVICE, non_blocking=True)
        out = self.model(x)
        return out.float().cpu().numpy()

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()


def _ensure_torch_cuda_dlls_on_path() -> None:
    """ORT CUDA EP needs cublas/cudnn on PATH; torch ships CUDA 12.x libs."""
    try:
        import torch as _torch

        lib = Path(_torch.__file__).resolve().parent / "lib"
        if lib.is_dir():
            lib_s = str(lib)
            path = os.environ.get("PATH", "")
            if lib_s not in path.split(os.pathsep):
                os.environ["PATH"] = lib_s + os.pathsep + path
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(lib_s)
                except (FileNotFoundError, OSError):
                    pass
    except Exception:  # noqa: BLE001
        pass


class OnnxGpuPredictor(Predictor):
    def __init__(self, path: Path):
        _ensure_torch_cuda_dlls_on_path()
        import onnxruntime as ort

        self.name = "onnx"
        self.device_label = "cuda"
        available = ort.get_available_providers()
        want = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in want if p in available]
        if "CUDAExecutionProvider" not in providers:
            raise RuntimeError(
                f"ONNX Runtime has no CUDA EP. Available={available}. "
                "Install/repair onnxruntime-gpu (CUDA 12 build) and uninstall plain onnxruntime."
            )
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Prefer CUDA; provider options help on multi-GPU / driver mismatches
        provider_opts = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            ),
            "CPUExecutionProvider",
        ]
        self.sess = ort.InferenceSession(
            str(path), sess_options=so, providers=provider_opts
        )
        used = self.sess.get_providers()
        if used[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                f"Expected CUDAExecutionProvider first, got {used}. "
                "ORT 1.27+ needs CUDA 13; with torch cu128 use onnxruntime-gpu~=1.20–1.22 "
                "and ensure torch/lib is on PATH."
            )
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        # Export used example batch=1; many graphs freeze dim0=1 (no dynamic axes).
        shape = inp.shape
        self.fixed_batch = None
        if shape and isinstance(shape[0], int) and shape[0] > 0:
            self.fixed_batch = int(shape[0])
        self.device_label = f"cuda ({used[0]})"

    def predict_batch(self, x_nchw: np.ndarray) -> np.ndarray:
        if self.fixed_batch is None or x_nchw.shape[0] == self.fixed_batch:
            return self.sess.run(None, {self.input_name: x_nchw})[0]
        # Slice to the frozen batch size (typically 1)
        outs = []
        bs = self.fixed_batch
        for i in range(0, x_nchw.shape[0], bs):
            chunk = x_nchw[i : i + bs]
            if chunk.shape[0] < bs:
                # pad last incomplete chunk then truncate
                pad = np.zeros((bs, *chunk.shape[1:]), dtype=chunk.dtype)
                pad[: chunk.shape[0]] = chunk
                out = self.sess.run(None, {self.input_name: pad})[0][: chunk.shape[0]]
            else:
                out = self.sess.run(None, {self.input_name: chunk})[0]
            outs.append(out)
        return np.concatenate(outs, axis=0)

    def close(self) -> None:
        del self.sess


class TfliteCpuPredictor(Predictor):
    """TFLite float32/float16 on CPU. Input may be NHWC or NCHW — we detect."""

    def __init__(self, path: Path, label: str):
        self.name = label
        self.device_label = "cpu"
        self.interp = self._make_interpreter(path)
        self.interp.allocate_tensors()
        inp = self.interp.get_input_details()[0]
        out = self.interp.get_output_details()[0]
        self.in_idx = inp["index"]
        self.out_idx = out["index"]
        self.in_shape = tuple(inp["shape"])
        self.in_dtype = inp["dtype"]
        # Detect layout: NHWC if last dim == 3, else NCHW
        if len(self.in_shape) == 4 and self.in_shape[-1] == 3:
            self.layout = "NHWC"
        else:
            self.layout = "NCHW"
        self.batch = int(self.in_shape[0]) if self.in_shape[0] > 0 else 1

    @staticmethod
    def _make_interpreter(path: Path):
        # num_threads must be >= 1 for both LiteRT and tf.lite
        n_threads = max(1, (os.cpu_count() or 4))
        try:
            from ai_edge_litert.interpreter import Interpreter

            return Interpreter(model_path=str(path), num_threads=n_threads)
        except Exception:
            import tensorflow as tf

            return tf.lite.Interpreter(model_path=str(path), num_threads=n_threads)

    def predict_batch(self, x_nchw: np.ndarray) -> np.ndarray:
        # Process one (or fixed) batch size expected by the model
        n = x_nchw.shape[0]
        outs = []
        for i in range(n):
            sample = x_nchw[i : i + 1]
            if self.layout == "NHWC":
                sample = np.transpose(sample, (0, 2, 3, 1))
            sample = np.ascontiguousarray(sample, dtype=self.in_dtype)
            self.interp.set_tensor(self.in_idx, sample)
            self.interp.invoke()
            outs.append(self.interp.get_tensor(self.out_idx).copy())
        logits = np.concatenate(outs, axis=0)
        # If output somehow NHWC-ish for logits it's still [N, C]
        if logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1)
        return logits.astype(np.float32)

    def close(self) -> None:
        del self.interp


def build_predictor(fmt: str, backbone: str, path: Path) -> Predictor:
    if fmt == "pth":
        return PthPredictor(backbone, path)
    if fmt == "pt":
        return TorchScriptPredictor(path)
    if fmt == "onnx":
        return OnnxGpuPredictor(path)
    if fmt in ("tflite", "tflite_float16"):
        return TfliteCpuPredictor(path, fmt)
    raise ValueError(f"Unknown format: {fmt}")


# ---------------------------------------------------------------------------
# Inference over test loader
# ---------------------------------------------------------------------------

def run_inference(pred: Predictor, loader, desc: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    all_logits, all_y = [], []
    t0 = time.perf_counter()
    for x, y in tqdm(loader, desc=desc, leave=False):
        x_np = x.numpy().astype(np.float32, copy=False)
        logits = pred.predict_batch(x_np)
        all_logits.append(np.asarray(logits, dtype=np.float32))
        all_y.append(y.numpy())
    elapsed = time.perf_counter() - t0
    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    probs = logits_to_probs(logits)
    return y_true, logits, probs, elapsed


def scalar_keys(m: dict) -> list[str]:
    return [k for k in m if not str(k).startswith("_")]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbones", nargs="+", default=BACKBONES, choices=BACKBONES)
    parser.add_argument("--formats", nargs="+", default=FORMATS_DEFAULT, choices=FORMATS_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", choices=["test", "val"], default="test")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    _ensure_torch_cuda_dlls_on_path()

    print("DATA_ROOT", S.DATA_ROOT)
    print("EXPORT_ROOT", EXPORT_ROOT)
    print("DEVICE torch", S.DEVICE, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    print("split", args.split, "batch", args.batch_size)
    print("backbones", args.backbones)
    print("formats", args.formats)

    # ORT GPU sanity
    try:
        import onnxruntime as ort

        print("onnxruntime", ort.__version__, "providers", ort.get_available_providers())
    except Exception as exc:  # noqa: BLE001
        print("onnxruntime import failed:", exc)

    _, _, _, _, val_loader, test_loader = S.make_loaders(args.batch_size)
    loader = test_loader if args.split == "test" else val_loader

    rows: list[dict] = []
    all_reports: list[dict] = []

    for backbone in args.backbones:
        arts = resolve_artifacts(backbone)
        T_val = E.load_temperature(backbone)
        print(f"\n######## {backbone}  T={T_val:.6f} (baked) ########")

        for fmt in args.formats:
            path = arts.get(fmt)
            if path is None:
                print(f"  SKIP {fmt}: artifact missing under {EXPORT_ROOT / backbone}")
                rows.append(
                    {
                        "backbone": backbone,
                        "format": fmt,
                        "status": "missing",
                        "temperature": T_val,
                    }
                )
                continue

            print(f"\n=== {backbone} / {fmt} ===")
            print(f"  path: {path}")
            try:
                pred = build_predictor(fmt, backbone, path)
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR load: {exc}")
                rows.append(
                    {
                        "backbone": backbone,
                        "format": fmt,
                        "status": f"load_error: {exc}",
                        "temperature": T_val,
                        "path": str(path),
                    }
                )
                continue

            print(f"  runtime device: {pred.device_label}")
            try:
                y_true, logits, probs, elapsed = run_inference(
                    pred, loader, desc=f"{backbone}/{fmt}"
                )
                m = T.compute_all_metrics(y_true, logits, probs)
                preds = probs.argmax(axis=1)
                per_df = T.per_class_binary_rates(y_true, preds, S.NUM_CLASSES)

                print(
                    f"  acc={m['accuracy']:.4f}  f1_macro={m['f1_macro']:.4f}  "
                    f"ece={m['ece']:.4f}  log_loss={m['log_loss']:.4f}  "
                    f"time={elapsed:.1f}s  n={m['n_test']}"
                )

                case_dir = OUT_ROOT / backbone / fmt
                case_dir.mkdir(parents=True, exist_ok=True)

                report = {
                    "backbone": backbone,
                    "format": fmt,
                    "path": str(path),
                    "split": args.split,
                    "device": pred.device_label,
                    "temperature": T_val,
                    "temperature_baked": True,
                    "inference_seconds": elapsed,
                    "images_per_second": float(m["n_test"] / elapsed) if elapsed > 0 else None,
                    "test_metrics": T._jsonable(m),
                    "reliability_bins": m["_reliability_bins"],
                    "top_confusions": m["_top_confusions"],
                    "classification_report": m["_classification_report"],
                }
                (case_dir / "report.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8"
                )
                (case_dir / "classification_report.txt").write_text(
                    m["_classification_report_text"], encoding="utf-8"
                )
                scalar = {k: v for k, v in m.items() if not str(k).startswith("_")}
                row = {
                    "backbone": backbone,
                    "format": fmt,
                    "status": "ok",
                    "device": pred.device_label,
                    "temperature": T_val,
                    "path": str(path),
                    "inference_seconds": elapsed,
                    "images_per_second": report["images_per_second"],
                    **scalar,
                }
                pd.DataFrame([row]).to_csv(case_dir / "metrics.csv", index=False)
                per_df.to_csv(case_dir / "per_class.csv", index=False)
                cm = m["_confusion_matrix"]
                pd.DataFrame(cm, index=S.CLASS_NAMES, columns=S.CLASS_NAMES).to_csv(
                    case_dir / "confusion_matrix.csv"
                )
                pd.DataFrame(
                    {
                        "y_true": y_true,
                        "y_pred": preds,
                        "y_true_name": [S.CLASS_NAMES[i] for i in y_true],
                        "y_pred_name": [S.CLASS_NAMES[i] for i in preds],
                        "confidence": probs.max(axis=1),
                        "correct": preds == y_true,
                    }
                ).to_csv(case_dir / "predictions.csv", index=False)

                rows.append(row)
                all_reports.append(report)
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR run: {exc}")
                rows.append(
                    {
                        "backbone": backbone,
                        "format": fmt,
                        "status": f"run_error: {exc}",
                        "temperature": T_val,
                        "path": str(path),
                        "device": getattr(pred, "device_label", None),
                    }
                )
            finally:
                pred.close()
                torch.cuda.empty_cache()

    # Summary tables
    summary_df = pd.DataFrame(rows)
    summary_csv = OUT_ROOT / "summary_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Compact pivot for key metrics
    ok = summary_df[summary_df.get("status", pd.Series(dtype=str)) == "ok"].copy()
    key_cols = [
        "backbone",
        "format",
        "device",
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
        "cohen_kappa",
        "mcc",
        "log_loss",
        "brier_multiclass",
        "ece",
        "mce",
        "roc_auc_ovr_macro",
        "top3_accuracy",
        "top5_accuracy",
        "inference_seconds",
        "images_per_second",
    ]
    present = [c for c in key_cols if c in ok.columns]
    compact = ok[present] if len(ok) else ok
    compact_path = OUT_ROOT / "summary_key_metrics.csv"
    compact.to_csv(compact_path, index=False)

    summary_json = OUT_ROOT / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "split": args.split,
                "batch_size": args.batch_size,
                "backbones": args.backbones,
                "formats": args.formats,
                "note": "Exports output calibrated_logits (T baked). Softmax applied in metrics.",
                "rows": json.loads(summary_df.to_json(orient="records")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n======== SUMMARY (key metrics) ========")
    if len(compact):
        with pd.option_context("display.max_rows", 100, "display.width", 200, "display.max_columns", 20):
            show = compact.copy()
            for c in ("accuracy", "f1_macro", "ece", "log_loss"):
                if c in show.columns:
                    show[c] = show[c].map(lambda v: f"{v:.4f}")
            print(show.to_string(index=False))
    else:
        print("(no successful runs)")

    print(f"\nWrote {summary_csv}")
    print(f"Wrote {compact_path}")
    print(f"Wrote {summary_json}")
    print(f"Per-case artifacts under {OUT_ROOT}/<backbone>/<format>/")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
