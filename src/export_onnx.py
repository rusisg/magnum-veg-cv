"""Phase 3.4 (Bonus 2): ONNX export (opset 17, dynamic batch) & CPU latency benchmark.

1. Exports `models/best_model.pt` to a single self-contained `models/veg_classifier.onnx`
   (opset 17, dynamic axis 0 `batch_size`, size <= 20 MB).
2. Verifies structural validity (`onnx.checker.check_model`) and numerical equivalence
   against PyTorch CPU for both single-image (`B=1`) and batched (`B=8`) inputs
   (`np.allclose(..., atol=1e-4)`).
3. Benchmarks single-image CPU inference latency (`mean`, `p50`, `p95` across 200 timed
   runs after 20 warmup runs) comparing PyTorch CPU vs ONNX Runtime `CPUExecutionProvider`.
4. Records exact benchmark metrics in `artifacts/metrics.json`.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.evaluate import load_trained_model
from src.train import seed_everything


def export_self_contained_onnx(
    model: torch.nn.Module,
    out_path: Path,
    img_size: int = 224,
    opset: int = 17,
) -> float:
    """Export PyTorch model to a single self-contained ONNX file and return size in MB."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_sidecar = Path(str(out_path) + ".data")
    if data_sidecar.exists():
        data_sidecar.unlink()

    dummy = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)
    export_kwargs: dict[str, Any] = {
        "export_params": True,
        "opset_version": opset,
        "do_constant_folding": True,
        "input_names": ["input"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(model, dummy, str(out_path), **export_kwargs)

    # Ensure weights are embedded in a single .onnx file (no external .data sidecar)
    onnx_model = onnx.load(str(out_path), load_external_data=True)
    onnx.save_model(onnx_model, str(out_path), save_as_external_data=False)
    if data_sidecar.exists():
        data_sidecar.unlink()

    onnx.checker.check_model(str(out_path))
    size_mb = out_path.stat().st_size / (1024.0 * 1024.0)
    assert size_mb <= 20.0, f"ONNX artifact {size_mb:.2f} MB exceeds 20 MB limit!"
    return size_mb


def benchmark_cpu_latency(
    model: torch.nn.Module,
    ort_session: ort.InferenceSession,
    img_size: int = 224,
    warmup_runs: int = 20,
    timed_runs: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Benchmark single-image CPU inference latency (PyTorch CPU vs ONNX Runtime CPU)."""
    rng = np.random.default_rng(seed)
    inputs_np = [
        rng.standard_normal((1, 3, img_size, img_size), dtype=np.float32)
        for _ in range(warmup_runs + timed_runs)
    ]
    inputs_pt = [torch.from_numpy(arr) for arr in inputs_np]

    # Use consistent single-core / CPU thread settings for fair single-image latency comparison
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model.eval()

    with torch.no_grad():
        for i in range(warmup_runs):
            _ = model(inputs_pt[i])

        pt_times_ms: list[float] = []
        for i in range(warmup_runs, warmup_runs + timed_runs):
            t0 = time.perf_counter()
            _ = model(inputs_pt[i])
            pt_times_ms.append((time.perf_counter() - t0) * 1000.0)

    input_name = ort_session.get_inputs()[0].name
    for i in range(warmup_runs):
        _ = ort_session.run(None, {input_name: inputs_np[i]})

    ort_times_ms: list[float] = []
    for i in range(warmup_runs, warmup_runs + timed_runs):
        t0 = time.perf_counter()
        _ = ort_session.run(None, {input_name: inputs_np[i]})
        ort_times_ms.append((time.perf_counter() - t0) * 1000.0)

    pt_arr = np.array(pt_times_ms, dtype=np.float64)
    ort_arr = np.array(ort_times_ms, dtype=np.float64)

    return {
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "pytorch_cpu_ms": {
            "mean": round(float(pt_arr.mean()), 3),
            "std": round(float(pt_arr.std()), 3),
            "p50": round(float(np.percentile(pt_arr, 50)), 3),
            "p95": round(float(np.percentile(pt_arr, 95)), 3),
        },
        "onnx_cpu_ms": {
            "mean": round(float(ort_arr.mean()), 3),
            "std": round(float(ort_arr.std()), 3),
            "p50": round(float(np.percentile(ort_arr, 50)), 3),
            "p95": round(float(np.percentile(ort_arr, 95)), 3),
        },
        "speedup_mean_x": round(float(pt_arr.mean() / max(ort_arr.mean(), 1e-6)), 2),
        "speedup_p50_x": round(float(np.percentile(pt_arr, 50) / max(np.percentile(ort_arr, 50), 1e-6)), 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MobileNetV3-Large to ONNX and benchmark CPU latency.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_model.pt"))
    parser.add_argument("--out", type=Path, default=Path("models/veg_classifier.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cpu")
    model, ckpt = load_trained_model(args.checkpoint, device)
    img_size = int(ckpt.get("img_size", 224))

    size_mb = export_self_contained_onnx(model, args.out, img_size=img_size, opset=args.opset)
    print(f"[OK] Exported {args.out} ({size_mb:.2f} MB, opset={args.opset})")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
    ort_session = ort.InferenceSession(
        str(args.out),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    input_name = ort_session.get_inputs()[0].name

    # Numerical equivalence verification (B=1 and dynamic B=8)
    rng = np.random.default_rng(args.seed)
    max_diffs: list[float] = []
    for batch_sz in (1, 8):
        test_x = rng.standard_normal((batch_sz, 3, img_size, img_size), dtype=np.float32)
        with torch.no_grad():
            pt_out = model(torch.from_numpy(test_x)).numpy()
        ort_out = ort_session.run(None, {input_name: test_x})[0]
        diff = float(np.max(np.abs(pt_out - ort_out)))
        max_diffs.append(diff)
        assert np.allclose(pt_out, ort_out, atol=1e-4), (
            f"ONNX numerical equivalence failed for batch={batch_sz}: max_diff={diff:.6e}"
        )

    max_abs_diff = max(max_diffs)
    print(f"[OK] Verified numerical equivalence (B=1 & B=8): max |PT - ONNX| = {max_abs_diff:.6e} (< 1e-4)")

    bench = benchmark_cpu_latency(
        model,
        ort_session,
        img_size=img_size,
        warmup_runs=args.warmup,
        timed_runs=args.runs,
        seed=args.seed,
    )
    pt_m = bench["pytorch_cpu_ms"]
    ox_m = bench["onnx_cpu_ms"]
    print(
        f"[BENCHMARK] PyTorch CPU:      mean={pt_m['mean']:.2f} ms | p50={pt_m['p50']:.2f} ms | p95={pt_m['p95']:.2f} ms\n"
        f"[BENCHMARK] ONNX Runtime CPU: mean={ox_m['mean']:.2f} ms | p50={ox_m['p50']:.2f} ms | p95={ox_m['p95']:.2f} ms\n"
        f"[BENCHMARK] Speedup:          {bench['speedup_mean_x']:.2f}x (mean), {bench['speedup_p50_x']:.2f}x (p50)"
    )

    metrics_path = REPO_ROOT / "artifacts" / "metrics.json"
    metrics_data: dict[str, Any] = {}
    if metrics_path.exists():
        metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))

    metrics_data["onnx_export"] = {
        "onnx_path": str(args.out),
        "opset_version": args.opset,
        "size_mb": round(size_mb, 2),
        "max_abs_diff_vs_pytorch": float(f"{max_abs_diff:.6e}"),
        "allclose_atol_1e_4": True,
        "benchmark": bench,
    }
    metrics_path.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")
    print(f"[OK] Updated {metrics_path}")


if __name__ == "__main__":
    main()
