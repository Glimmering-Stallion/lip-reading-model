#!/usr/bin/env python3
"""Export VSRM structural twin to ONNX. Usually invoked from Rust `cargo run -- export --model <id>`."""



# Three checks for .onnx file validity:

# 1) Schema check
# python3 -c "import onnx; onnx.checker.check_model('exports/<model_id_export>/onnx/vsrm_export.onnx'); print('ok')"

# 2) Runtime load (after: pip install onnxruntime)
# python3 -c "import onnxruntime as ort; ort.InferenceSession('exports/<model_id_export>/onnx/vsrm_export.onnx', providers=['CPUExecutionProvider']); print('ok')"

# 3) Human-readable graph (long output)
# python3 -c "import onnx; m=onnx.load('exports/<model_id_export>/onnx/vsrm_export.onnx'); print(onnx.helper.printable_graph(m.graph))"



from __future__ import annotations

import argparse
import json
import inspect
import sys
from pathlib import Path

import torch

from vsrm_twin import VsrTwin



def _frame_dims_from_json(d: object) -> tuple[int, int]:
    if isinstance(d, list) and len(d) == 2:
        return int(d[0]), int(d[1])
    raise ValueError(f"expected frame_dims as [H, W], got {d!r}")



def load_vsrm_config(path: Path) -> dict:
    with path.open() as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("model_config.json must be a JSON object")
    return raw



def build_model(cfg: dict) -> VsrTwin:
    _frame_dims_from_json(cfg["frame_dims"])
    return VsrTwin(
        in_channels=int(cfg["in_channels"]),
        out_channels=int(cfg["out_channels"]),
        hidden_dim=int(cfg["hidden_dim"]),
        frame_hw=_frame_dims_from_json(cfg["frame_dims"]),
        norm_groups=int(cfg["norm_groups"]),
        vocab_size=int(cfg["vocab_size"]),
    )



def main() -> int:
    p = argparse.ArgumentParser(description="Export VSRM PyTorch twin to ONNX")
    p.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing model_config.json (same layout as train/infer)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .onnx path (Rust export passes <bundle>/onnx/vsrm_export.onnx; if omitted when run standalone: <model-dir>/vsrm_export.onnx)",
    )
    p.add_argument("--opset", type=int, default=17)
    p.add_argument(
        "--time-steps",
        type=int,
        default=96,
        help="Dummy T for static export (must exceed model receptive field)",
    )
    args = p.parse_args()

    model_dir: Path = args.model_dir
    cfg_path = model_dir / "model_config.json"
    if not cfg_path.is_file():
        print(f"error: missing {cfg_path}", file=sys.stderr)
        return 1

    out_path = args.output or (model_dir / "vsrm_export.onnx")

    cfg = load_vsrm_config(cfg_path)
    model = build_model(cfg)
    model.eval()
    h, w = _frame_dims_from_json(cfg["frame_dims"])
    n = 1
    c = int(cfg["in_channels"])
    t = args.time_steps
    dummy = torch.zeros(n, c, t, h, w, requires_grad=False)

    out_file = str(out_path)
    sig = inspect.signature(torch.onnx.export)
    with torch.no_grad():
        if "dynamo" in sig.parameters:
            torch.onnx.export(
                model,
                (dummy,),
                out_file,
                input_names=["video"],
                output_names=["logits"],
                opset_version=args.opset,
                dynamo=False,
            )
        else:
            torch.onnx.export(
                model,
                (dummy,),
                out_file,
                input_names=["video"],
                output_names=["logits"],
                opset_version=args.opset,
            )
    print(f"wrote {out_path.resolve()}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
