#!/usr/bin/env python3
"""Emit PlotNeuralNet VSRM macro architecture from model_config.json.

Writes under --output-dir:
  - vsrm_export.tex — system-level flow (ResBlocks, AAP, Flatten, Linear+ReLU, TCN stacks, head)
  - tcn_export.tex — TCN detail: thin solid input Box ($C\times T$), Conv1D+LN+ReLU slabs ($C$ on pipeline; $T$ on depth; tap groups)
  - resblk_export.tex — ResBlock3D: Conv3D+GN+ReLU, Conv3D+GN, sum + skip proj, post-sum ReLU, output tensor (see residual.rs)

Optional input thumbnail for the macro diagram (left of the Video box):

- ``--input-image PATH`` — single image, copied to ``input_image.<ext>`` (highest priority).
- ``--input-bundle-dir DIR`` or a directory ``input_bundle/`` next to this script (under ``tools/tex_export/``) —
  all ``.png`` / ``.jpg`` / ``.jpeg`` files in that directory (non-recursive, sorted by name), copied to
  ``input_bundle/000.*`` under ``--output-dir``, … and drawn as an evenly spaced frame strip; omitted if the
  directory is missing or has no images.
- Otherwise ``input_image.*`` next to ``--output-dir`` or under ``--model-dir`` as before.

Requires vendored PlotNeuralNet at tools/plotneuralnet/. Syncs layers/ into output-dir for LaTeX.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent
TEX_EXPORT_DIR = Path(__file__).resolve().parent
PLOTNN_DIR = TOOLS_DIR / "plotneuralnet"
sys.path.insert(0, str(PLOTNN_DIR))

try:
    from pycore.tikzeng import (  # pyright: ignore[reportMissingImports] — vendored at tools/plotneuralnet
        to_begin,
        to_Conv,
        to_ConvConvRelu,
        to_connection,
        to_slab_depth_time_ticks,
        to_causal_temporal_tap_arrows,
        to_adjacent_slabs_span_label_below,
        to_tensor_io_box,
        to_res_sum_ball,
        to_res_skip_manhattan_xz_to_sum,
        to_res_skip_manhattan_xy_to_sum_with_proj,
        to_connection_through_sum,
        to_Fc,
        to_cor,
        to_end,
        to_generate,
        to_head,
        to_input,
        to_input_frame_strip,
        to_Pool,
        to_SoftMax,
    )
except ImportError:
    print(f"error: PlotNeuralNet not found at {PLOTNN_DIR}", file=sys.stderr)
    print(
        "  git clone https://github.com/HarisIqbal88/PlotNeuralNet.git "
        f'"{PLOTNN_DIR}"',
        file=sys.stderr,
    )
    raise SystemExit(1) from None


def load_config(model_dir: Path) -> dict:
    path = model_dir / "model_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing model_config.json: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def downsample_spatial(size: int, kernel: int = 3, pad: int = 1, stride: int = 2) -> int:
    """Match Rust ResidualBlock3D stride (1,2,2): ((size + 2*pad - kernel) // stride) + 1."""
    return ((size + 2 * pad - kernel) // stride) + 1


def vis_pair(h: int, w: int, base: float = 40.0) -> tuple[int, int]:
    m = max(h, w, 1)
    return max(8, int(base * h / m)), max(8, int(base * w / m))


def vis_square_side(h: int, w: int, base: float = 40.0) -> int:
    """Equal ``height`` and ``depth`` for 3D boxes (square ZY face); avoids landscape-only aspect ratios."""
    rh, rd = vis_pair(h, w, base)
    return max(rh, rd)


# PlotNeuralNet Box.sty default `scale=.2`: on the ZY face, vertical span is `height*scale` (cubey = ``ih``),
# horizontal span is `depth*scale` (cubez = ``id_``). Match `to_input` includegraphics to that face.
_BOX_FACE_SCALE = 0.2

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
INPUT_BUNDLE_OUT = "input_bundle"


def input_image_cm_for_input_box(ih: int, id_: int) -> tuple[float, float]:
    """``\\includegraphics`` width/height in cm to match the Video Box ZY face (same ``ih``/``id_`` as ``to_Conv``).

    With `canvas is zy plane`, width runs along ``z`` (``depth`` = ``id_``) and height along ``y`` (``height`` = ``ih``).
    """
    w_cm = id_ * _BOX_FACE_SCALE
    h_cm = ih * _BOX_FACE_SCALE
    return (w_cm, h_cm)


def braced_caption(s: str) -> str:
    """Wrap PlotNeuralNet/TikZ caption text so commas (e.g. in $[N,C,T,H,W]$) are not parsed as pgf key separators."""
    return "{" + s + "}"


def tex_hw_mbox(h: int, w: int) -> str:
    """Unbreakable $H\\times W$ on depth edges (avoids awkward line breaks inside \\times)."""
    return "\\mbox{$" + str(h) + "\\times" + str(w) + "$}"


def tex_vocab_zlabel(vs: int) -> str:
    """Single-line vocab size on depth edge"""
    return "\\mbox{$\\vert\\mathcal{V}\\vert{=}" + str(vs) + "$}"


def caption_parbox(lines: list[str], width: str = "2.4cm", *, small: bool = False) -> str:
    """Forced newlines inside a \\parbox (safe inside PlotNeuralNet edge captions; avoids bad wraps mid-equation)."""
    sz = r"\footnotesize " if small else ""
    body = r"\\".join(lines)
    inner = rf"\parbox{{{width}}}{{\raggedright {sz}{body}}}"
    return braced_caption(inner)


def parse_vsrm(cfg: dict) -> dict:
    """Numeric fields aligned with VsrModelConfig / VsrModel::new."""
    ic = int(cfg["in_channels"])
    oc = int(cfg["out_channels"])
    hd = int(cfg["hidden_dim"])
    fd = cfg["frame_dims"]
    if not isinstance(fd, list) or len(fd) != 2:
        raise ValueError(f"frame_dims must be [H, W], got {fd!r}")
    fh, fw = int(fd[0]), int(fd[1])
    vs = int(cfg["vocab_size"])
    ng = int(cfg["norm_groups"])
    tcn_kernel = int(cfg.get("tcn_kernel_size", 3))
    tcn_layers = int(cfg.get("tcn_layers", 3))
    tcn_dropout = float(cfg.get("tcn_dropout_prob", 0.1))

    c1, c2, c3 = oc, 2 * oc, 4 * oc
    h1, w1 = downsample_spatial(fh), downsample_spatial(fw)
    h2, w2 = downsample_spatial(h1), downsample_spatial(w1)
    h3, w3 = downsample_spatial(h2), downsample_spatial(w2)

    return {
        "ic": ic,
        "oc": oc,
        "hd": hd,
        "fh": fh,
        "fw": fw,
        "vs": vs,
        "ng": ng,
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "h1": h1,
        "w1": w1,
        "h2": h2,
        "w2": w2,
        "h3": h3,
        "w3": w3,
        "tcn_kernel": tcn_kernel,
        "tcn_layers": tcn_layers,
        "tcn_dropout": tcn_dropout,
        # Dilation of the “middle” block in the stack ($2^i$ in code); VSRM uses 3 layers → $2^1$.
        "tcn_dilation_exemplar": 1 << (tcn_layers // 2),
    }


def build_macro_arch(
    p: dict,
    *,
    input_includegraphics: str | None = None,
    input_bundle_frames: list[str] | None = None,
) -> list[str]:
    """System-level VSRM (`vsrm.rs`): RB×3 → AAP → Flatten → Linear+ReLU → TCN×2 → FC → logits.

    Optional ``input_includegraphics`` or non-empty ``input_bundle_frames`` (mutually exclusive in practice): input thumbnails
    left of the Video tensor; no arrow to the input box. A bundle draws several frames evenly in ``x`` before the box,
    each with the same ``\\includegraphics`` width/height in cm as the Video box ZY face (``input_image_cm_for_input_box``).

    Caption pattern: ``layer name`` + newline + ``$C:\\mathrm{in}\\to\\mathrm{out}$`` (or ``$D$`` / ``$|\\mathcal{V}|$``
    where that matches the tensor). Time $T$ is implicit along the pipeline unless stated.
    """
    ic, hd, fh, fw, vs = p["ic"], p["hd"], p["fh"], p["fw"], p["vs"]
    tcn_layers = p["tcn_layers"]
    c1, c2, c3 = p["c1"], p["c2"], p["c3"]
    h1, w1, h2, w2, h3, w3 = p["h1"], p["w1"], p["h2"], p["w2"], p["h3"], p["w3"]
    flat_pre_mlp = c3 * 16

    ih, id_ = vis_pair(fh, fw, 45.0)
    r1h, r1d = vis_pair(h1, w1, 35.0)
    r2h, r2d = vis_pair(h2, w2, 25.0)
    r3h, r3d = vis_pair(h3, w3, 15.0)

    cap_in = caption_parbox([r"\mbox{Input}", f"$C:{ic}$"], "1.9cm")
    cap_rb1 = caption_parbox(["ResBlock3D", f"$C:{ic}\\to{c1}$"], "2.5cm")
    cap_rb2 = caption_parbox(["ResBlock3D", f"$C:{c1}\\to{c2}$"], "2.5cm")
    cap_rb3 = caption_parbox(["ResBlock3D", f"$C:{c2}\\to{c3}$"], "2.5cm")
    cap_pool = caption_parbox(["AAP"], "1.2cm")
    # \mbox{...} on dim lines: \parbox+\raggedright otherwise breaks at \to / colons.
    cap_flat = caption_parbox(
        [r"\mbox{Flatten}", f"\\mbox{{$C\\times4\\times4\\to{flat_pre_mlp}$}}"],
        "2.4cm",
    )
    cap_proj = caption_parbox(
        ["Linear+ReLU", f"\\mbox{{${flat_pre_mlp}\\to D:{hd}$}}"],
        "2.6cm",
    )
    # \mbox keeps “(N blocks)” from breaking between the number and “blocks”.
    cap_tcn = caption_parbox(
        [rf"\mbox{{TCN ({tcn_layers} blocks)}}", f"\\mbox{{$D:{hd}\\to {hd}$}}"],
        "2.5cm",
    )
    cap_fc = caption_parbox(
        ["FC", f"\\mbox{{$D:{hd}\\to |\\mathcal{{V}}|:{vs}$}}"],
        "2.2cm",
    )
    cap_out = caption_parbox(["logits", f"$|\\mathcal{{V}}|:{vs}$"], "1.9cm")

    # Backend: plain Box for input / pool / flatten; RightBandedBox for rb1–3 (composite), proj (Linear+ReLU),
    # and tcn1–2 (macro policy: band = post-weight ops or composite internal structure).
    input_pipe_w = 1.5
    slab_h = input_pipe_w
    rb3_pipe_sum = 16 + 16  # keep equal to rb3 ``width=(16, 16)`` cubex sum below
    tcn_fc_z = max(1, int(round(rb3_pipe_sum * 0.6)))
    logits_z = max(8, rb3_pipe_sum // 3)
    tcn_ribbon_w = 28
    proj_w, proj_h = input_pipe_w, slab_h
    proj_z = max(tcn_fc_z + 1, int(round(rb3_pipe_sum * 2.0)))

    arch: list[str] = [
        to_head("."),
        to_cor(),
        to_begin(),
    ]
    if input_bundle_frames:
        input_w_cm, input_h_cm = input_image_cm_for_input_box(ih, id_)
        arch.append(
            to_input_frame_strip(
                input_bundle_frames,
                width_cm=input_w_cm,
                height_cm=input_h_cm,
            )
        )
    elif input_includegraphics:
        input_w_cm, input_h_cm = input_image_cm_for_input_box(ih, id_)
        arch.append(
            to_input(
                input_includegraphics,
                to="(-1.2,0,0)",
                width=input_w_cm,
                height=input_h_cm,
                name="input_in",
            )
        )
    arch.extend(
        [
        to_Conv(
            "input",
            s_filer=tex_hw_mbox(fh, fw),
            n_filer=ic,
            offset="(0,0,0)",
            to="(0,0,0)",
            width=input_pipe_w,
            height=ih,
            depth=id_,
            caption=cap_in,
        ),
        to_ConvConvRelu(
            "rb1",
            s_filer=tex_hw_mbox(h1, w1),
            n_filer=(c1, c1),
            offset="(2.0,0,0)",
            to="(input-east)",
            width=(4, 4),
            height=r1h,
            depth=r1d,
            caption=cap_rb1,
        ),
        to_connection("input", "rb1"),
        to_ConvConvRelu(
            "rb2",
            s_filer=tex_hw_mbox(h2, w2),
            n_filer=(c2, c2),
            offset="(2.0,0,0)",
            to="(rb1-east)",
            width=(8, 8),
            height=r2h,
            depth=r2d,
            caption=cap_rb2,
        ),
        to_connection("rb1", "rb2"),
        to_ConvConvRelu(
            "rb3",
            s_filer=tex_hw_mbox(h3, w3),
            n_filer=(c3, c3),
            offset="(2.0,0,0)",
            to="(rb2-east)",
            width=(16, 16),
            height=r3h,
            depth=r3d,
            caption=cap_rb3,
        ),
        to_connection("rb2", "rb3"),
        to_Pool(
            "pool",
            offset="(2.0,0,0)",
            to="(rb3-east)",
            width=16,
            height=4,
            depth=4,
            opacity=0.5,
            caption=cap_pool,
            zlabel=tex_hw_mbox(4, 4),
            n_filer=c3,
        ),
        to_connection("rb3", "pool"),
        to_Conv(
            "flatten",
            s_filer=rf"${flat_pre_mlp}$",
            n_filer=1,
            offset="(2.0,0,0)",
            to="(pool-east)",
            width=proj_w,
            height=proj_h,
            depth=proj_z,
            caption=cap_flat,
        ),
        to_connection("pool", "flatten"),
        r"""
\draw[densely dashed]
    (pool-nearnortheast) -- (flatten-nearnorthwest)
    (pool-nearsoutheast) -- (flatten-nearsouthwest)
    (pool-farsoutheast) -- (flatten-farsouthwest)
    (pool-farnortheast) -- (flatten-farnorthwest)
;
""",
        to_ConvConvRelu(
            "proj",
            s_filer=f"${hd}$",
            n_filer=(1,),
            offset="(2.0,0,0)",
            to="(flatten-east)",
            width=(proj_w,),
            height=proj_h,
            depth=tcn_fc_z,
            caption=cap_proj,
        ),
        to_connection("flatten", "proj"),
        to_ConvConvRelu(
            "tcn1",
            s_filer=f"${hd}$",
            n_filer=(r"$T$",),
            offset="(1.5,0,0)",
            to="(proj-east)",
            width=(tcn_ribbon_w,),
            height=slab_h,
            depth=tcn_fc_z,
            caption=cap_tcn,
        ),
        to_connection("proj", "tcn1"),
        to_ConvConvRelu(
            "tcn2",
            s_filer=f"${hd}$",
            n_filer=(r"$T$",),
            offset="(1.0,0,0)",
            to="(tcn1-east)",
            width=(tcn_ribbon_w,),
            height=slab_h,
            depth=tcn_fc_z,
            caption=cap_tcn,
        ),
        to_connection("tcn1", "tcn2"),
        to_Fc(
            "fc",
            n_in=hd,
            offset="(1.8,0,0)",
            to="(tcn2-east)",
            width=input_pipe_w,
            height=slab_h,
            depth=tcn_fc_z,
            caption=cap_fc,
            pipeline_xlabel=1,
        ),
        to_connection("tcn2", "fc"),
        to_SoftMax(
            "out",
            s_filer=tex_vocab_zlabel(vs),
            offset="(2.0,0,0)",
            to="(fc-east)",
            width=input_pipe_w,
            height=slab_h,
            depth=logits_z,
            caption=cap_out,
        ),
        to_connection("fc", "out"),
        to_end(),
        ]
    )
    return arch


def build_tcn_arch(p: dict) -> list[str]:
    """Three TCN blocks × two Conv1D+LN+ReLU slabs each (TCN detail figure; matches ``tcn.rs``).

    Within each block: tap groups along ``$T$`` between the two slabs. Between blocks: thick
    ``to_connection`` from each block's second slab to the next block's first slab.

    Time grid ``tap_steps`` and slab ``depth`` are chosen together (approach A): a shorter discrete
    ``T`` axis and proportionally shallower boxes so the drawn causal taps fill the visible depth
    without a long empty ``T`` tail.
    """
    _ = p["hd"]

    input_pipe_w = 1.5
    slab_h = input_pipe_w
    # RightBandedBox default ``scale`` in plotneuralnet/layers/RightBandedBox.sty (pipeline extent = width * scale).
    _rb_x_scale = 0.2
    rb3_pipe_sum = 16 + 16
    tcn_depth_full = max(48, int(round(rb3_pipe_sum * 2)))
    w_tcn = 10

    tap_steps = 15
    tap_groups = None  # one tap group per output time index (2..tap_steps); omit legs with source index < 0
    tcn_depth_z = max(40, int(round(tcn_depth_full * tap_steps / 20)))

    dim_line = f"\\mbox{{$C\\to C$}}"
    cap_stage = caption_parbox(
        [
            r"\mbox{Conv1D+}",
            r"\mbox{LN+}",
            r"\mbox{ReLU}",
            dim_line,
        ],
        "2.5cm",
    )

    z_time = r"$\leftarrow T$"
    x_c = (r"$C$",)
    cap_tcn_in = caption_parbox([r"\mbox{Input}", r"\mbox{Tensor}", r"\mbox{$C\times T$}"], "2.2cm")
    cap_tcn_out = caption_parbox([r"\mbox{Output}", r"\mbox{Tensor}", r"\mbox{$C\times T$}"], "2.2cm")
    tcn_intra_gap = 1.5  # horizontal offset between the two stage slabs inside each TCN block
    # Between TCN blocks: room for residual ``+`` and long feedforward segment.
    tcn_feedforward_gap = 3.0
    # Input / output tensors: tighter gap to first/last conv than ``tcn_feedforward_gap``.
    tcn_endcap_gap = 2.75
    tcn_in_w = w_tcn * 0.66
    tcn_in_shift_x = -(tcn_endcap_gap + tcn_in_w * _rb_x_scale)
    # First-block identity skip: step along ``z`` (xz canvas) before routing to ``tcn_res_12``.
    tcn_res_skip_z = 7.0
    # Manhattan skip final segment meets each ``+`` ball at ``-{near}`` or ``-{far}`` (see ``Ball.sty``); per-ball below.
    # ``tcn_res_12`` only: skip in at ``-far``; skip toward ``tcn_res_23`` starts from ``-near`` (not center).
    tcn_skip_12_sum_join = "far"
    tcn_skip_from_12_origin = "tcn_res_12-near"
    # ``tcn_res_23`` only: skip in at ``-near``; skip toward ``tcn_res_3out`` starts from ``-far``.
    tcn_skip_23_sum_join = "near"
    tcn_skip_from_23_origin = "tcn_res_23-far"
    # ``tcn_res_3out`` only: skip in at ``-far``.
    tcn_skip_3out_sum_join = "far"
    # Extra caption drop under all TCN slabs (RightBandedBox/Box default = 30); raise for clearance over residual strokes.
    tcn_caption_yshift_pt = 62
    # TCN Block N row: larger base than slab captions so titles sit further below the pipeline.
    tcn_super_label_below_pt = 74.0 + (tcn_caption_yshift_pt - 30)
    # Horizontal caption nudge (pipeline x): negative = toward west (visual left under 3D tilt).
    tcn_slab_caption_xshift_pt = -14  # stage / I/O tensor captions under each slab
    tcn_block_title_xshift_pt = -60  # ``TCN Block N`` row only (``to_adjacent_slabs_span_label_below``)

    return [
        to_head("."),
        to_cor(),
        to_begin(),
        to_ConvConvRelu(
            "tcn_b1_c1",
            s_filer="",
            n_filer=x_c,
            offset="(0,0,0)",
            to="(0,0,0)",
            width=(w_tcn,),
            height=slab_h,
            depth=tcn_depth_z,
            caption=cap_stage,
            caption_yshift=tcn_caption_yshift_pt,
            caption_xshift=tcn_slab_caption_xshift_pt,
        ),
        to_tensor_io_box(
            "tcn_in",
            caption=cap_tcn_in,
            offset=f"({tcn_in_shift_x},0,0)",
            to="(tcn_b1_c1-west)",
            width=tcn_in_w,
            height=slab_h,
            depth=tcn_depth_z,
            xlabel=r"$C$",
            zlabel=z_time,
            caption_yshift=tcn_caption_yshift_pt,
            caption_xshift=tcn_slab_caption_xshift_pt,
        ),
        to_connection("tcn_in", "tcn_b1_c1"),
        to_ConvConvRelu(
            "tcn_b1_c2",
            s_filer=z_time,
            zlabel_pos=1,
            n_filer=x_c,
            offset=f"({tcn_intra_gap},0,0)",
            to="(tcn_b1_c1-east)",
            width=(w_tcn,),
            height=slab_h,
            depth=tcn_depth_z,
            caption=cap_stage,
            caption_yshift=tcn_caption_yshift_pt,
            caption_xshift=tcn_slab_caption_xshift_pt,
        ),
        to_ConvConvRelu(
            "tcn_b2_c1",
            s_filer="",
            n_filer=x_c,
            offset=f"({tcn_feedforward_gap},0,0)",
            to="(tcn_b1_c2-east)",
            width=(w_tcn,),
            height=slab_h,
            depth=tcn_depth_z,
            caption=cap_stage,
            caption_yshift=tcn_caption_yshift_pt,
            caption_xshift=tcn_slab_caption_xshift_pt,
        ),
        to_ConvConvRelu(
            "tcn_b2_c2",
            s_filer=z_time,
            zlabel_pos=1,
            n_filer=x_c,
            offset=f"({tcn_intra_gap},0,0)",
            to="(tcn_b2_c1-east)",
            width=(w_tcn,),
            height=slab_h,
            depth=tcn_depth_z,
            caption=cap_stage,
            caption_yshift=tcn_caption_yshift_pt,
            caption_xshift=tcn_slab_caption_xshift_pt,
        ),
        to_ConvConvRelu(
            "tcn_b3_c1",
            s_filer="",
            n_filer=x_c,
            offset=f"({tcn_feedforward_gap},0,0)",
            to="(tcn_b2_c2-east)",
            width=(w_tcn,),
            height=slab_h,
            depth=tcn_depth_z,
            caption=cap_stage,
            caption_yshift=tcn_caption_yshift_pt,
            caption_xshift=tcn_slab_caption_xshift_pt,
        ),
        to_ConvConvRelu(
            "tcn_b3_c2",
            s_filer=z_time,
            zlabel_pos=1,
            n_filer=x_c,
            offset=f"({tcn_intra_gap},0,0)",
            to="(tcn_b3_c1-east)",
            width=(w_tcn,),
            height=slab_h,
            depth=tcn_depth_z,
            caption=cap_stage,
            caption_yshift=tcn_caption_yshift_pt,
            caption_xshift=tcn_slab_caption_xshift_pt,
        ),
        to_tensor_io_box(
            "tcn_out",
            caption=cap_tcn_out,
            offset=f"({tcn_endcap_gap},0,0)",
            to="(tcn_b3_c2-east)",
            width=tcn_in_w,
            height=slab_h,
            depth=tcn_depth_z,
            xlabel=r"$C$",
            zlabel=z_time,
            caption_yshift=tcn_caption_yshift_pt,
            caption_xshift=tcn_slab_caption_xshift_pt,
        ),
        to_slab_depth_time_ticks("tcn_in", tap_steps),
        to_slab_depth_time_ticks("tcn_b1_c1", tap_steps),
        to_slab_depth_time_ticks("tcn_b1_c2", tap_steps),
        to_slab_depth_time_ticks("tcn_b2_c1", tap_steps),
        to_slab_depth_time_ticks("tcn_b2_c2", tap_steps),
        to_slab_depth_time_ticks("tcn_b3_c1", tap_steps),
        to_slab_depth_time_ticks("tcn_b3_c2", tap_steps),
        to_slab_depth_time_ticks("tcn_out", tap_steps),
        to_causal_temporal_tap_arrows(
            "tcn_b1_c1",
            "tcn_b1_c2",
            tap_steps,
            tap_groups,
            dilation=1,
            emphasize_front_ranks=(1, 3, 5, 7, 9, 11, 13),
        ),
        to_res_sum_ball("tcn_res_12", "tcn_b1_c2-east", "tcn_b2_c1-west"),
        to_res_skip_manhattan_xz_to_sum(
            "tcn_res_12",
            "tcn_in-east",
            "tcn_b1_c1-west",
            step_z=tcn_res_skip_z,
            sum_join=tcn_skip_12_sum_join,
        ),
        to_connection_through_sum("tcn_b1_c2", "tcn_res_12", "tcn_b2_c1"),
        to_causal_temporal_tap_arrows(
            "tcn_b2_c1",
            "tcn_b2_c2",
            tap_steps,
            tap_groups,
            dilation=2,
            emphasize_front_ranks=(1, 5, 9),
        ),
        to_res_sum_ball("tcn_res_23", "tcn_b2_c2-east", "tcn_b3_c1-west"),
        to_res_skip_manhattan_xz_to_sum(
            "tcn_res_23",
            "tcn_res_12-east",
            "tcn_b2_c1-west",
            step_z=tcn_res_skip_z,
            coord_prefix="tcn_skip_res_23",
            z_first_leg_sign=1,
            origin=tcn_skip_from_12_origin,
            sum_join=tcn_skip_23_sum_join,
        ),
        to_connection_through_sum("tcn_b2_c2", "tcn_res_23", "tcn_b3_c1"),
        to_causal_temporal_tap_arrows(
            "tcn_b3_c1",
            "tcn_b3_c2",
            tap_steps,
            tap_groups,
            dilation=4,
            emphasize_front_ranks=(1,),
        ),
        to_res_sum_ball("tcn_res_3out", "tcn_b3_c2-east", "tcn_out-west"),
        to_res_skip_manhattan_xz_to_sum(
            "tcn_res_3out",
            "tcn_res_23-east",
            "tcn_b3_c1-west",
            step_z=tcn_res_skip_z,
            coord_prefix="tcn_skip_res_3out",
            origin=tcn_skip_from_23_origin,
            sum_join=tcn_skip_3out_sum_join,
        ),
        to_connection_through_sum("tcn_b3_c2", "tcn_res_3out", "tcn_out"),
        to_adjacent_slabs_span_label_below("tcn_b1_c1", "tcn_b1_c2", label=r"\textbf{TCN Block 1}", below_front_pt=tcn_super_label_below_pt, xshift_pt=tcn_block_title_xshift_pt),
        to_adjacent_slabs_span_label_below("tcn_b2_c1", "tcn_b2_c2", label=r"\textbf{TCN Block 2}", below_front_pt=tcn_super_label_below_pt, xshift_pt=tcn_block_title_xshift_pt),
        to_adjacent_slabs_span_label_below("tcn_b3_c1", "tcn_b3_c2", label=r"\textbf{TCN Block 3}", below_front_pt=tcn_super_label_below_pt, xshift_pt=tcn_block_title_xshift_pt),
        to_end(),
    ]


def build_resblk_arch(p: dict) -> list[str]:
    """Single ResBlock3D slice (``residual.rs``): ``Conv3D+GN+ReLU`` then ``Conv3D+GN``, sum + skip proj, ``ReLU``, output tensor.

    Matches ``activation::relu(x.add(residual))``: nonlinearity after the residual add, not after the second conv.
    Captions use symbolic ``C_{\mathrm{in}}``, ``C_{\mathrm{out}}``, ``H``, ``W``, ``T``, and stride ``S`` on
    the downsampled path (VSRM spatial stride 2; labels stay generic).
    Box geometry: ``fh``, ``fw`` (``frame_dims``) → larger square-faced input; ``h1``, ``w1`` (``downsample_spatial``
    of ``fh``, ``fw``) → smaller conv slabs and output—matching one strided conv step.
    Depth labels: ``H\times W`` on input; ``H/\mathrm{S}\times W/\mathrm{S}`` on convs, proj, post-sum ReLU, and output.
    Pipeline ``xlabel``: input ``C_{\mathrm{in}}``; first conv, skip proj, conv2, and output ``C_{\mathrm{out}}``; post-sum ReLU none.
    Output ``Box`` uses ``zlabel pos=0.5``. Skip path: ``to_res_skip_manhattan_xy_to_sum_with_proj`` after main ``to_connection`` (skip proj uses ``proj_caption_anchor="ne"`` so the caption clears conv1 below).
    """
    fh, fw = p["fh"], p["fw"]
    h1, w1 = p["h1"], p["w1"]

    side_in = vis_square_side(fh, fw, 35.0)
    side_out_base = vis_square_side(h1, w1, 35.0)
    # ``vis_square_side`` can match ``side_in`` when aspect ratio is unchanged; tie-break with numeric H/W ratio.
    scale_hw = min(h1 / max(fh, 1), w1 / max(fw, 1))
    side_out = max(8, min(side_out_base, int(round(side_in * scale_hw))))
    w_conv = 4
    # Uniform x-shift from each node’s ``-east`` to the next slab (input→conv1→conv2→out/sum gap).
    rb_x_gap = 3.0
    rb_in_w = w_conv * 0.66

    z_hw_sym = r"\mbox{$H\times W$}"
    z_path_sym = r"\mbox{$H/\mathrm{S}\times W/\mathrm{S}$}"
    out_spatial_tex = r"H/\mathrm{S}\times W/\mathrm{S}"
    cap_rb_in = caption_parbox(
        [
            r"\mbox{Input}",
            r"\mbox{Tensor}",
            r"\mbox{$C_{\mathrm{in}}\times T\times H\times W$}",
        ],
        "2.3cm",
    )
    cap_rb_out = caption_parbox(
        [
            r"\mbox{Output}",
            r"\mbox{Tensor}",
            rf"\mbox{{$C_{{\mathrm{{out}}}}\times T\times {out_spatial_tex}$}}",
        ],
        "2.3cm",
    )
    cap_conv1 = caption_parbox(
        [
            r"\mbox{Conv3D+}",
            r"\mbox{GN+}",
            r"\mbox{ReLU}",
            r"$C_{\mathrm{in}}\to C_{\mathrm{out}}$",
        ],
        "2.5cm",
    )
    cap_conv2 = caption_parbox(
        [
            r"\mbox{Conv3D+GN}",
            r"$C_{\mathrm{out}}\to C_{\mathrm{out}}$",
        ],
        "2.5cm",
    )
    cap_relu = caption_parbox([r"\mbox{ReLU}"], "1.2cm")
    cap_skip_proj = caption_parbox(
        [
            r"\mbox{Proj}",
            r"$C_{\mathrm{in}}\to C_{\mathrm{out}}$",
        ],
        "2.5cm",
    )
    cap_y = 52
    cap_x = -12

    return [
        to_head("."),
        to_cor(),
        to_begin(),
        to_tensor_io_box(
            "rb_in",
            caption=cap_rb_in,
            offset="(0,0,0)",
            to="(0,0,0)",
            width=rb_in_w,
            height=side_in,
            depth=side_in,
            xlabel=r"$C_{\mathrm{in}}$",
            zlabel=z_hw_sym,
            zlabel_pos=0.5,
            caption_yshift=cap_y,
            caption_xshift=cap_x,
        ),
        to_ConvConvRelu(
            "rb_conv1",
            s_filer=z_path_sym,
            n_filer=(r"$C_{\mathrm{out}}$",),
            offset=f"({rb_x_gap},0,0)",
            to="(rb_in-east)",
            width=(w_conv,),
            height=side_out,
            depth=side_out,
            caption=cap_conv1,
            caption_yshift=cap_y,
            caption_xshift=cap_x,
        ),
        to_Conv(
            "rb_conv2",
            s_filer=z_path_sym,
            n_filer=r"$C_{\mathrm{out}}$",
            offset=f"({rb_x_gap},0,0)",
            to="(rb_conv1-east)",
            width= rb_in_w,
            height=side_out,
            depth=side_out,
            caption=cap_conv2,
            caption_yshift=cap_y,
            caption_xshift=cap_x,
        ),
        to_Conv(
            "rb_relu",
            s_filer=z_path_sym,
            n_filer=None,
            offset=f"({rb_x_gap},0,0)",
            to="(rb_conv2-east)",
            width=w_conv * 0.33,
            height=side_out,
            depth=side_out,
            caption=cap_relu,
            fill_tex=r"\ConvReluColor",
            caption_yshift=cap_y,
            caption_xshift=cap_x,
            opacity=0.6,
        ),
        to_tensor_io_box(
            "rb_out",
            caption=cap_rb_out,
            offset=f"({rb_x_gap},0,0)",
            to="(rb_relu-east)",
            width=rb_in_w,
            height=side_out,
            depth=side_out,
            xlabel=r"$C_{\mathrm{out}}$",
            zlabel=z_path_sym,
            zlabel_pos=0.5,
            caption_yshift=cap_y,
            caption_xshift=cap_x,
        ),
        to_res_sum_ball("rb_sum", "rb_conv2-east", "rb_relu-west"),
        to_connection("rb_in", "rb_conv1"),
        to_connection("rb_conv1", "rb_conv2"),
        to_connection_through_sum("rb_conv2", "rb_sum", "rb_relu"),
        to_connection("rb_relu", "rb_out"),
        to_res_skip_manhattan_xy_to_sum_with_proj(
            "rb_sum",
            "rb_in-east",
            "rb_conv1-west",
            step_y=5.5,
            coord_prefix="rb_skip_xy",
            y_first_leg_sign=1,
            sum_join="north",
            proj_half_width_x=0.2,
            proj_caption=cap_skip_proj,
            proj_width=(rb_in_w,),
            proj_height=side_out,
            proj_depth=side_out,
            proj_n_filer=(r"$C_{\mathrm{out}}$",),
            proj_zlabel=z_path_sym,
            proj_caption_yshift=46,
            proj_caption_xshift=-6,
            proj_caption_anchor="ne",
        ),
        to_end(),
    ]


def resolve_input_image_basename(
    model_dir: Path, out_dir: Path, input_image_arg: Path | None
) -> str | None:
    """Basename for \\includegraphics in out_dir, or None if no input asset.

    Priority: ``--input-image`` (copy to ``input_image.<ext>``), then ``model_dir/input_image.*``,
    then an existing ``out_dir/input_image.*`` (e.g. user placed file next to ``vsrm_export.tex``).
    """
    standard = ("input_image.png", "input_image.jpg", "input_image.jpeg")
    if input_image_arg is not None:
        src = input_image_arg.expanduser().resolve()
        if not src.is_file():
            raise ValueError(f"--input-image not a file: {src}")
        ext = src.suffix.lower()
        if ext not in (".png", ".jpg", ".jpeg"):
            ext = ".png"
        name = f"input_image{ext}"
        shutil.copy2(src, out_dir / name)
        return name
    for name in standard:
        mp = model_dir / name
        if mp.is_file():
            shutil.copy2(mp, out_dir / name)
            return name
    for name in standard:
        if (out_dir / name).is_file():
            return name
    return None


def list_sorted_bundle_images(d: Path) -> list[Path]:
    """Non-recursive children of ``d`` that look like raster images."""

    if not d.is_dir():
        return []
    out = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(out, key=lambda p: p.name.lower())


def materialize_input_bundle(src_dir: Path, out_dir: Path) -> list[str]:
    """Copy bundle images into ``out_dir/input_bundle/`` with stable names; return ``.tex``-relative paths."""

    images = list_sorted_bundle_images(src_dir)
    if not images:
        return []
    dest = out_dir / INPUT_BUNDLE_OUT
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    rels: list[str] = []
    for i, p in enumerate(images):
        ext = p.suffix.lower()
        bn = f"{i:03d}{ext}"
        shutil.copy2(p, dest / bn)
        rels.append(f"{INPUT_BUNDLE_OUT}/{bn}")
    return rels


def resolve_input_diagram_assets(
    model_dir: Path,
    out_dir: Path,
    input_image_arg: Path | None,
    input_bundle_dir_arg: Path | None,
) -> tuple[str | None, list[str] | None]:
    """Return either a single ``input_image.*`` basename or a list of paths like ``input_bundle/000.png``.

    Strip source: ``--input-bundle-dir``, else ``TEX_EXPORT_DIR/input_bundle/`` if present, else single-image fallback.
    """

    if input_image_arg is not None:
        return (resolve_input_image_basename(model_dir, out_dir, input_image_arg), None)

    bundle_src: Path | None = None
    if input_bundle_dir_arg is not None:
        bundle_src = input_bundle_dir_arg.expanduser().resolve()
        if not bundle_src.is_dir():
            raise ValueError(f"--input-bundle-dir is not a directory: {bundle_src}")
    else:
        cand = TEX_EXPORT_DIR / INPUT_BUNDLE_OUT
        if cand.is_dir():
            bundle_src = cand

    if bundle_src is not None:
        strip = materialize_input_bundle(bundle_src, out_dir)
        if strip:
            return (None, strip)

    return (resolve_input_image_basename(model_dir, out_dir, None), None)


def sync_plotnn_layers(dest_parent: Path) -> None:
    src = PLOTNN_DIR / "layers"
    if not src.is_dir():
        print(f"warning: no {src}; LaTeX needs ./layers/ from PlotNeuralNet", file=sys.stderr)
        return
    dst = dest_parent / "layers"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"synced PlotNeuralNet layers -> {dst}")


def main() -> int:
    p = argparse.ArgumentParser(description="Export VSRM PlotNeuralNet macro diagram from model_config.json")
    p.add_argument("--model-dir", type=Path, required=True, help="Directory containing model_config.json")
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for .tex files and synced layers/ (e.g. exports/<id>_export/tex)",
    )
    p.add_argument(
        "--no-sync-layers",
        action="store_true",
        help="Do not copy plotneuralnet/layers into output-dir",
    )
    p.add_argument(
        "--input-image",
        type=Path,
        default=None,
        help="input crop image to copy to output-dir as input_image.<ext> and show left of Video block",
    )
    p.add_argument(
        "--input-bundle-dir",
        type=Path,
        default=None,
        help="Frame images (.png/.jpg/.jpeg); if omitted, uses tools/tex_export/input_bundle/ when that dir exists",
    )
    args = p.parse_args()

    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        print(f"error: model-dir is not a directory: {model_dir}", file=sys.stderr)
        return 1

    try:
        cfg = load_config(model_dir)
        pr = parse_vsrm(cfg)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        input_bn, input_strip = resolve_input_diagram_assets(
            model_dir, out_dir, args.input_image, args.input_bundle_dir
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if input_strip:
        print(f"input frame strip in diagram: {len(input_strip)} images -> {INPUT_BUNDLE_OUT}/")
    elif input_bn:
        print(f"input crop in diagram: {input_bn}")

    exports: list[tuple[str, list[str]]] = [
        (
            "vsrm_export.tex",
            build_macro_arch(pr, input_includegraphics=input_bn, input_bundle_frames=input_strip),
        ),
        ("tcn_export.tex", build_tcn_arch(pr)),
        ("resblk_export.tex", build_resblk_arch(pr)),
    ]
    for fname, arch in exports:
        path = out_dir / fname
        to_generate(arch, str(path))
        print(f"wrote {path}")

    if not args.no_sync_layers:
        sync_plotnn_layers(out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
