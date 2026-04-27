//! CTC Loss forward lattice visualization: log-alpha grid capture, ASCII printout and SVG export of heatmap.
//!
//! Captures the forward DP log-alpha grid (`N = 1` single sample batch) and per-transition edge fractions. A shared
//! trace struct feeds both renderers so numeric values and edge sets are identical.
//!
//! **ASCII printout:** monospace heatmap with block-shade characters (`░`, `▒`, `▓`, `█`) mapping
//! log-alpha magnitude, interleaved target labels on the Y axis, timesteps on X.
//!
//! **SVG:** blue-white heatmap cells with DP transition arrows overlaid. Solid arrows mark
//! edges on at least one complete CTC alignment; dashed arrows mark paths that form an incomplete alignment.
//! Arrow opacity scales with conditional edge mass (ratio of predecessor cell's log-alpha mass to sum of
//! all predecessors' masses arriving at the current cell). A vertical colorbar legend shows the log-alpha
//! range with white-to-blue shading (with white indicating highest log-alpha and blue indicating lowest).
//!
//! SVG visual parameters (colors, fonts, spacing, etc.) are configurable with [`ForwardLatticeSvgTheme`];
//! decode-time counterpart lives in [`super::prefix_beam_viz`].
//!
//! # Quick start (fixture tests)
//!
//! 1. Edit `FIXTURE_SEQS` in the `tests` module to add or change target words/seqs.
//! 2. Run the ASCII printout:
//!    ```sh
//!    cargo test ctc_loss_lattice_ascii_printout -- --nocapture
//!    ```
//! 3. Run the SVG export (to `LRM Rust/outputs/`):
//!    ```sh
//!    cargo test ctc_loss_lattice_svg_export -- --ignored --nocapture
//!    ```
//! Codeveloped with Claude Opus 4.6.



use std::collections::HashSet;

use super::SVG_UI_FONT;
use crate::ctc::ctc_loss::CtcLoss;
use crate::utils::log_sum_exp_3_tensor;
use burn::tensor::{
    activation::log_softmax,
    backend::Backend,
    Tensor,
    Int,
};



/// Forward lattice for batch index `0`: `grid_t_s[t][s]` is the log-alpha value at time `t` and interleaved state `s`.
#[derive(Clone, Debug)]
pub struct ForwardLatticeTrace {
    grid_t_s: Vec<Vec<f32>>,
    interleaved_ids: Vec<i64>,
    input_len: usize,
    target_len: usize,
    /// Per transition `t-1 → t` (length `T-1`): edges `(s_from, s_to, frac)` where `frac` is the
    /// share of incoming log-mass at `(t, s_to)` from `(t-1, s_from)` under stay / adv1 / adv2.
    dp_edge_fractions: Vec<Vec<(usize, usize, f32)>>,
}



fn row0_curr_fwd<B: Backend>(curr_fwd: Tensor<B, 2>, intr_pad_targ_length: usize) -> Vec<f32> {
    curr_fwd
        .slice([0..1, 0..intr_pad_targ_length])
        .squeeze_dim::<1>(0)
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap()
}



/// Forward-only CTC DP with `N = 1`, mirroring `forward_no_reduction` and recording `curr_fwd` after each timestep.
fn lattice_trace_sample0_inner<B: Backend>(
    loss: &CtcLoss,
    inputs: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    input_lengths: Tensor<B, 1, Int>,
    target_lengths: Tensor<B, 1, Int>,
) -> ForwardLatticeTrace {
    let device = inputs.device();
    let [n, t, _v] = inputs.dims();
    assert_eq!(n, 1, "lattice_trace_sample0 expects batch size 1");

    let sentinel_value = -1e30;

    let orig_pad_targ_length: usize = targets.dims()[1];
    let intr_pad_targ_length = 2 * orig_pad_targ_length + 1;

    let intr_targ_lengths = target_lengths.clone() * 2 + 1;

    let log_probs = log_softmax(inputs, 2);

    let targets_intr = loss.interleave_targets_with_blanks(targets, &device);

    let interleaved_ids: Vec<i64> = targets_intr
        .clone()
        .into_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .unwrap();

    let log_probs_targets = log_probs.clone().gather(
        2,
        targets_intr
            .clone()
            .reshape([n, 1, intr_pad_targ_length])
            .expand([n, t, intr_pad_targ_length]),
    );

    let mut curr_fwd: Tensor<B, 2> =
        Tensor::full([n, intr_pad_targ_length], sentinel_value, &device);

    let (skip_1_mask, skip_2_mask) =
        loss.compute_skip_validity_masks(targets_intr.clone(), &device);

    let time_mask = Tensor::<B, 1, Int>::arange(0..t as i64, &device)
        .reshape([1, t])
        .expand([n, t])
        .lower(input_lengths.clone().reshape([n, 1]));

    let length_mask = Tensor::<B, 1, Int>::arange(0..(intr_pad_targ_length as i64), &device)
        .expand([n, intr_pad_targ_length])
        .lower(intr_targ_lengths.clone().reshape([n, 1]));

    let mut grid_t_s: Vec<Vec<f32>> = Vec::with_capacity(t);
    let mut dp_edge_fractions: Vec<Vec<(usize, usize, f32)>> = Vec::with_capacity(t.saturating_sub(1));

    for i in 0..2 {
        let log_prob_0_i = log_probs_targets
            .clone()
            .slice([0..n, 0..1, i..(i + 1)])
            .reshape([n, 1]);
        curr_fwd = curr_fwd.slice_assign([0..n, i..(i + 1)], log_prob_0_i);
    }

    grid_t_s.push(row0_curr_fwd(curr_fwd.clone(), intr_pad_targ_length));

    for t_idx in 1..t {
        let time_mask_t = time_mask
            .clone()
            .slice([0..n, t_idx..(t_idx + 1)])
            .expand([n, intr_pad_targ_length]);

        let log_probs_t = log_probs_targets
            .clone()
            .slice([0..n, t_idx..(t_idx + 1), 0..intr_pad_targ_length])
            .squeeze_dim(1)
            .mask_where(
                time_mask_t.clone().bool_not(),
                Tensor::full([n, intr_pad_targ_length], sentinel_value, &device),
            );

        let stay = curr_fwd.clone();
        let adv_1 = curr_fwd
            .clone()
            .roll_dim(-1, 1)
            .mask_fill(skip_1_mask.clone().bool_not(), sentinel_value);
        let adv_2 = curr_fwd
            .clone()
            .roll_dim(-2, 1)
            .mask_fill(skip_2_mask.clone().bool_not(), sentinel_value);

        let next_fwd = (log_sum_exp_3_tensor(stay.clone(), adv_1.clone(), adv_2.clone())
            + log_probs_t.clone())
            .mask_fill(length_mask.clone().bool_not(), sentinel_value);

        let stay_v = row0_curr_fwd(stay, intr_pad_targ_length);
        let adv1_v = row0_curr_fwd(adv_1, intr_pad_targ_length);
        let adv2_v = row0_curr_fwd(adv_2, intr_pad_targ_length);
        let next_v = row0_curr_fwd(next_fwd.clone(), intr_pad_targ_length);
        let neg_cut = sentinel_value + 1000.0;

        let mut edges: Vec<(usize, usize, f32)> = Vec::new();
        for s in 0..intr_pad_targ_length {
            if !(next_v[s].is_finite() && next_v[s] > neg_cut) {
                continue;
            }
            let l_in = log_sum_exp_3_f32_edge(stay_v[s], adv1_v[s], adv2_v[s], neg_cut);
            if !l_in.is_finite() || l_in <= neg_cut {
                continue;
            }
            if stay_v[s].is_finite() && stay_v[s] > neg_cut {
                let frac = ((stay_v[s] - l_in) as f64).exp();
                if frac.is_finite() && frac > 0.0 {
                    edges.push((s, s, frac as f32));
                }
            }
            if s >= 1 && adv1_v[s].is_finite() && adv1_v[s] > neg_cut {
                let frac = ((adv1_v[s] - l_in) as f64).exp();
                if frac.is_finite() && frac > 0.0 {
                    edges.push((s - 1, s, frac as f32));
                }
            }
            if s >= 2 && adv2_v[s].is_finite() && adv2_v[s] > neg_cut {
                let frac = ((adv2_v[s] - l_in) as f64).exp();
                if frac.is_finite() && frac > 0.0 {
                    edges.push((s - 2, s, frac as f32));
                }
            }
        }
        dp_edge_fractions.push(edges);

        curr_fwd = curr_fwd.mask_where(time_mask_t, next_fwd);
        grid_t_s.push(row0_curr_fwd(curr_fwd.clone(), intr_pad_targ_length));
    }

    let input_len = input_lengths
        .clone()
        .into_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .unwrap()[0] as usize;
    let target_len = target_lengths
        .clone()
        .into_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .unwrap()[0] as usize;

    ForwardLatticeTrace {
        grid_t_s,
        interleaved_ids,
        input_len,
        target_len,
        dp_edge_fractions,
    }
}



/// Stable `log(exp(a)+exp(b)+exp(c))` over the three incoming paths, ignoring terms at or below `neg_cut` like the forward DP.
fn log_sum_exp_3_f32_edge(a: f32, b: f32, c: f32, neg_cut: f32) -> f32 {
    let m = a.max(b).max(c);
    if m <= neg_cut {
        return f32::NEG_INFINITY;
    }
    let mut sum = 0f64;
    for x in [a, b, c] {
        if x.is_finite() && x > neg_cut {
            sum += ((x - m) as f64).exp();
        }
    }
    if sum <= 0.0 {
        f32::NEG_INFINITY
    } else {
        m + sum.ln() as f32
    }
}



const SENTINEL_CUTOFF: f32 = -1e20_f32;
const ASCII_SHADES: [&str; 5] = [" ", "░", "▒", "▓", "█"];

// heatmap cells and vertical colorbar share this rgb mapping

/// Color for masked or sentinel grid cells: blue-tinted and darker than any in-range log-alpha cell.
const FORWARD_LATTICE_INVALID_CELL_RGB: (u8, u8, u8) = (60, 60, 120);

/// Blue channel at normalized `u = 0`; added blue at `u = 1` scales with [`FORWARD_LATTICE_HEATMAP_B1_DELTA`].
const FORWARD_LATTICE_HEATMAP_B0: f32 = 220.0;
const FORWARD_LATTICE_HEATMAP_B1_DELTA: f32 = 35.0;

/// Maps `u` in \[0, 1\] (low `vmin` to high `vmax`) to RGB: R and G follow a gray ramp; B is lifted for a cool low-to-white high ramp.
fn forward_lattice_heatmap_rgb(u: f32) -> (u8, u8, u8) {
    let u = u.clamp(0.0, 1.0);
    let r = (u * 255.0) as u8;
    let g = r;
    let b = (FORWARD_LATTICE_HEATMAP_B0 + (u * FORWARD_LATTICE_HEATMAP_B1_DELTA))
        .clamp(0.0, 255.0) as u8;
    (r, g, b)
}

/// Maps a row label character to the glyph printed in the ASCII heatmap (space and blank tokens get distinct marks).
fn ascii_y_tick_char(c: char) -> char {
    match c {
        ' ' => '·',
        '_' => '|',
        _ => c,
    }
}

/// Comma-separated interleaved row tokens for ASCII headers (space → `·` so gaps stay visible).
fn interleaved_tokens_csv(chars: &[char]) -> String {
    chars
        .iter()
        .map(|c| match c {
            ' ' => "·".to_string(),
            c => c.to_string(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Raw target tokens for captions: CTC interleaving is `blank, y0, blank, y1, …`, so labels sit at odd indices `2*i+1` for `i < target_len`.
fn raw_target_for_svg_caption(trace: &ForwardLatticeTrace, row_chars: Option<&[char]>) -> String {
    let n = trace.target_len;
    if let Some(chars) = row_chars {
        (0..n)
            .filter_map(|i| chars.get(2 * i + 1).copied())
            .collect::<String>()
    } else {
        (0..n)
            .filter_map(|i| trace.interleaved_ids.get(2 * i + 1).copied())
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Renders a monospace text heatmap: rows are interleaved state index (or glyphs from `y_row_chars`), columns are time.
fn render_ascii_lattice_heatmap_inner(
    trace: &ForwardLatticeTrace,
    y_row_chars: Option<&[char]>,
) -> String {
    let s_len = trace.interleaved_ids.len();
    let t_len = trace.grid_t_s.len();
    assert!(
        trace.grid_t_s.iter().all(|col| col.len() == s_len),
        "grid column length mismatch"
    );

    let mut finite: Vec<f32> = Vec::new();
    for col in &trace.grid_t_s {
        for &v in col {
            if v.is_finite() && v > SENTINEL_CUTOFF {
                finite.push(v);
            }
        }
    }
    let (vmin, vmax) = if finite.is_empty() {
        (0.0_f32, 1.0_f32)
    } else {
        let mn = finite.iter().copied().fold(f32::INFINITY, f32::min);
        let mx = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if (mx - mn).abs() < 1e-12 {
            (mn, mn + 1.0)
        } else {
            (mn, mx)
        }
    };

    let y_labels = y_row_chars.filter(|c| c.len() == s_len);

    let mut out = String::new();
    let target_seq = raw_target_for_svg_caption(trace, y_row_chars);
    out.push_str(&format!("Target Sequence: \"{target_seq}\"\n"));
    out.push_str("Interleaved Target Tokens (row order): ");
    if let Some(chars) = y_labels {
        out.push_str(&interleaved_tokens_csv(chars));
        out.push('\n');
    } else {
        out.push_str("(n/a — pass y_row_chars with length == interleaved grid rows)\n");
    }
    out.push_str("Interleaved Target IDs (row order): ");
    out.push_str(
        &trace
            .interleaved_ids
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", "),
    );
    out.push_str("\n\nRows = Target Tokens/Indices, Cols = Timesteps\n");
    out.push_str(&format!(
        "Timesteps = {}, Seq Len = {}, Inter Seq Len = {}\n\n",
        t_len, trace.target_len, s_len
    ));
    out.push_str(
        "Legend = ['·' = unreachable, ' ' = low, '░' = mid-low, '▒' = mid, '▓' = mid-high, '█' = high]\n",
    );
    let bins_n = ASCII_SHADES.len();
    let mut edges: Vec<f32> = Vec::with_capacity(bins_n + 1);
    for k in 0..=bins_n {
        let r = k as f32 / bins_n as f32;
        edges.push(vmin + (vmax - vmin) * r);
    }
    out.push_str(&format!(
        "Bins = ['\u{0020}': [{:.2}, {:.2}), '░': [{:.2}, {:.2}), '▒': [{:.2}, {:.2}), '▓': [{:.2}, {:.2}), '█': [{:.2}, {:.2}]]\n\n",
        edges[0], edges[1],
        edges[1], edges[2],
        edges[2], edges[3],
        edges[3], edges[4],
        edges[4], edges[5],
    ));

    // header: three monospace columns per timestep except the last (no trailing pad before row body)
    let mut header = String::from("     ");
    for ti in 0..t_len {
        let tick = ti % 100;
        if ti + 1 == t_len {
            header.push_str(&format!("{tick}"));
        } else {
            header.push_str(&format!("{:<3}", tick));
        }
    }
    out.push_str(&header);
    out.push('\n');

    for si in 0..s_len {
        let row_prefix = if let Some(chars) = y_labels {
            format!("{:>3} |", ascii_y_tick_char(chars[si]))
        } else {
            format!("{:>3} |", si)
        };
        out.push_str(&row_prefix);
        for ti in 0..t_len {
            let v = trace.grid_t_s[ti][si];
            let ch = if !v.is_finite() || v <= SENTINEL_CUTOFF {
                '·'
            } else {
                let u = ((v - vmin) / (vmax - vmin)).clamp(0.0, 1.0);
                let b = (u * (ASCII_SHADES.len() - 1) as f32).round() as usize;
                let idx = b.min(ASCII_SHADES.len() - 1);
                ASCII_SHADES[idx].chars().next().unwrap_or(' ')
            };
            if ti + 1 == t_len {
                out.push(ch);
            } else {
                out.push_str(&format!("{:<3}", ch));
            }
        }
        out.push_str(&format!(
            "  id = {}\n",
            trace.interleaved_ids[si]
        ));
    }
    out
}



/// Pixel layout, fonts, arrow styling, and legend geometry for the forward-lattice SVG export.
///
/// `heat_x0` equals `margin_left`. `heat_y0` is `margin_top` raised if needed so
/// `axes_to_heatmap_margin + x_axis_tick_band_h + title_h` fits above the grid. Y-axis ticks sit
/// left of the heatmap. Below the heatmap, x ticks and the axis label use `x_axis_tick_band_h` and
/// `x_axis_label_h`; above it, the target caption and main title mirror that stacking using the same
/// gap and tick band, with `title_h` in the role of `x_axis_label_h`.
///
/// To match [`CtcForwardLatticeViz::write_svg`], use [`Default`]. Override fields and pass the value to
/// [`CtcForwardLatticeViz::write_svg_with_theme`].
#[derive(Clone, Copy, Debug)]
pub struct ForwardLatticeSvgTheme {
    pub x_axis_label_fs: f64,
    pub x_axis_tick_fs: f64,
    pub y_axis_label_fs: f64,
    pub y_axis_tick_fs: f64,
    /// Font size for the main SVG title.
    pub title_fs: f64,
    /// Height of the title strip above the mirrored x-tick band (same role as [`Self::x_axis_label_h`] under the heatmap). The title is centered at `heat_y0 - axes_to_heatmap_margin - x_axis_tick_band_h - title_h/2`.
    pub title_h: u32,
    pub margin_top: u32,
    pub margin_bottom: u32,
    pub margin_left: u32,
    pub margin_right: u32,
    /// Height of the strip below the x tick band for the x-axis label (e.g. "Timesteps (t) →").
    pub x_axis_label_h: u32,
    pub x_axis_tick_band_h: u32,
    /// Width of the strip left of the y tick band for the y-axis label.
    pub y_axis_label_w: u32,
    pub y_axis_tick_band_w: u32,
    /// Gap between the heatmap and the y tick band (left) or x tick band (bottom). The target caption above the heatmap centers at `heat_y0 - axes_to_heatmap_margin - x_axis_tick_band_h/2`, matching the x tick row below.
    pub axes_to_heatmap_margin: u32,
    /// Nudge for the x-axis label anchor (after horizontal centering on the heatmap).
    pub x_axis_label_x_offset: i32,
    /// Nudge for the y-axis label anchor (after vertical centering on the heatmap).
    pub y_axis_label_y_offset: i32,
    pub cell_min: u32,
    pub cell_max: u32,
    pub cell_max_axis_px: u32,
    /// Stroke width for DP transition arrows; set to `0` to omit arrows. Opacity follows conditional edge mass and the other `alignment_arrow_*` fields.
    pub alignment_arrow_stroke_px: u32,
    /// Stroke color for edges that are not on any complete CTC alignment (dashed).
    pub alignment_arrow_rgb: (u8, u8, u8),
    /// Stroke color for edges on at least one complete CTC alignment (solid).
    pub alignment_arrow_complete_rgb: (u8, u8, u8),
    /// When greater than `0`, edges with conditional fraction below this value are skipped. `0.0` draws every edge in the trace (if arrows are enabled).
    pub alignment_arrow_min_frac: f32,
    /// Multiply edge fraction by this for stroke/fill alpha (clamped to 1).
    pub alignment_arrow_max_alpha: f64,
    /// Lower bound on arrow opacity so low-mass edges remain faintly visible.
    pub alignment_arrow_min_alpha: f64,
    pub legend_fs: f64,
    pub legend_bar_w: u32,
    pub legend_bar_min_h: u32,
    /// RGB for all SVG text (title, ticks, axis labels, legend numerals).
    pub text_rgb: (u8, u8, u8),
}



impl Default for ForwardLatticeSvgTheme {
    fn default() -> Self {
        Self {
            title_fs: 20.0,
            title_h: 22,
            x_axis_label_fs: 20.0,
            y_axis_label_fs: 20.0,
            x_axis_label_h: 22,
            y_axis_label_w: 22,
            x_axis_label_x_offset: 0,
            y_axis_label_y_offset: 0,
            x_axis_tick_fs: 16.0,
            y_axis_tick_fs: 16.0,
            x_axis_tick_band_h: 32,
            y_axis_tick_band_w: 32,
            margin_top: 75,
            margin_bottom: 75,
            margin_left: 75,
            margin_right: 90,
            axes_to_heatmap_margin: 2,
            cell_min: 6,
            cell_max: 28,
            cell_max_axis_px: 920,
            alignment_arrow_stroke_px: 2,
            alignment_arrow_rgb: (210, 130, 130),
            alignment_arrow_complete_rgb: (90, 185, 125),
            alignment_arrow_min_frac: 0.0,
            alignment_arrow_max_alpha: 0.78,
            alignment_arrow_min_alpha: 0.15,
            legend_fs: 12.0,
            legend_bar_w: 14,
            legend_bar_min_h: 120,
            text_rgb: (100, 106, 118),
        }
    }
}



/// Pixel center of cell `(ti, si)` after flipping rows so interleaved index `0` is at the bottom of the heatmap.
fn lattice_cell_center_px(
    ti: usize,
    si: usize,
    s_len: usize,
    heat_x0: u32,
    heat_y0: u32,
    cw: u32,
    ch: u32,
) -> (i32, i32) {
    let si_flip = s_len - 1 - si;
    let cx = lattice_col_center_x_px(heat_x0, ti, cw);
    let cy = heat_y0 as i64 + (si_flip as i64) * (ch as i64) + (ch as i64) / 2;
    let cy = cy.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    (cx, cy)
}



/// Scales tick label font size with cell pitch so glyphs stay near the cell center on dense grids.
///
/// Anchors are already at cell centers; a fixed large point size would spill outside small cells.
fn svg_tick_fs_for_cell_pitch(cell_px: u32, theme_fs: f64) -> f64 {
    const REF_PITCH_PX: f64 = 16.0;
    let scale = (cell_px as f64 / REF_PITCH_PX).clamp(0.3, 1.0);
    (theme_fs * scale).clamp(4.5, theme_fs)
}



/// Left edge x-coordinate of timestep column `ti` in pixels (`i64` intermediate avoids overflow on wide grids).
fn lattice_col_x0_px(heat_x0: u32, ti: usize, cw: u32) -> i32 {
    let x = heat_x0 as i64 + (ti as i64) * (cw as i64);
    x.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}



/// Horizontal center of column `ti`; matches heatmap cells and arrow endpoints.
fn lattice_col_center_x_px(heat_x0: u32, ti: usize, cw: u32) -> i32 {
    let x = heat_x0 as i64 + (ti as i64) * (cw as i64) + (cw as i64) / 2;
    x.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}



/// Chooses a timestep stride so consecutive tick labels are unlikely to overlap at the given font size.
fn svg_x_tick_stride(t_len: usize, cw: u32, tick_fs: f64) -> usize {
    if t_len <= 1 {
        return 1;
    }
    let last = t_len - 1;
    let max_digits = last.to_string().len().max(1);
    // per-digit width for collision estimate
    let est_label_px = (max_digits as f64) * tick_fs * 0.62 + 8.0;
    let cell = cw.max(1) as f64;
    let stride = (est_label_px / cell).ceil() as usize;
    stride.max(1)
}



/// Timestep indices to draw as x tick labels: `0`, then every `stride` step along `0..t_len`.
///
/// Index `t_len - 1` is included only when `(t_len - 1) % stride == 0`, so an off-grid last column is not labeled alone beyond the previous tick.
fn svg_x_tick_indices(t_len: usize, stride: usize) -> Vec<usize> {
    if t_len == 0 {
        return Vec::new();
    }
    let stride = stride.max(1);
    (0..t_len).step_by(stride).collect()
}



/// Blank-interleaved label length for true target length `L` (`2L + 1`), matching [`CtcLoss::forward_no_reduction`].
fn ctc_interleaved_true_len(target_len: usize) -> usize {
    target_len.saturating_mul(2).saturating_add(1)
}



/// DP edges `(tau, s_from, s_to)` that belong to at least one valid full CTC alignment in the traced graph.
///
/// Reachability starts from states `0` and `1` at `t = 0` (when present), ends in a CTC terminal at
/// `t = input_len - 1`, and follows only edges listed in `trace.dp_edge_fractions`. Edge mass is not
/// used here; rendering still scales arrow opacity by mass with a floor of `alignment_arrow_min_alpha`.
fn ctc_complete_path_edge_set(trace: &ForwardLatticeTrace) -> HashSet<(usize, usize, usize)> {
    let t_len = trace.grid_t_s.len();
    let s_len = trace.interleaved_ids.len();
    let time_steps = trace.input_len.min(t_len);
    if time_steps == 0 || s_len == 0 {
        return HashSet::new();
    }
    let intr_true = ctc_interleaved_true_len(trace.target_len).min(s_len);
    if intr_true == 0 {
        return HashSet::new();
    }

    // forward reachability
    let mut f = vec![vec![false; s_len]; time_steps];
    for s in 0..intr_true.min(2).min(s_len) {
        let v = trace.grid_t_s[0][s];
        if v.is_finite() && v > SENTINEL_CUTOFF {
            f[0][s] = true;
        }
    }
    for (tau, edges) in trace.dp_edge_fractions.iter().enumerate() {
        if tau + 1 >= time_steps {
            break;
        }
        for &(s_from, s_to, _) in edges {
            if s_from < s_len && s_to < s_len && f[tau][s_from] {
                f[tau + 1][s_to] = true;
            }
        }
    }

    // backward reachability from terminal states
    let last_t = time_steps - 1;
    let mut b = vec![vec![false; s_len]; time_steps];
    let terminals: &[usize] = if intr_true >= 2 {
        &[intr_true - 1, intr_true - 2]
    } else {
        &[0]
    };
    for &e in terminals {
        if e < s_len && f[last_t][e] {
            b[last_t][e] = true;
        }
    }
    for (tau, edges) in trace.dp_edge_fractions.iter().enumerate().rev() {
        if tau + 1 >= time_steps {
            continue;
        }
        for &(s_from, s_to, _) in edges {
            if s_to < s_len && b[tau + 1][s_to] && s_from < s_len {
                b[tau][s_from] = true;
            }
        }
    }

    let mut out = HashSet::new();
    for (tau, edges) in trace.dp_edge_fractions.iter().enumerate() {
        if tau + 1 >= time_steps {
            continue;
        }
        for &(s_from, s_to, _) in edges {
            if s_from < s_len && s_to < s_len && f[tau][s_from] && b[tau + 1][s_to] {
                out.insert((tau, s_from, s_to));
            }
        }
    }
    out
}



/// Builds a polyline shaft and a triangular arrowhead between two cell centers.
///
/// The shaft ends slightly before the tip. Head size derives from cell pitch, not edge length, and shrinks on very short edges.
fn arrow_shaft_and_head(
    from: (i32, i32),
    to: (i32, i32),
    cw: u32,
    ch: u32,
) -> Option<(((i32, i32), (i32, i32)), [(i32, i32); 3])> {
    let (x0, y0) = (from.0 as f32, from.1 as f32);
    let (x1, y1) = (to.0 as f32, to.1 as f32);
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1.0 {
        return None;
    }
    let ux = dx / len;
    let uy = dy / len;
    let cell_min = cw.min(ch) as f32;
    // head length from cell size (shared for stay / advance edges in one figure)
    let mut head_len = (cell_min * 0.30).max(4.0);
    // leave at least a one-pixel shaft when the edge is short
    head_len = head_len.min((len - 1.0).max(1.0));
    let half_w = (cell_min * 0.14).max(2.0);
    let bx = x1 - ux * head_len;
    let by = y1 - uy * head_len;
    let px = -uy;
    let py = ux;
    let tip = (to.0, to.1);
    let left = (bx + px * half_w, by + py * half_w);
    let right = (bx - px * half_w, by - py * half_w);
    // nudge shaft end toward tip so stroke meets filled head when rasterized
    let join_overlap_px = 0.6_f32;
    let sx = bx + ux * join_overlap_px;
    let sy = by + uy * join_overlap_px;
    let shaft_end = (sx.round() as i32, sy.round() as i32);
    let tri = [
        tip,
        (left.0.round() as i32, left.1.round() as i32),
        (right.0.round() as i32, right.1.round() as i32),
    ];
    Some(((from, shaft_end), tri))
}

/// Writes the lattice SVG to `path`. `row_chars` supplies one display character per interleaved row when present.
fn write_forward_lattice_svg(
    trace: &ForwardLatticeTrace,
    path: &std::path::Path,
    row_chars: Option<&[char]>,
    theme: &ForwardLatticeSvgTheme,
) -> Result<(), String> {
    use plotters::element::{Circle, DashedPathElement, PathElement, Polygon, Rectangle, Text};
    use plotters::prelude::*;
    use plotters::style::text_anchor::{HPos, Pos, VPos};
    use plotters::style::Color;

    let s_len = trace.interleaved_ids.len();
    let t_len = trace.grid_t_s.len();
    if t_len == 0 || s_len == 0 {
        return Err("empty lattice trace".to_string());
    }
    if let Some(rc) = row_chars {
        if rc.len() != s_len {
            return Err(format!(
                "row_chars length {} != s_len {}",
                rc.len(),
                s_len
            ));
        }
    }

    let mut finite: Vec<f32> = Vec::new();
    for col in &trace.grid_t_s {
        for &v in col {
            if v.is_finite() && v > SENTINEL_CUTOFF {
                finite.push(v);
            }
        }
    }
    let (vmin, vmax) = if finite.is_empty() {
        (0.0_f32, 1.0_f32)
    } else {
        let mn = finite.iter().copied().fold(f32::INFINITY, f32::min);
        let mx = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if (mx - mn).abs() < 1e-12 {
            (mn, mn + 1.0)
        } else {
            (mn, mx)
        }
    };

    let n_max = t_len.max(s_len) as u32;
    let cell = (theme.cell_max_axis_px / n_max.max(1)).clamp(theme.cell_min, theme.cell_max);
    let (cw, ch) = (cell, cell);
    let x_axis_tick_fs_eff = svg_tick_fs_for_cell_pitch(cell, theme.x_axis_tick_fs);
    let y_axis_tick_fs_eff = svg_tick_fs_for_cell_pitch(cell, theme.y_axis_tick_fs);
    let heat_w: u32 = (t_len as u128)
        .saturating_mul(cw as u128)
        .min(u32::MAX as u128) as u32;
    let heat_h = s_len as u32 * ch;

    // heatmap origin; heat_y0 accounts for title strip + tick band above
    let heat_x0 = theme.margin_left;
    let top_stack_min = theme
        .axes_to_heatmap_margin
        .saturating_add(theme.x_axis_tick_band_h)
        .saturating_add(theme.title_h);
    let heat_y0 = theme.margin_top.max(top_stack_min);
    let heat_yn = heat_y0 + heat_h;
    let pw = heat_x0 + heat_w + theme.margin_right;
    let ph = heat_yn + theme.margin_bottom;

    let root = SVGBackend::new(path, (pw, ph)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| format!("plotters fill: {e}"))?;

    let text_rgb = RGBColor(theme.text_rgb.0, theme.text_rgb.1, theme.text_rgb.2);
    let style_x_tick = TextStyle::from((SVG_UI_FONT, x_axis_tick_fs_eff).into_font().color(&text_rgb))
        .pos(Pos::new(HPos::Center, VPos::Center));
    let style_svg_title = TextStyle::from((SVG_UI_FONT, theme.title_fs).into_font().color(&text_rgb))
        .pos(Pos::new(HPos::Center, VPos::Center));
    let subtitle_fs = (theme.title_fs * 0.75).clamp(13.0, 18.0);
    // subtitle font: same vertical band model as x ticks
    let style_svg_subtitle = TextStyle::from((SVG_UI_FONT, subtitle_fs).into_font().color(&text_rgb))
        .pos(Pos::new(HPos::Center, VPos::Center));
    let style_x_axis_label = TextStyle::from((SVG_UI_FONT, theme.x_axis_label_fs).into_font().color(&text_rgb))
        .pos(Pos::new(HPos::Center, VPos::Center));
    let style_y_tick = TextStyle::from((SVG_UI_FONT, y_axis_tick_fs_eff).into_font().color(&text_rgb))
        .transform(FontTransform::Rotate270)
        .pos(Pos::new(HPos::Center, VPos::Center));
    let style_y_axis_label = TextStyle::from((SVG_UI_FONT, theme.y_axis_label_fs).into_font().color(&text_rgb))
        .transform(FontTransform::Rotate270)
        .pos(Pos::new(HPos::Center, VPos::Center));
    let style_legend = TextStyle::from((SVG_UI_FONT, theme.legend_fs).into_font().color(&text_rgb))
        .pos(Pos::new(HPos::Left, VPos::Center));
    let style_legend_title = TextStyle::from((SVG_UI_FONT, theme.legend_fs).into_font().color(&text_rgb))
        .pos(Pos::new(HPos::Center, VPos::Center));

    // main title y: mirrors x-axis label placement above the heatmap
    let y_svg_title = (heat_y0 as i32
        - theme.axes_to_heatmap_margin as i32
        - theme.x_axis_tick_band_h as i32
        - (theme.title_h / 2) as i32)
        .max(1);
    // title x: centered on heatmap width
    let x_svg_title = (heat_x0 + heat_w / 2) as i32;
    root.draw(
        &Text::new(
            "CTC Loss: Forward Lattice Heatmap",
            (x_svg_title, y_svg_title),
            &style_svg_title,
        )
        .into_dyn(),
    )
    .map_err(|e| format!("plotters draw svg title: {e}"))?;

    let target_seq_body = raw_target_for_svg_caption(trace, row_chars);
    let target_seq_line = format!("Target Sequence: \"{target_seq_body}\"");
    // target caption y: mirrors x tick center positioning
    let y_target_seq = (heat_y0 as i32
        - theme.axes_to_heatmap_margin as i32
        - (theme.x_axis_tick_band_h / 2) as i32)
        .max(1);
    let x_target_seq = (heat_x0 + heat_w / 2) as i32;
    root.draw(
        &Text::new(target_seq_line, (x_target_seq, y_target_seq), &style_svg_subtitle).into_dyn(),
    )
    .map_err(|e| format!("plotters draw target sequence caption: {e}"))?;

    // y ticks: row s=0 at bottom, labels read upward
    let x_y_ticks = heat_x0 as i32 - theme.axes_to_heatmap_margin as i32 - (theme.y_axis_tick_band_w / 2) as i32;
    for si in 0..s_len {
        let row_txt = if let Some(chars) = row_chars {
            chars[si].to_string()
        } else {
            trace.interleaved_ids[si].to_string()
        };
        let si_flip = s_len - 1 - si;
        let cy = heat_y0 as i32 + (si_flip as u32 * ch) as i32 + (ch / 2) as i32;
        root.draw(
            &Text::new(row_txt, (x_y_ticks, cy), &style_y_tick).into_dyn(),
        )
        .map_err(|e| format!("plotters draw row label: {e}"))?;
    }

    let y_axis_label_x = heat_x0 as i32
        - theme.axes_to_heatmap_margin as i32
        - theme.y_axis_tick_band_w as i32
        - (theme.y_axis_label_w / 2) as i32;
    let y_axis_label_y =
        heat_y0 as i32 + (heat_h / 2) as i32 + theme.y_axis_label_y_offset;
    root.draw(
        &Text::new(
            "Targ Seq (L') →",
            (y_axis_label_x, y_axis_label_y),
            &style_y_axis_label,
        )
        .into_dyn(),
    )
    .map_err(|e| format!("plotters draw y axis label: {e}"))?;

    for ti in 0..t_len {
        for si in 0..s_len {
            let v = trace.grid_t_s[ti][si];
            let (r, g, b) = if !v.is_finite() || v <= SENTINEL_CUTOFF {
                FORWARD_LATTICE_INVALID_CELL_RGB
            } else {
                let u = ((v - vmin) / (vmax - vmin)).clamp(0.0, 1.0);
                forward_lattice_heatmap_rgb(u)
            };
            let color = RGBColor(r, g, b);
            let x0 = lattice_col_x0_px(heat_x0, ti, cw);
            let si_flip = s_len - 1 - si;
            let y0 = heat_y0 as i32 + (si_flip as u32 * ch) as i32;
            let x1 = x0 + cw as i32;
            let y1 = y0 + ch as i32;
            let rect = Rectangle::new([(x0, y0), (x1, y1)], color.filled());
            root.draw(&rect)
                .map_err(|e| format!("plotters draw: {e}"))?;
        }
    }
    let border_style = ShapeStyle::from(&text_rgb).stroke_width(1);
    root.draw(&Rectangle::new(
        [
            (heat_x0 as i32, heat_y0 as i32),
            ((heat_x0 + heat_w) as i32, (heat_y0 + heat_h) as i32),
        ],
        border_style,
    ))
    .map_err(|e| format!("plotters draw heatmap border: {e}"))?;
    // cell grid lines
    let grid_style = RGBColor(25, 25, 60).mix(0.18).stroke_width(1);
    for ti in 0..=t_len {
        let x = lattice_col_x0_px(heat_x0, ti, cw);
        root.draw(&PathElement::new(
            vec![(x, heat_y0 as i32), (x, (heat_y0 + heat_h) as i32)],
            grid_style,
        ))
        .map_err(|e| format!("plotters draw heatmap v-grid: {e}"))?;
    }
    for si in 0..=s_len {
        let y = heat_y0 as i32 + (si as u32 * ch) as i32;
        root.draw(&PathElement::new(
            vec![(heat_x0 as i32, y), ((heat_x0 + heat_w) as i32, y)],
            grid_style,
        ))
        .map_err(|e| format!("plotters draw heatmap h-grid: {e}"))?;
    }

    // anchor dots at cell centers
    let dot_r = (cw.min(ch) / 9).clamp(2, 5);
    let dot_style = RGBColor(58, 58, 72).mix(0.62);
    for ti in 0..t_len {
        for si in 0..s_len {
            let (cx, cy) = lattice_cell_center_px(ti, si, s_len, heat_x0, heat_y0, cw, ch);
            root.draw(&Circle::new((cx, cy), dot_r, dot_style.filled()))
                .map_err(|e| format!("plotters draw cell anchor dot: {e}"))?;
        }
    }

    // colorbar gradient
    let bar_h = (heat_h / 2).max(1);
    let bar_x0 = heat_x0 + heat_w + 14;
    let bar_y0 = heat_y0;
    let bar_steps = bar_h.max(2);
    for i in 0..bar_steps {
        let u = 1.0_f32 - (i as f32 / (bar_steps - 1) as f32); // top is high log-alpha
        let (r, g, b) = forward_lattice_heatmap_rgb(u);
        let c = RGBColor(r, g, b);
        let y0 = bar_y0 as i32 + i as i32;
        let y1 = y0 + 1;
        let x0 = bar_x0 as i32;
        let x1 = x0 + theme.legend_bar_w as i32;
        root.draw(&Rectangle::new([(x0, y0), (x1, y1)], c.filled()))
            .map_err(|e| format!("plotters draw legend bar: {e}"))?;
    }
    // colorbar border
    root.draw(&Rectangle::new(
        [
            (bar_x0 as i32, bar_y0 as i32),
            (
                (bar_x0 + theme.legend_bar_w) as i32,
                (bar_y0 + bar_h) as i32,
            ),
        ],
        border_style,
    ))
    .map_err(|e| format!("plotters draw legend border: {e}"))?;
    let lx = (bar_x0 + theme.legend_bar_w + 8) as i32;
    let y_top = bar_y0 as i32;
    let y_bottom = (bar_y0 + bar_h) as i32;
    let y_mid = (y_top + y_bottom) / 2;
    root.draw(&Text::new(format!("{:.2}", vmax), (lx, y_top), &style_legend).into_dyn())
        .map_err(|e| format!("plotters draw legend max: {e}"))?;
    root.draw(&Text::new(format!("{:.2}", vmin), (lx, y_bottom), &style_legend).into_dyn())
        .map_err(|e| format!("plotters draw legend min: {e}"))?;
    root.draw(&Text::new(
        format!("{:.2}", 0.5 * (vmin + vmax)),
        (lx, y_mid),
        &style_legend,
    ).into_dyn())
    .map_err(|e| format!("plotters draw legend mid: {e}"))?;
    root.draw(&Text::new(
        "α",
        (
            (bar_x0 + theme.legend_bar_w / 2) as i32,
            (bar_y0 as i32 - 10).max(8),
        ),
        &style_legend_title,
    ).into_dyn())
    .map_err(|e| format!("plotters draw legend title: {e}"))?;

    // masked-cell swatch (N/A)
    let legend_invalid_gap: u32 = 8;
    let invalid_side = theme.legend_bar_w;
    let invalid_y0 = bar_y0 + bar_h + legend_invalid_gap;
    let (ir, ig, ib) = FORWARD_LATTICE_INVALID_CELL_RGB;
    let invalid_fill = RGBColor(ir, ig, ib);
    root.draw(&Rectangle::new(
        [
            (bar_x0 as i32, invalid_y0 as i32),
            (
                (bar_x0 + invalid_side) as i32,
                (invalid_y0 + invalid_side) as i32,
            ),
        ],
        invalid_fill.filled(),
    ))
    .map_err(|e| format!("plotters draw legend invalid fill: {e}"))?;
    root.draw(&Rectangle::new(
        [
            (bar_x0 as i32, invalid_y0 as i32),
            (
                (bar_x0 + invalid_side) as i32,
                (invalid_y0 + invalid_side) as i32,
            ),
        ],
        border_style,
    ))
    .map_err(|e| format!("plotters draw legend invalid border: {e}"))?;
    let y_invalid_label = (invalid_y0 + invalid_side / 2) as i32;
    root.draw(
        &Text::new("N/A", (lx, y_invalid_label), &style_legend).into_dyn(),
    )
    .map_err(|e| format!("plotters draw legend invalid label: {e}"))?;

    // dp arrows; color by reachability, opacity by conditional mass
    if theme.alignment_arrow_stroke_px > 0 && !trace.dp_edge_fractions.is_empty() {
        let (r_inc, g_inc, b_inc) = theme.alignment_arrow_rgb;
        let (r_ok, g_ok, b_ok) = theme.alignment_arrow_complete_rgb;
        let rgb_incomplete = RGBColor(r_inc, g_inc, b_inc);
        let rgb_complete = RGBColor(r_ok, g_ok, b_ok);
        let sw = theme.alignment_arrow_stroke_px;
        let min_f = theme.alignment_arrow_min_frac;
        let max_a = theme.alignment_arrow_max_alpha;
        let min_a = theme.alignment_arrow_min_alpha;
        let complete_edges = ctc_complete_path_edge_set(trace);

        let mut edges_draw: Vec<(usize, usize, usize, f32)> = Vec::new();
        for (tau, ed) in trace.dp_edge_fractions.iter().enumerate() {
            for &(s_from, s_to, frac) in ed {
                if s_from >= s_len || s_to >= s_len || !frac.is_finite() || frac <= 0.0 {
                    continue;
                }
                if min_f > 0.0 && frac < min_f {
                    continue;
                }
                edges_draw.push((tau, s_from, s_to, frac));
            }
        }
        edges_draw.sort_by(|a, b| {
            let ac = complete_edges.contains(&(a.0, a.1, a.2));
            let bc = complete_edges.contains(&(b.0, b.1, b.2));
            ac.cmp(&bc)
                .then_with(|| {
                    a.3.partial_cmp(&b.3)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        let dash_px = (cw.min(ch) / 5).clamp(2, 5);
        for (tau, s0, s1, frac) in edges_draw {
            let alpha = (frac as f64 * max_a).clamp(min_a, 1.0);
            let on_complete = complete_edges.contains(&(tau, s0, s1));
            let arrow_rgb = if on_complete {
                rgb_complete
            } else {
                rgb_incomplete
            };
            let rgba = arrow_rgb.mix(alpha);
            let p0 = lattice_cell_center_px(tau, s0, s_len, heat_x0, heat_y0, cw, ch);
            let p1 = lattice_cell_center_px(tau + 1, s1, s_len, heat_x0, heat_y0, cw, ch);
            if let Some((shaft, tri)) = arrow_shaft_and_head(p0, p1, cw, ch) {
                let shaft_stroke = rgba.stroke_width(sw);
                if on_complete {
                    root.draw(&PathElement::new(vec![shaft.0, shaft.1], shaft_stroke))
                        .map_err(|e| format!("plotters draw arrow shaft: {e}"))?;
                } else {
                    root.draw(&DashedPathElement::new(
                        vec![shaft.0, shaft.1],
                        dash_px,
                        dash_px,
                        shaft_stroke,
                    ))
                    .map_err(|e| format!("plotters draw arrow shaft dashed: {e}"))?;
                }
                root.draw(&Polygon::new(tri.to_vec(), rgba.filled()))
                    .map_err(|e| format!("plotters draw arrow head: {e}"))?;
            }
        }
    }

    // x ticks below heatmap
    let y_x_ticks = heat_yn as i32
        + theme.axes_to_heatmap_margin as i32
        + (theme.x_axis_tick_band_h / 2) as i32;
    let x_stride = svg_x_tick_stride(t_len, cw, x_axis_tick_fs_eff);
    for ti in svg_x_tick_indices(t_len, x_stride) {
        let cx = lattice_col_center_x_px(heat_x0, ti, cw);
        root.draw(
            &Text::new(format!("{ti}"), (cx, y_x_ticks), &style_x_tick).into_dyn(),
        )
        .map_err(|e| format!("plotters draw col label: {e}"))?;
    }

    let y_x_axis_label = heat_yn as i32
        + theme.axes_to_heatmap_margin as i32
        + theme.x_axis_tick_band_h as i32
        + (theme.x_axis_label_h / 2) as i32;
    root.draw(
        &Text::new(
            "Timesteps (t) →",
            (
                heat_x0 as i32 + (heat_w / 2) as i32 + theme.x_axis_label_x_offset,
                y_x_axis_label,
            ),
            &style_x_axis_label,
        )
        .into_dyn(),
    )
    .map_err(|e| format!("plotters draw x axis label: {e}"))?;

    root.present()
        .map_err(|e| format!("plotters present: {e}"))?;
    Ok(())
}



/// Captures a forward lattice trace and renders ASCII or SVG for inspection and fixtures.
pub struct CtcForwardLatticeViz;



impl CtcForwardLatticeViz {
    /// Captures log-alpha grid and DP edges for the first batch item (requires `N = 1`).
    pub fn lattice_trace_sample0<B: Backend>(
        loss: &CtcLoss,
        inputs: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        input_lengths: Tensor<B, 1, Int>,
        target_lengths: Tensor<B, 1, Int>,
    ) -> ForwardLatticeTrace {
        lattice_trace_sample0_inner(loss, inputs, targets, input_lengths, target_lengths)
    }

    /// One display character per interleaved row, same order as [`ForwardLatticeTrace::interleaved_ids`].
    ///
    /// In ASCII output, space prints as `·` and blank token `_` as `|`; SVG uses the raw characters (including rotated `_`). If `None` or the length does not match, row headers use the numeric index `s`.
    pub fn render_ascii(trace: &ForwardLatticeTrace, y_row_chars: Option<&[char]>) -> String {
        render_ascii_lattice_heatmap_inner(trace, y_row_chars)
    }

    /// Renders [`ForwardLatticeTrace`] to an SVG file. `row_chars` labels each interleaved row when provided.
    ///
    /// Layout and style follow [`ForwardLatticeSvgTheme::default`]; use [`Self::write_svg_with_theme`] to override.
    pub fn write_svg(
        trace: &ForwardLatticeTrace,
        path: &std::path::Path,
        row_chars: Option<&[char]>,
    ) -> Result<(), String> {
        write_forward_lattice_svg(trace, path, row_chars, &ForwardLatticeSvgTheme::default())
    }

    /// Same as [`Self::write_svg`], using a custom [`ForwardLatticeSvgTheme`].
    pub fn write_svg_with_theme(
        trace: &ForwardLatticeTrace,
        path: &std::path::Path,
        row_chars: Option<&[char]>,
        theme: &ForwardLatticeSvgTheme,
    ) -> Result<(), String> {
        write_forward_lattice_svg(trace, path, row_chars, theme)
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::ctc::ctc_loss::{CtcLoss, CtcLossConfig};
    use crate::vocab::{TokenMap, BLANK_ID, VOCAB, VOCAB_SIZE};
    use burn::{
        backend::ndarray::NdArray,
        nn::loss::Reduction,
        prelude::Int,
        tensor::{backend::Backend, Tensor, TensorData},
    };

    type B = NdArray<f32>;

    /// Sample words for ASCII and optional SVG export tests; every character must appear in [`VOCAB`].
    const FIXTURE_SEQS: &[&str] = &["cat", "gobbledygook", "hippopotomonstrosesquippedaliophobia"];

    /// Base timestep count step when building synthetic logits for fixtures.
    const FIXTURE_T_BUFFER: usize = 20;

    /// Returns a timestep count at least [`FIXTURE_T_BUFFER`], rounded up in steps of `FIXTURE_T_BUFFER` until `T >= 2L + 1`.
    fn fixture_timesteps(target_token_len: usize) -> usize {
        let interleaved_len = target_token_len.saturating_mul(2).saturating_add(1);
        let buf = FIXTURE_T_BUFFER;
        assert!(buf > 0, "FIXTURE_T_BUFFER must be positive");
        if interleaved_len <= buf {
            buf
        } else {
            let mut t = buf;
            while t < interleaved_len {
                t += buf;
            }
            t
        }
    }

    fn outputs_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("outputs")
    }


    /// Builds logits that emphasize the interleaved target index that sweeps with time (fixture only).
    fn synthetic_logits_sweep_interleaved<Bk: Backend>(
        loss: &CtcLoss,
        device: &Bk::Device,
        targets: Tensor<Bk, 2, Int>,
        t: usize,
        vocab_size: usize,
    ) -> Tensor<Bk, 3> {
        let targets_intr = loss.interleave_targets_with_blanks(targets.clone(), device);
        let intr_ids: Vec<i32> = targets_intr
            .clone()
            .into_data()
            .convert::<i32>()
            .into_vec::<i32>()
            .unwrap();
        let intr_len = intr_ids.len();

        let mut logits = Tensor::<Bk, 3>::zeros([1, t, vocab_size], device);
        for t_idx in 0..t {
            let s = ((t_idx * intr_len) / t).min(intr_len - 1);
            let sym = intr_ids[s] as usize;
            logits = logits.slice_assign(
                [0..1, t_idx..(t_idx + 1), sym..(sym + 1)],
                Tensor::<Bk, 3>::from_floats([[[4.0]]], device),
            );
        }
        logits
    }

    /// Builds a lattice trace and interleaved row characters for `word` using [`VOCAB`] and [`BLANK_ID`].
    fn forward_lattice_trace_for_word(word: &str, token_map: &TokenMap) -> (ForwardLatticeTrace, Vec<char>) {
        let device = Default::default();
        let chars: Vec<char> = word.chars().collect();
        let ids_usize = token_map.chars_to_ids(&chars).unwrap_or_else(|| {
            panic!("fixture word {word:?} must be fully encodable in VOCAB={VOCAB:?}");
        });
        let l_max = ids_usize.len();
        assert!(l_max > 0, "fixture word must be non-empty");
        let t = fixture_timesteps(l_max);

        let loss = CtcLossConfig::new()
            .with_blank_id(BLANK_ID)
            .with_reduction(Reduction::Mean)
            .init();

        let ids_i64: Vec<i64> = ids_usize.iter().map(|&x| x as i64).collect();
        let targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(ids_i64, vec![1, l_max]),
            &device,
        );
        let in_len = Tensor::<B, 1, Int>::from_ints([t as i64], &device);
        let tgt_len = Tensor::<B, 1, Int>::from_ints([l_max as i64], &device);

        let logits = synthetic_logits_sweep_interleaved(
            &loss,
            &device,
            targets.clone(),
            t,
            VOCAB_SIZE,
        );

        let loss_scalar = loss
            .forward_no_reduction(
                logits.clone(),
                targets.clone(),
                in_len.clone(),
                tgt_len.clone(),
            )
            .into_scalar();
        assert!(
            loss_scalar.is_finite(),
            "loss should be finite for viz fixture (word={word:?})"
        );

        let trace = CtcForwardLatticeViz::lattice_trace_sample0(
            &loss,
            logits,
            targets,
            in_len,
            tgt_len,
        );
        assert_eq!(trace.grid_t_s.len(), t);
        assert_eq!(trace.interleaved_ids.len(), 2 * l_max + 1);

        let row_chars: Vec<char> = trace
            .interleaved_ids
            .iter()
            .map(|&id| {
                token_map.char_of(id as usize).unwrap_or_else(|| {
                    panic!("interleaved id {id} out of VOCAB for word={word:?}");
                })
            })
            .collect();

        (trace, row_chars)
    }

    #[test]
    fn ctc_complete_path_edge_set_nonempty_for_fixture() {
        let token_map = TokenMap::new(VOCAB);
        let (trace, _) = forward_lattice_trace_for_word("hi", &token_map);
        let complete = ctc_complete_path_edge_set(&trace);
        assert!(
            !complete.is_empty(),
            "expected some DP edges on a full CTC alignment for fixture word 'hi'"
        );
    }

    #[test]
    fn ctc_loss_lattice_ascii_printout() {
        let token_map = TokenMap::new(VOCAB);
        for word in FIXTURE_SEQS {
            let (trace, row_chars) = forward_lattice_trace_for_word(word, &token_map);
            let ascii = CtcForwardLatticeViz::render_ascii(&trace, Some(row_chars.as_slice()));
            println!("\n=== CTC Loss: ASCII Forward Log-Alpha Heatmap Lattice ===\n\n{ascii}\n");
            assert!(
                ascii.contains("Rows = Target Tokens/Indices, Cols = Timesteps"),
                "word={word:?}: expected ascii axis description"
            );
            assert!(
                ascii.contains(&format!("Target Sequence: \"{word}\"")),
                "word={word:?}: expected target sequence line"
            );
            assert!(
                ascii.contains("Interleaved Target Tokens (row order): ")
                    && !ascii.contains("(n/a — pass y_row_chars"),
                "word={word:?}: expected interleaved token line when row_chars provided"
            );
            assert!(
                ascii.contains("Interleaved Target IDs (row order): "),
                "word={word:?}: expected interleaved id line"
            );
            assert!(
                ascii.contains(&format!(
                    "Timesteps = {}, Seq Len = {}, Inter Seq Len = {}",
                    trace.grid_t_s.len(),
                    trace.target_len,
                    trace.interleaved_ids.len()
                )),
                "word={word:?}: expected length line"
            );
            assert!(
                ascii.contains("Legend = ['·' = unreachable, ' ' = low,")
                    && ascii.contains("Bins = [")
                    && ascii.contains("' ': [")
                    && ascii.contains("'░': ["),
                "word={word:?}: expected legend and bins lines"
            );
        }
    }

    #[test]
    #[ignore = "writes outputs/ctc_loss_lattice_*.svg; run with --ignored to regenerate"]
    fn ctc_loss_lattice_svg_export() {
        let token_map = TokenMap::new(VOCAB);
        let out_dir = outputs_dir();
        std::fs::create_dir_all(&out_dir).expect("create outputs/");

        for (seq_idx, word) in FIXTURE_SEQS.iter().enumerate() {
            let (trace, row_chars) = forward_lattice_trace_for_word(word, &token_map);
            let filename = format!("ctc_loss_lattice_{seq_idx:02}.svg");
            let out = out_dir.join(&filename);
            CtcForwardLatticeViz::write_svg(&trace, &out, Some(row_chars.as_slice()))
                .unwrap_or_else(|e| panic!("write svg {}: {e}", out.display()));
            let shown = out.canonicalize().unwrap_or_else(|_| out.clone());
            println!("wrote {}  (seq[{seq_idx:02}] = {:?})", shown.display(), word);
        }
    }
}
