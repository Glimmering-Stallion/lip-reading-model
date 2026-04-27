//! CTC Decode prefix beam visualization: beam snapshots, ASCII printout and SVG export of DAG.
//!
//! Captures beam search state after each timestep (`N = 1` single sample batch) and derives parent → child edges
//! from CTC prefix structure (stay or one-token extend). A shared [`GraphLayout`] feeds both
//! renderers so the column packing and rank ordering are identical.
//!
//! **ASCII printout:** vertical DAG with box-drawing pipe characters (`│`, `├`, `┬`, `┐`, etc.), beam rank
//! left-to-right (with 0-th rank holding best prefix path), time top-to-bottom.
//! Best-path brackets `[..]` vs other hypotheses `(..)`.
//!
//! **SVG export:** cubic Bézier edges between prefix-node boxes and top-K emission token chip boxes. Node
//! fill/border desaturates green → gray across increasing beam ranks; chip fill/border desaturates lavender →
//! gray across decreasing emission probabilities. Lineage palette assigns maximally-separated hues on the color
//! spectrum per child; the decode-highlight best path is a distinct bold green always along leftmost rank 0.
//!
//! SVG visual parameters (colors, fonts, spacing, etc.) are configurable with [`PrefixBeamSvgTheme`];
//! loss-time counterpart lives in [`super::forward_lattice_viz`].
//!
//! # Quick start (fixture tests)
//!
//! 1. Edit `FIXTURE_SEQS` in the `tests` module to add or change target words/seqs.
//! 2. Adjust `FIXTURE_LOGITS_SEED` or per-call seeds to vary the synthetic top-K orderings.
//! 3. Run the ASCII printout:
//!    ```sh
//!    cargo test ctc_decode_beam_ascii_printout -- --nocapture
//!    ```
//! 4. Run the SVG export (to `LRM Rust/outputs/`):
//!    ```sh
//!    cargo test ctc_decode_beam_svg_export -- --ignored --nocapture
//!    ```
//! Codeveloped with Claude Opus 4.6.



pub use crate::ctc::ctc_decode::BeamHypothesisSnapshot;

use super::SVG_UI_FONT;
use crate::ctc::ctc_decode::{Beam, Prefix, CtcDecodeType, CtcDecoder};
use crate::vocab::TokenMap;
use burn::tensor::{activation::log_softmax, backend::Backend, Tensor};
use plotters::style::RGBColor;

const LEGEND_STRIP_CELLS: i32 = 6;

// ---------------------------------------------------------------------------
// trace struct
// ---------------------------------------------------------------------------

/// Metadata plus decoded result and beam snapshots from a single forward pass (`N = 1`).
#[derive(Clone, Debug)]
pub struct PrefixBeamTrace {
    pub blank_id: usize,
    pub beam_width: usize,
    pub timesteps: usize,
    pub vocab_size: usize,
    pub decoded: Vec<i64>,
    /// Per-frame argmax token id on log-softmax (`length == timesteps`), no CTC collapse.
    pub greedy_argmax_per_t: Vec<i64>,
    /// Top-K emission ids at each frame (same K as beam extend: `min(beam_width, vocab_size - 1)`),
    /// sorted by log-prob descending. Length `timesteps`; frame `t` feeds `steps[t] → steps[t+1]`.
    pub top_k_emissions_per_t: Vec<Vec<i64>>,
    /// `steps[0]` = beam before processing frame 0; `steps[t + 1]` = beam after processing frame `t`.
    /// Length is always `timesteps + 1`.
    pub steps: Vec<Vec<BeamHypothesisSnapshot>>,
}

impl PrefixBeamTrace {
    pub fn decoded_ids(&self) -> &[i64] { &self.decoded }
    /// Beam after initial state, then after each frame: length `timesteps + 1`.
    pub fn steps(&self) -> &[Vec<BeamHypothesisSnapshot>] { &self.steps }
}



// ---------------------------------------------------------------------------
// edge derivation
// ---------------------------------------------------------------------------

/// Edge between ranked beam entries at consecutive snapshots.
#[derive(Clone, Debug)]
struct BeamEdge {
    parent_rank: usize,
    child_rank: usize,
}

/// Derives edges between `steps[si]` and `steps[si+1]` using CTC prefix structure:
/// child.seq == parent.seq (stay) or child.seq[..len-1] == parent.seq (extend by one token).
fn derive_edges(parent_step: &[BeamHypothesisSnapshot], child_step: &[BeamHypothesisSnapshot]) -> Vec<BeamEdge> {
    let mut edges = Vec::new();
    for (cr, child) in child_step.iter().enumerate() {
        let cseq = &child.sequence;
        for (pr, parent) in parent_step.iter().enumerate() {
            let pseq = &parent.sequence;
            let is_stay = cseq == pseq;
            let is_extend = !is_stay
                && cseq.len() == pseq.len() + 1
                && cseq[..pseq.len()] == pseq[..];
            if is_stay || is_extend {
                edges.push(BeamEdge { parent_rank: pr, child_rank: cr });
            }
        }
    }
    edges
}

/// All inter-step edges for the full trace. `result[si]` connects `steps[si]` → `steps[si+1]`.
fn derive_all_edges(steps: &[Vec<BeamHypothesisSnapshot>]) -> Vec<Vec<BeamEdge>> {
    if steps.len() <= 1 {
        return Vec::new();
    }
    (0..steps.len() - 1)
        .map(|si| derive_edges(&steps[si], &steps[si + 1]))
        .collect()
}



// ---------------------------------------------------------------------------
// trace capture
// ---------------------------------------------------------------------------

/// Converts beam prefixes to sorted snapshots (descending by combined score).
fn beam_to_snapshots(prefixes: &[Prefix]) -> Vec<BeamHypothesisSnapshot> {
    let mut v: Vec<BeamHypothesisSnapshot> = prefixes
        .iter()
        .map(|p| BeamHypothesisSnapshot {
            sequence: p.sequence.clone(),
            log_prob_blank: p.log_prob_blank,
            log_prob_non_blank: p.log_prob_non_blank,
            combined_log_prob: p.combined_log_prob,
        })
        .collect();
    v.sort_by(|a, b| {
        b.combined_log_prob
            .partial_cmp(&a.combined_log_prob)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    v
}

/// Prefix beam search with per-step recording, batch size 1 only.
fn trace_beam_sample0_inner<B: Backend>(
    decoder: &CtcDecoder,
    logits: Tensor<B, 3>,
) -> PrefixBeamTrace {
    let [_n, t, v] = logits.dims();
    let w = decoder.beam_width;
    let k = w.min(v - 1);

    let batch_log_probs = log_softmax(logits, 2)
        .slice([0..1, 0..t, 0..v])
        .squeeze::<2>();
    let mut log_probs = batch_log_probs
        .to_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap();

    let mut beam = Beam::new();
    let mut steps = vec![beam_to_snapshots(beam.hypotheses())];
    let mut top_k_pairs: Vec<(usize, f32)> = vec![(0, 0.0); v];
    let mut greedy_argmax_per_t = Vec::with_capacity(t);
    let mut top_k_emissions_per_t: Vec<Vec<i64>> = Vec::with_capacity(t);

    for t_idx in 0..t {
        let t_chunk = t_idx * v;
        // Greedy preview: argmax on the full softmax row (blank allowed), before blank masking for beam extend.
        let row = &log_probs[t_chunk..(t_chunk + v)];
        let argmax_id = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as i64)
            .unwrap_or(0);
        greedy_argmax_per_t.push(argmax_id);

        let log_probs_t = &mut log_probs[t_chunk..(t_chunk + v)];
        let log_prob_blank = log_probs_t[decoder.blank_id];
        log_probs_t[decoder.blank_id] = f32::NEG_INFINITY;
        decoder.select_top_k_pairs(log_probs_t, k, &mut top_k_pairs);
        let mut tops: Vec<(usize, f32)> = top_k_pairs[..k].to_vec();
        tops.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        top_k_emissions_per_t.push(tops.iter().map(|(id, _)| *id as i64).collect());

        let extended_prefix = decoder.prefix_extend_step(
            beam.into_hypotheses(),
            &top_k_pairs[..k],
            log_prob_blank,
            decoder.lm.as_deref(),
        );
        beam = Beam::from_hypotheses(decoder.beam_prune_step(extended_prefix, w, decoder.lm_alpha, decoder.lm_beta));
        steps.push(beam_to_snapshots(beam.hypotheses()));
    }

    let decoded = decoder.select_best_prefix(beam.into_hypotheses());
    debug_assert_eq!(steps.len(), t + 1);
    debug_assert_eq!(greedy_argmax_per_t.len(), t);
    PrefixBeamTrace {
        blank_id: decoder.blank_id,
        beam_width: w,
        timesteps: t,
        vocab_size: v,
        decoded,
        greedy_argmax_per_t,
        top_k_emissions_per_t,
        steps,
    }
}



// ---------------------------------------------------------------------------
// display helpers
// ---------------------------------------------------------------------------

fn sequence_display(seq: &[usize], token_map: Option<&TokenMap>) -> String {
    if seq.is_empty() {
        return "\u{03B5}".to_string();
    }
    match token_map {
        Some(m) => {
            let s: String = seq.iter().filter_map(|&id| m.char_of(id)).collect();
            if s.is_empty() { format!("{seq:?}") } else { s }
        }
        None => format!("{seq:?}"),
    }
}

fn decoded_display(decoded: &[i64], token_map: Option<&TokenMap>) -> String {
    let u: Vec<usize> = decoded.iter().map(|&x| x as usize).collect();
    sequence_display(&u, token_map)
}

const GREEDY_ARGMAX_PREVIEW: usize = 32;

fn token_id_display(id: usize, blank_id: usize, token_map: Option<&TokenMap>) -> String {
    if id == blank_id {
        return "_".to_string();
    }
    match token_map {
        Some(m) => m.char_of(id).map(|c| c.to_string()).unwrap_or_else(|| format!("{id}")),
        None => format!("{id}"),
    }
}

fn format_greedy_argmax_logits_line(trace: &PrefixBeamTrace, token_map: Option<&TokenMap>) -> String {
    let ids = &trace.greedy_argmax_per_t;
    let blank_id = trace.blank_id;
    if ids.is_empty() {
        return "Greedy Argmax Logits: []\n".to_string();
    }
    let n = ids.len();
    let show = n.min(GREEDY_ARGMAX_PREVIEW);
    let mut inner = String::new();
    for (i, &id) in ids.iter().take(show).enumerate() {
        if i > 0 { inner.push_str(", "); }
        inner.push_str(&token_id_display(id as usize, blank_id, token_map));
    }
    if n > show {
        inner.push_str(&format!(", … (+{} more)", n - show));
    }
    format!("Greedy Argmax Logits: [{inner}]\n")
}



// ---------------------------------------------------------------------------
// color helpers (HSL, lineage palette, picker, propagation)
// ---------------------------------------------------------------------------

fn rgb_sq_dist(a: (u8, u8, u8), b: (u8, u8, u8)) -> u32 {
    let dr = i32::from(a.0) - i32::from(b.0);
    let dg = i32::from(a.1) - i32::from(b.1);
    let db = i32::from(a.2) - i32::from(b.2);
    (dr * dr + dg * dg + db * db) as u32
}

/// `h` in [0, 1), `s` and `l` in [0, 1].
fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (u8, u8, u8) {
    let h = h.rem_euclid(1.0);
    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;
    let hue_to_rgb = |mut t: f64| {
        t = t.rem_euclid(1.0);
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };
    let r = hue_to_rgb(h + 1.0 / 3.0);
    let g = hue_to_rgb(h);
    let b = hue_to_rgb(h - 1.0 / 3.0);
    (
        (r.clamp(0.0, 1.0) * 255.0).round() as u8,
        (g.clamp(0.0, 1.0) * 255.0).round() as u8,
        (b.clamp(0.0, 1.0) * 255.0).round() as u8,
    )
}

/// sRGB u8 → HSL with h in [0, 1), s and l in [0, 1].
fn rgb_to_hsl(r: u8, g: u8, b: u8) -> (f64, f64, f64) {
    let rf = r as f64 / 255.0;
    let gf = g as f64 / 255.0;
    let bf = b as f64 / 255.0;
    let max = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
    let l = (max + min) * 0.5;
    if (max - min) < 1e-9 {
        return (0.0, 0.0, l);
    }
    let d = max - min;
    let s = if l > 0.5 { d / (2.0 - max - min) } else { d / (max + min) };
    let h = if (max - rf).abs() < 1e-9 {
        (gf - bf) / d + if gf < bf { 6.0 } else { 0.0 }
    } else if (max - gf).abs() < 1e-9 {
        (bf - rf) / d + 2.0
    } else {
        (rf - gf) / d + 4.0
    };
    ((h / 6.0).rem_euclid(1.0), s, l)
}

/// Scales saturation of an sRGB u8 triple toward a target neutral lightness.
/// At `sat_scale = 1` the color is unchanged; at `sat_scale = 0` it becomes a pure gray
/// at `desat_l`. This ensures different hues converge to the same gray endpoint.
fn desaturate_rgb8(r: u8, g: u8, b: u8, sat_scale: f64, desat_l: f64) -> (u8, u8, u8) {
    let (h, s, l) = rgb_to_hsl(r, g, b);
    let t = sat_scale.clamp(0.0, 1.0);
    let new_s = s * t;
    let new_l = l * t + desat_l * (1.0 - t);
    hsl_to_rgb(h, new_s.clamp(0.0, 1.0), new_l.clamp(0.0, 1.0))
}

/// Evenly-spaced hues with alternating lightness; sized to `n` (caller passes `beam_width + 3`).
/// When `shuffle_seed` is `Some`, the palette is deterministically shuffled via Fisher-Yates
/// so that different traces get distinct color orderings while remaining reproducible.
fn build_lineage_palette(
    n: usize,
    edge_rgb_best: (u8, u8, u8),
    min_dist_sq: u32,
    shuffle_seed: Option<u64>,
) -> Vec<(u8, u8, u8)> {
    let want = n.max(4).min(48);
    let golden = 0.618_033_988_749_895;
    let mut out: Vec<(u8, u8, u8)> = Vec::with_capacity(want);
    for i in 0..want {
        let mut h = ((i as f64) / (want as f64)).rem_euclid(1.0);
        let s = 0.68;
        let l = if i % 2 == 0 { 0.44 } else { 0.56 };
        let mut rgb = hsl_to_rgb(h, s, l);
        let mut bump = 0u32;
        while rgb_sq_dist(rgb, edge_rgb_best) < min_dist_sq && bump < 48 {
            h = (h + golden / (want as f64)).rem_euclid(1.0);
            rgb = hsl_to_rgb(h, s, l);
            bump += 1;
        }
        out.push(rgb);
    }
    if let Some(seed) = shuffle_seed {
        let mut s = seed;
        for i in (1..out.len()).rev() {
            s = s.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x6A09E667F3BCC908);
            let j = (s >> 33) as usize % (i + 1);
            out.swap(i, j);
        }
    }
    out
}

/// Palette entry that maximizes minimum RGB distance to `forbidden` (and stays ≥ `min_dist_sq`
/// from `edge_rgb_best`). Falls back to the least-bad entry when nothing fully qualifies.
fn pick_lineage_color(
    palette: &[(u8, u8, u8)],
    forbidden: &[(u8, u8, u8)],
    edge_rgb_best: (u8, u8, u8),
    min_dist_sq: u32,
    fallback_rgb: (u8, u8, u8),
    si: usize,
    cr: usize,
) -> (u8, u8, u8) {
    if palette.is_empty() {
        return fallback_rgb;
    }
    let mut best: Option<(u32, u32, usize, (u8, u8, u8))> = None;
    for (idx, &c) in palette.iter().enumerate() {
        if forbidden.contains(&c) {
            continue;
        }
        let d_best = rgb_sq_dist(c, edge_rgb_best);
        if d_best < min_dist_sq {
            continue;
        }
        let min_fb = forbidden.iter().map(|&f| rgb_sq_dist(c, f)).min().unwrap_or(u32::MAX);
        let replace = match best {
            None => true,
            Some((best_fb, best_db, best_i, _)) => {
                min_fb > best_fb
                    || (min_fb == best_fb && d_best > best_db)
                    || (min_fb == best_fb && d_best == best_db && idx < best_i)
            }
        };
        if replace {
            best = Some((min_fb, d_best, idx, c));
        }
    }
    if let Some((_, _, _, c)) = best {
        return c;
    }
    // last resort: any entry far enough from decode-highlight green
    for &c in palette {
        if rgb_sq_dist(c, edge_rgb_best) >= min_dist_sq {
            return c;
        }
    }
    palette[(si * 31 + cr) % palette.len()]
}

/// `lineage_rgb[si][r]` = RGB carried **out of** node `(si, r)`.
///
/// **Stay-inherit:** if a child has the same sequence as a parent (blank/same-char stay), it
/// inherits that parent's color — the lineage is continuous, not a merge.
///
/// **Row-used awareness:** all colors assigned at `si+1` accumulate in a per-transition
/// `row_used` set, which is included in the forbidden list for every subsequent pick so that
/// siblings, dead-lineage colors, and fan-out branches stay visually distinct within a row.
fn compute_lineage_rgb_per_node(
    steps: &[Vec<BeamHypothesisSnapshot>],
    edges: &[Vec<BeamEdge>],
    beam_width: usize,
    palette: &[(u8, u8, u8)],
    fallback_rgb: (u8, u8, u8),
    edge_rgb_best: (u8, u8, u8),
    min_dist_sq: u32,
) -> Vec<Vec<Option<(u8, u8, u8)>>> {
    let sn = steps.len();
    let w = beam_width;
    let plen = palette.len().max(1);
    let mut lineage = vec![vec![None; w]; sn];

    for r in 0..w {
        if steps.first().is_some_and(|s| s.get(r).is_some()) {
            lineage[0][r] = Some(palette[r % plen]);
        }
    }

    for si in 0..sn.saturating_sub(1) {
        let step_edges = &edges[si];

        // seed forbidden list with all colors visible in the current row
        let mut row_used: Vec<(u8, u8, u8)> = Vec::new();
        for r in 0..w {
            if let Some(col) = lineage[si][r] {
                if !row_used.contains(&col) {
                    row_used.push(col);
                }
            }
        }

        // fan-out colors: parent with >1 child gets a distinct color per non-stay child
        let mut fanout_child_rgb: std::collections::HashMap<(usize, usize), (u8, u8, u8)> =
            std::collections::HashMap::new();
        for pr in 0..w {
            let parent_seq = match steps[si].get(pr) {
                Some(h) => &h.sequence,
                None => continue,
            };
            let mut ch: Vec<usize> = step_edges
                .iter()
                .filter(|e| e.parent_rank == pr)
                .map(|e| e.child_rank)
                .collect();
            ch.sort_unstable();
            ch.dedup();
            if ch.len() <= 1 {
                continue;
            }
            // exclude stay-children (they inherit the parent color)
            let fan_ch: Vec<usize> = ch.into_iter().filter(|&cr| {
                steps[si + 1].get(cr).map_or(true, |child| child.sequence != *parent_seq)
            }).collect();
            if fan_ch.is_empty() {
                continue;
            }
            let mut forbidden: Vec<(u8, u8, u8)> = vec![edge_rgb_best];
            forbidden.extend_from_slice(&row_used);
            if let Some(pc) = lineage[si][pr] {
                forbidden.push(pc);
            }
            for &cr in &fan_ch {
                let col = pick_lineage_color(palette, &forbidden, edge_rgb_best, min_dist_sq, fallback_rgb, si, cr);
                fanout_child_rgb.insert((pr, cr), col);
                forbidden.push(col);
                row_used.push(col);
            }
        }

        for cr in 0..w {
            let child_snap = match steps.get(si + 1).and_then(|s| s.get(cr)) {
                Some(h) => h,
                None => continue,
            };
            let parents: Vec<usize> = step_edges
                .iter()
                .filter(|e| e.child_rank == cr)
                .map(|e| e.parent_rank)
                .collect();
            if parents.is_empty() {
                continue;
            }

            // stay-inherit: if any parent has the same sequence as the child, this is a
            // lineage continuation — inherit that parent's color unconditionally.
            let stay_parent = parents.iter().find(|&&pr| {
                steps[si].get(pr).map_or(false, |p| p.sequence == child_snap.sequence)
            });
            if let Some(&pr) = stay_parent {
                let col = lineage[si][pr].unwrap_or(fallback_rgb);
                lineage[si + 1][cr] = Some(col);
                if !row_used.contains(&col) {
                    row_used.push(col);
                }
                continue;
            }

            let resolved: Vec<(u8, u8, u8)> = parents
                .iter()
                .map(|&pr| lineage[si][pr].unwrap_or(fallback_rgb))
                .collect();
            let mut distinct: Vec<(u8, u8, u8)> = Vec::new();
            for c in resolved {
                if !distinct.contains(&c) {
                    distinct.push(c);
                }
            }

            let col = if distinct.len() > 1 {
                let mut forbidden: Vec<(u8, u8, u8)> = distinct.clone();
                forbidden.push(edge_rgb_best);
                forbidden.extend_from_slice(&row_used);
                pick_lineage_color(palette, &forbidden, edge_rgb_best, min_dist_sq, fallback_rgb, si, cr)
            } else if parents.len() == 1 {
                let pr = parents[0];
                fanout_child_rgb.get(&(pr, cr)).copied().unwrap_or(distinct[0])
            } else {
                distinct[0]
            };
            lineage[si + 1][cr] = Some(col);
            if !row_used.contains(&col) {
                row_used.push(col);
            }
        }
    }

    lineage
}



// ---------------------------------------------------------------------------
// shared layout: consumed by both SVG and ASCII renderers
// ---------------------------------------------------------------------------

/// Shared graph layout for SVG and ASCII renderers.
struct GraphLayout {
    labels: Vec<Vec<String>>,
    scores: Vec<Vec<f32>>,
    edges: Vec<Vec<BeamEdge>>,
    best_ranks: Vec<Option<usize>>,
    beam_width: usize,
    steps_n: usize,
}

impl GraphLayout {
    fn from_trace(trace: &PrefixBeamTrace, token_map: Option<&TokenMap>) -> Self {
        let steps = trace.steps();
        let steps_n = steps.len();
        let w = trace.beam_width;
        let edges = derive_all_edges(steps);
        let decoded_seq: Vec<usize> = trace.decoded_ids().iter().map(|&x| x as usize).collect();
        let best_ranks: Vec<Option<usize>> = steps
            .iter()
            .map(|step| {
                step.iter().position(|h| {
                    decoded_seq.starts_with(&h.sequence) || h.sequence == decoded_seq
                })
            })
            .collect();

        let mut labels = Vec::with_capacity(steps_n);
        let mut scores = Vec::with_capacity(steps_n);
        for step in steps {
            let mut col_labels = Vec::with_capacity(w);
            let mut col_scores = Vec::with_capacity(w);
            for r in 0..w {
                if let Some(h) = step.get(r) {
                    col_labels.push(sequence_display(&h.sequence, token_map));
                    col_scores.push(h.combined_log_prob);
                } else {
                    col_labels.push(String::new());
                    col_scores.push(f32::NAN);
                }
            }
            labels.push(col_labels);
            scores.push(col_scores);
        }

        GraphLayout { labels, scores, edges, best_ranks, beam_width: w, steps_n }
    }

    fn is_on_best_path(&self, si: usize, rank: usize) -> bool {
        self.best_ranks.get(si).copied().flatten() == Some(rank)
    }

    fn is_edge_on_best_path(&self, si: usize, edge: &BeamEdge) -> bool {
        self.is_on_best_path(si, edge.parent_rank) && self.is_on_best_path(si + 1, edge.child_rank)
    }
}



// ---------------------------------------------------------------------------
// ascii graph renderer (vertical, box-drawing)
// ---------------------------------------------------------------------------

const BOX_H: char = '\u{2500}';
const BOX_V: char = '\u{2502}';
const BOX_TL: char = '\u{250C}';
const BOX_TR: char = '\u{2510}';
const BOX_BL: char = '\u{2514}';
const BOX_BR: char = '\u{2518}';
const BOX_LT: char = '\u{251C}';
const BOX_RT: char = '\u{2524}';
const BOX_TT: char = '\u{252C}';
const BOX_BT: char = '\u{2534}';
const BOX_X: char = '\u{253C}';

const DIR_UP: u8 = 0b0001;
const DIR_DOWN: u8 = 0b0010;
const DIR_LEFT: u8 = 0b0100;
const DIR_RIGHT: u8 = 0b1000;

fn dirs_to_box_char(d: u8) -> char {
    match d {
        0b0011 => BOX_V,
        0b1100 => BOX_H,
        0b1001 => BOX_BL,
        0b0101 => BOX_BR,
        0b1010 => BOX_TL,
        0b0110 => BOX_TR,
        0b1011 => BOX_LT,
        0b0111 => BOX_RT,
        0b1110 => BOX_TT,
        0b1101 => BOX_BT,
        0b1111 => BOX_X,
        _ => ' ',
    }
}

struct TransitionData {
    same_ranks: Vec<usize>,
    cross: Vec<[usize; 4]>,
    edge_layer: Vec<usize>,
    num_layers: usize,
    edge_parent_xoff: Vec<usize>,
    edge_child_xoff: Vec<usize>,
    max_xoff: Vec<usize>,
}

fn compute_transition(edges: &[BeamEdge], beam_width: usize) -> TransitionData {
    let mut same_ranks: Vec<usize> = Vec::new();
    let mut cross: Vec<[usize; 4]> = Vec::new();
    for edge in edges {
        if edge.parent_rank == edge.child_rank {
            same_ranks.push(edge.parent_rank);
        } else {
            let lo = edge.parent_rank.min(edge.child_rank);
            let hi = edge.parent_rank.max(edge.child_rank);
            cross.push([edge.parent_rank, edge.child_rank, lo, hi]);
        }
    }

    let mut edge_layer = vec![0usize; cross.len()];
    let mut layer_spans: Vec<Vec<(usize, usize, usize)>> = Vec::new();
    for (ei, ce) in cross.iter().enumerate() {
        let [p, _c, lo, hi] = *ce;
        let mut placed = false;
        for (li, spans) in layer_spans.iter_mut().enumerate() {
            let conflicts = spans.iter().any(|&(lo2, hi2, p2)| lo <= hi2 && lo2 <= hi && p != p2);
            if !conflicts {
                spans.push((lo, hi, p));
                edge_layer[ei] = li;
                placed = true;
                break;
            }
        }
        if !placed {
            edge_layer[ei] = layer_spans.len();
            layer_spans.push(vec![(lo, hi, p)]);
        }
    }
    let num_layers = layer_spans.len().max(1);

    let edge_parent_xoff = vec![1usize; cross.len()];
    let mut edge_child_xoff = vec![1usize; cross.len()];

    for r in 0..beam_width {
        let mut left: Vec<(usize, usize)> = Vec::new();
        let mut right: Vec<(usize, usize)> = Vec::new();
        for (ei, ce) in cross.iter().enumerate() {
            if ce[1] != r { continue; }
            let dist = (ce[0] as isize - r as isize).unsigned_abs();
            if ce[0] < r { left.push((ei, dist)); } else { right.push((ei, dist)); }
        }
        if left.is_empty() && right.is_empty() { continue; }

        left.sort_by_key(|&(_, d)| d);
        right.sort_by_key(|&(_, d)| d);

        if let Some(&(ei, _)) = left.first() {
            edge_child_xoff[ei] = 0;
        }

        let mut right_side: Vec<(usize, usize)> = Vec::new();
        right_side.extend_from_slice(&right);
        right_side.extend(left.iter().skip(1));
        right_side.sort_by_key(|&(_, d)| d);

        let mut off = 2;
        for (ei, _) in right_side {
            edge_child_xoff[ei] = off;
            off += 1;
        }
    }

    // snap-to-center
    let has_same_rank: Vec<bool> = (0..beam_width).map(|r| same_ranks.contains(&r)).collect();
    let mut max_parent_layer: Vec<Option<usize>> = vec![None; beam_width];
    for (ei, ce) in cross.iter().enumerate() {
        let p = ce[0];
        let el = edge_layer[ei];
        max_parent_layer[p] = Some(max_parent_layer[p].map_or(el, |prev| prev.max(el)));
    }

    let mut center_taken = vec![false; beam_width];
    let mut snap_order: Vec<(usize, usize, usize)> = cross
        .iter()
        .enumerate()
        .map(|(ei, ce)| {
            let dist = (ce[0] as isize - ce[1] as isize).unsigned_abs();
            (ei, ce[1], dist)
        })
        .collect();
    snap_order.sort_by_key(|&(_, _, d)| d);

    for (ei, r, _) in snap_order {
        if center_taken[r] || has_same_rank[r] { continue; }
        let li = edge_layer[ei];
        let free = match max_parent_layer[r] {
            None => true,
            Some(mpl) => li > mpl,
        };
        if free {
            edge_child_xoff[ei] = 1;
            center_taken[r] = true;
        }
    }

    // backfill after snap
    for r in 0..beam_width {
        let off0_used = cross.iter().enumerate()
            .any(|(ei, ce)| ce[1] == r && edge_child_xoff[ei] == 0);
        if off0_used { continue; }
        let mut best: Option<(usize, usize)> = None;
        for (ei, ce) in cross.iter().enumerate() {
            if ce[1] != r || ce[0] >= r { continue; }
            if edge_child_xoff[ei] <= 1 { continue; }
            let d = (ce[0] as isize - r as isize).unsigned_abs();
            if best.map_or(true, |(_, bd)| d < bd) {
                best = Some((ei, d));
            }
        }
        if let Some((ei, _)) = best {
            edge_child_xoff[ei] = 0;
        }
    }

    let mut max_xoff = vec![1usize; beam_width];
    for (ei, ce) in cross.iter().enumerate() {
        max_xoff[ce[0]] = max_xoff[ce[0]].max(edge_parent_xoff[ei]);
        max_xoff[ce[1]] = max_xoff[ce[1]].max(edge_child_xoff[ei]);
    }

    TransitionData { same_ranks, cross, edge_layer, num_layers, edge_parent_xoff, edge_child_xoff, max_xoff }
}

struct BeamRankCharLayout {
    margin_w: usize,
    col_x: Vec<usize>,
    col_w: Vec<usize>,
    line_w: usize,
    transitions: Vec<TransitionData>,
}

fn beam_rank_char_layout(layout: &GraphLayout) -> BeamRankCharLayout {
    let w = layout.beam_width;
    let sn = layout.steps_n;
    let transitions: Vec<TransitionData> = (0..sn.saturating_sub(1))
        .map(|si| compute_transition(&layout.edges[si], w))
        .collect();

    let mut global_max_xoff = vec![1usize; w];
    for td in &transitions {
        for r in 0..w {
            global_max_xoff[r] = global_max_xoff[r].max(td.max_xoff[r]);
        }
    }

    let gap = 2usize;
    let margin_w = {
        let init_len = "init".len();
        let last_len = if sn > 1 { format!("t = {}", sn - 2).len() } else { 0 };
        init_len.max(last_len) + 2
    };

    let mut max_col = 3usize;
    for r in 0..w {
        for si in 0..sn {
            let label_len = layout.labels[si][r].chars().count() + 2;
            max_col = max_col.max(label_len);
        }
        max_col = max_col.max(global_max_xoff[r] + 1);
    }
    let col_w = vec![max_col; w];

    let mut col_x = vec![0usize; w];
    col_x[0] = margin_w;
    for r in 1..w {
        col_x[r] = col_x[r - 1] + max_col + gap;
    }
    let line_w = col_x[w - 1] + col_w[w - 1];

    BeamRankCharLayout { margin_w, col_x, col_w, line_w, transitions }
}

fn render_prefix_beam_ascii(trace: &PrefixBeamTrace, token_map: Option<&TokenMap>) -> String {
    let layout = GraphLayout::from_trace(trace, token_map);
    let w = layout.beam_width;
    let sn = layout.steps_n;
    if sn == 0 || w == 0 {
        return String::from("(empty trace)\n");
    }

    let grid = beam_rank_char_layout(&layout);
    let margin_w = grid.margin_w;
    let col_x = &grid.col_x;
    let line_w = grid.line_w;
    let transitions = &grid.transitions;

    let mut out = String::new();
    let dec_str = decoded_display(trace.decoded_ids(), token_map);
    out.push_str(&format!("Decoded Sequence: \"{dec_str}\"\n"));
    out.push_str(&format_greedy_argmax_logits_line(trace, token_map));
    out.push('\n');
    out.push_str("Rows = Timesteps, Cols = Ranks\n");
    out.push_str(&format!("Timesteps = {}, Beam width = {}\n", trace.timesteps, trace.beam_width));
    out.push_str("[..] = best path,  (..) = other hypotheses,  \u{03B5} = empty sequence\n\n");

    let mut hdr = vec![' '; line_w];
    for r in 0..w {
        let label = format!("r = {r}");
        for (i, ch) in label.chars().enumerate() {
            let x = col_x[r] + i;
            if x < line_w { hdr[x] = ch; }
        }
    }
    let hdr_line: String = hdr.iter().collect();
    out.push_str(hdr_line.trim_end());
    out.push_str("\n\n");

    for si in 0..sn {
        let mut line = vec![' '; line_w];
        let t_label = if si == 0 { "init".to_string() } else { format!("t = {}", si - 1) };
        for (i, ch) in t_label.chars().enumerate() {
            if i < margin_w { line[i] = ch; }
        }
        for r in 0..w {
            if layout.labels[si][r].is_empty() { continue; }
            let raw = &layout.labels[si][r];
            let on_best = layout.is_on_best_path(si, r);
            let cell = if on_best { format!("[{raw}]") } else { format!("({raw})") };
            for (i, ch) in cell.chars().enumerate() {
                let x = col_x[r] + i;
                if x < line_w { line[x] = ch; }
            }
        }
        let node_line: String = line.iter().collect();
        out.push_str(node_line.trim_end());
        out.push('\n');

        if si >= sn - 1 { continue; }
        let td = &transitions[si];

        for li in 0..td.num_layers {
            let mut dirs = vec![0u8; line_w];
            for &r in &td.same_ranks {
                let x = col_x[r] + 1;
                if x < line_w { dirs[x] |= DIR_UP | DIR_DOWN; }
            }
            for (ei, ce) in td.cross.iter().enumerate() {
                let el = td.edge_layer[ei];
                if el == li { continue; }
                let [p, c, _, _] = *ce;
                if li < el {
                    let x = col_x[p] + td.edge_parent_xoff[ei];
                    if x < line_w { dirs[x] |= DIR_UP | DIR_DOWN; }
                } else {
                    let x = col_x[c] + td.edge_child_xoff[ei];
                    if x < line_w { dirs[x] |= DIR_UP | DIR_DOWN; }
                }
            }
            for (ei, ce) in td.cross.iter().enumerate() {
                if td.edge_layer[ei] != li { continue; }
                let [p, c, _, _] = *ce;
                let px = col_x[p] + td.edge_parent_xoff[ei];
                let cx = col_x[c] + td.edge_child_xoff[ei];
                if px < line_w {
                    dirs[px] |= DIR_UP;
                    dirs[px] |= if cx > px { DIR_RIGHT } else { DIR_LEFT };
                }
                if cx < line_w {
                    dirs[cx] |= DIR_DOWN;
                    dirs[cx] |= if px < cx { DIR_LEFT } else { DIR_RIGHT };
                }
            }
            for (ei, ce) in td.cross.iter().enumerate() {
                if td.edge_layer[ei] != li { continue; }
                let [p, c, _, _] = *ce;
                let px = col_x[p] + td.edge_parent_xoff[ei];
                let cx = col_x[c] + td.edge_child_xoff[ei];
                let (x_lo, x_hi) = (px.min(cx), px.max(cx));
                for x in (x_lo + 1)..x_hi {
                    if x >= line_w { break; }
                    let full_vert = (dirs[x] & DIR_UP) != 0 && (dirs[x] & DIR_DOWN) != 0;
                    if !full_vert {
                        dirs[x] |= DIR_LEFT | DIR_RIGHT;
                    }
                }
            }

            let mut r_line = vec![' '; margin_w];
            for &d in &dirs[margin_w..] {
                r_line.push(dirs_to_box_char(d));
            }
            let routing: String = r_line.iter().collect();
            out.push_str(routing.trim_end());
            out.push('\n');
        }
    }

    out
}



// ---------------------------------------------------------------------------
// svg theme + default
// ---------------------------------------------------------------------------

/// Pixel layout, typography, colors, and SVG-only effects for the prefix-beam DAG export.
///
/// SVG text uses a system UI font stack and soft gray (`text_rgb`) rather than pure black.
///
/// To match [`CtcPrefixBeamViz::write_svg`], use [`Default`]. Override fields and pass the value to
/// [`CtcPrefixBeamViz::write_svg_with_theme`]. Naming parallels [`super::ForwardLatticeSvgTheme`]
/// (`axes_to_heatmap_margin`, mirrored tick bands, title stack above the drawable region).
#[derive(Clone, Copy, Debug)]
pub struct PrefixBeamSvgTheme {
    pub char_sx: f64,
    pub node_row_h: u32,
    pub node_ry: u32,
    pub margin_top: u32,
    pub margin_bottom: u32,
    pub margin_left: u32,
    pub margin_right: u32,
    /// Clearance between DAG content and axis-adjacent labels (see forward lattice `axes_to_heatmap_margin`).
    pub axes_to_heatmap_margin: u32,
    pub title_h: u32,
    pub title_cluster_lift_px: u32,
    pub x_axis_tick_band_h: u32,
    pub time_tick_band_w: u32,
    pub time_axis_label_w: u32,
    pub time_axis_label_x_offset: i32,
    pub time_axis_label_y_offset: i32,
    pub rank_axis_label_y_offset: i32,
    pub rank_axis_label_x_offset: i32,
    pub axis_origin_pad: u32,
    pub node_label_fs: f64,
    pub node_h_pad: u32,
    pub node_min_half_w: u32,
    pub node_label_dy: i32,
    pub node_score_dy: i32,
    pub cand_margin_v: u32,
    pub chip_w: u32,
    pub chip_h: u32,
    pub chip_h_gap: u32,
    pub title_fs: f64,
    pub time_label_fs: f64,
    pub rank_header_fs: f64,
    pub col_half_min: i32,
    pub chip_symbol_max_chars: usize,
    pub edge_rgb_normal: (u8, u8, u8),
    pub edge_rgb_best: (u8, u8, u8),
    pub edge_alpha_normal: f64,
    /// Multiplier on `edge_alpha_normal` for lineage-colored strokes (non-decode-highlight edges).
    pub lineage_edge_alpha_mul: f64,
    pub edge_alpha_best: f64,
    pub edge_stroke_w_normal: u32,
    pub edge_stroke_w_best: u32,
    pub fan_edge_rgb: (u8, u8, u8),
    pub fan_edge_alpha: f64,
    pub fan_edge_stroke_w: u32,
    pub node_score_rgb: (u8, u8, u8),
    /// Diagram copy: title, subtitle, prefix labels, chip symbols, timestep ticks, rank headers.
    pub text_rgb: (u8, u8, u8),
    pub chip_border_stroke_w: u32,
    pub node_border_stroke_w_normal: u32,
    pub node_border_stroke_w_best: u32,
    pub node_corner_rx: u32,
    pub chip_corner_rx: u32,
    pub ghost_token_strip_pad: i32,
    pub ghost_strip_feather_sigma: f64,
    pub ghost_dash_alpha_mul: f64,
    pub ghost_stroke_dash: (u32, u32),
    /// Minimum squared RGB distance between lineage palette entries and `edge_rgb_best`.
    pub lineage_min_rgb_dist_from_best_sq: u32,
    /// Font size for legend labels (beam rank, emission order, edge key).
    pub legend_fs: f64,
    /// Side length of each legend color swatch (px).
    pub legend_swatch_size: u32,
}

impl Default for PrefixBeamSvgTheme {
    fn default() -> Self {
        Self {
            char_sx: 7.28,
            node_row_h: 52,
            node_ry: 18,
            margin_top: 200,
            margin_bottom: 30,
            margin_left: 75,
            margin_right: 40,
            axes_to_heatmap_margin: 20,
            title_h: 22,
            title_cluster_lift_px: 100,
            x_axis_tick_band_h: 32,
            time_tick_band_w: 32,
            time_axis_label_w: 22,
            time_axis_label_x_offset: 0,
            time_axis_label_y_offset: 0,
            rank_axis_label_y_offset: 0,
            rank_axis_label_x_offset: 0,
            axis_origin_pad: 4,
            node_label_fs: 10.0,
            node_h_pad: 6,
            node_min_half_w: 18,
            node_label_dy: -5,
            node_score_dy: 7,
            cand_margin_v: 10,
            chip_w: 28,
            chip_h: 16,
            chip_h_gap: 6,
            title_fs: 20.0,
            time_label_fs: 16.0,
            rank_header_fs: 16.0,
            col_half_min: 6,
            chip_symbol_max_chars: 2,
            edge_rgb_normal: (160, 160, 180),
            edge_rgb_best: (32, 130, 72),
            edge_alpha_normal: 0.52,
            lineage_edge_alpha_mul: 0.82,
            edge_alpha_best: 0.88,
            edge_stroke_w_normal: 1,
            edge_stroke_w_best: 2,
            fan_edge_rgb: (150, 150, 170),
            fan_edge_alpha: 0.09,
            fan_edge_stroke_w: 1,
            node_score_rgb: (105, 110, 120),
            text_rgb: (100, 106, 118),
            chip_border_stroke_w: 1,
            node_border_stroke_w_normal: 1,
            node_border_stroke_w_best: 2,
            node_corner_rx: 7,
            chip_corner_rx: 5,
            ghost_token_strip_pad: 6,
            ghost_strip_feather_sigma: 4.0,
            ghost_dash_alpha_mul: 0.38,
            ghost_stroke_dash: (3, 5),
            lineage_min_rgb_dist_from_best_sq: 35 * 35,
            legend_fs: 10.0,
            legend_swatch_size: 10,
        }
    }
}



// ---------------------------------------------------------------------------
// svg helpers + renderer
// ---------------------------------------------------------------------------

fn svg_sans_text_width_px(char_count: usize, font_pt: f64) -> u32 {
    if char_count == 0 { return 0; }
    ((char_count as f64) * 0.55 * font_pt).ceil().max(1.0) as u32
}

fn svg_node_half_width(raw_label: &str, score: f32, fs: f64, h_pad: u32, min_half: u32) -> u32 {
    let label_line = format!("\"{raw_label}\"");
    let score_line = format!("{:.2}", score);
    let inner = svg_sans_text_width_px(label_line.chars().count(), fs)
        .max(svg_sans_text_width_px(score_line.chars().count(), fs));
    let full = inner + 2 * h_pad;
    ((full + 1) / 2).max(min_half)
}

fn svg_char_col_to_x(origin_x: i32, col: usize, sx: f64) -> i32 {
    origin_x + (col as f64 * sx).round() as i32
}

fn svg_rank_column_center_x(origin_x: i32, grid: &BeamRankCharLayout, r: usize, sx: f64) -> i32 {
    origin_x + ((grid.col_x[r] as f64 + grid.col_w[r] as f64 / 2.0) * sx).round() as i32
}

fn svg_beam_center_x(origin_x: i32, grid: &BeamRankCharLayout, beam_width: usize, sx: f64) -> i32 {
    let r0 = svg_rank_column_center_x(origin_x, grid, 0, sx);
    let rn = svg_rank_column_center_x(origin_x, grid, beam_width.saturating_sub(1), sx);
    (r0 + rn) / 2
}

fn svg_emission_chip_centers_x(
    band_center: i32,
    k: usize, chip_w: u32, chip_h_gap: u32,
) -> Vec<i32> {
    if k == 0 { return Vec::new(); }
    let total = k as i32 * chip_w as i32 + (k.saturating_sub(1) as i32) * chip_h_gap as i32;
    let start = band_center - total / 2;
    (0..k).map(|j| start + chip_w as i32 / 2 + j as i32 * (chip_w as i32 + chip_h_gap as i32)).collect()
}

/// Prefix node fill and border by beam rank: strongest accent at rank 0, fading to neutral gray.
/// Shared desaturation endpoints so nodes and chips converge to the same neutral gray.
const DESAT_FILL_L: f64 = 0.92;
const DESAT_BRDR_L: f64 = 0.52;

fn svg_rank_node_colors(rank: usize, beam_width: usize) -> (RGBColor, RGBColor) {
    const FILL: (u8, u8, u8) = (220, 245, 228);
    const BRDR: (u8, u8, u8) = (32, 130, 72);
    let t = if beam_width <= 1 { 0.0 } else { rank as f64 / (beam_width - 1) as f64 };
    let sat_scale = 1.0 - t;
    let fill = desaturate_rgb8(FILL.0, FILL.1, FILL.2, sat_scale, DESAT_FILL_L);
    let brdr = desaturate_rgb8(BRDR.0, BRDR.1, BRDR.2, sat_scale, DESAT_BRDR_L);
    (RGBColor(fill.0, fill.1, fill.2), RGBColor(brdr.0, brdr.1, brdr.2))
}

/// Top-K chip fill and border by emission order (left = highest log-prob): lavender accent fading to gray.
fn svg_token_chip_colors(chip_index: usize, k: usize) -> (RGBColor, RGBColor) {
    const FILL: (u8, u8, u8) = (225, 222, 250);
    const BRDR: (u8, u8, u8) = (88, 78, 168);
    let t = if k <= 1 { 0.0 } else { chip_index as f64 / (k - 1) as f64 };
    let sat_scale = 1.0 - t;
    let fill = desaturate_rgb8(FILL.0, FILL.1, FILL.2, sat_scale, DESAT_FILL_L);
    let brdr = desaturate_rgb8(BRDR.0, BRDR.1, BRDR.2, sat_scale, DESAT_BRDR_L);
    (RGBColor(fill.0, fill.1, fill.2), RGBColor(brdr.0, brdr.1, brdr.2))
}

/// Top-K row for one beam step.
struct StepEmitArt<'a> {
    ids: &'a [i64],
    cx_list: Vec<i32>,
    chip_top: i32,
    chip_bot: i32,
}

/// CTC emission id for an edge: blank if prefix unchanged, else last token of child.
fn svg_edge_emission_token(parent: &BeamHypothesisSnapshot, child: &BeamHypothesisSnapshot, blank_id: usize) -> usize {
    if parent.sequence == child.sequence { blank_id } else { *child.sequence.last().expect("ctc extend adds one token") }
}

#[derive(Clone, Copy)]
struct SvgCubicF {
    p0: (f64, f64),
    p1: (f64, f64),
    p2: (f64, f64),
    p3: (f64, f64),
}

fn svg_path_d_chain_cubics(cubics: &[SvgCubicF]) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    if cubics.is_empty() { return s; }
    let c0 = &cubics[0];
    write!(&mut s, "M {:.2} {:.2} C {:.2} {:.2} {:.2} {:.2} {:.2} {:.2}",
        c0.p0.0, c0.p0.1, c0.p1.0, c0.p1.1, c0.p2.0, c0.p2.1, c0.p3.0, c0.p3.1).unwrap();
    for c in &cubics[1..] {
        write!(&mut s, " C {:.2} {:.2} {:.2} {:.2} {:.2} {:.2}",
            c.p1.0, c.p1.1, c.p2.0, c.p2.1, c.p3.0, c.p3.1).unwrap();
    }
    s
}

fn svg_path_d_parent_chip_child(a: SvgCubicF, b: SvgCubicF) -> String {
    use std::fmt::Write;
    let mut s = svg_path_d_chain_cubics(std::slice::from_ref(&a));
    write!(&mut s, " M {:.2} {:.2} C {:.2} {:.2} {:.2} {:.2} {:.2} {:.2}",
        b.p0.0, b.p0.1, b.p1.0, b.p1.1, b.p2.0, b.p2.1, b.p3.0, b.p3.1).unwrap();
    s
}

fn svg_path_stroke_element(d: &str, r: u8, g: u8, b: u8, stroke_opacity: f64, stroke_width: u32) -> String {
    format!(
        r#"<path d="{}" fill="none" stroke="rgb({},{},{})" stroke-opacity="{:.3}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"/>"#,
        d, r, g, b, stroke_opacity, stroke_width
    )
}

fn svg_emit_band_mask_defs(
    pw: i32, ph: i32, y_emission_centers: &[i32], chip_h: u32, chip_w: u32,
    has_emit: &[bool], chip_cx_per_step: &[Vec<i32>], sn: usize,
    ghost_token_strip_pad: i32, ghost_strip_feather_sigma: f64,
) -> String {
    use std::fmt::Write;
    let cw = chip_w as i32;
    let ch = chip_h as i32;
    let pad = ghost_token_strip_pad;
    let sigma = ghost_strip_feather_sigma;
    let mut s = String::new();
    write!(&mut s,
        r#"<filter id="prefix_beam_strip_blur" x="-50%" y="-50%" width="200%" height="200%" color-interpolation-filters="sRGB"><feGaussianBlur in="SourceGraphic" stdDeviation="{sigma:.2}"/></filter>
"#).unwrap();
    for si in 0..sn {
        if si + 1 >= sn || !has_emit.get(si).copied().unwrap_or(false) { continue; }
        let yc = y_emission_centers[si];
        let cx_list = chip_cx_per_step.get(si).map(Vec::as_slice).unwrap_or(&[]);
        if cx_list.is_empty() { continue; }

        let mut x_min = i32::MAX;
        let mut x_max = i32::MIN;
        for &cx in cx_list {
            x_min = x_min.min(cx - cw / 2);
            x_max = x_max.max(cx - cw / 2 + cw);
        }
        let xl = (x_min - pad).clamp(0, pw);
        let xr = (x_max + pad).clamp(0, pw);
        let strip_w = xr - xl;
        if strip_w <= 0 { continue; }

        let chip_top = yc - ch / 2;
        let chip_bot = yc + ch / 2;
        let yt = (chip_top - pad).clamp(0, ph);
        let yb = (chip_bot + pad).clamp(0, ph);
        let strip_h = yb - yt;
        if strip_h <= 0 { continue; }

        write!(&mut s,
            r#"<mask id="prefix_beam_emit_mask_solid_{si}" maskUnits="userSpaceOnUse" maskContentUnits="userSpaceOnUse" x="0" y="0" width="{pw}" height="{ph}"><rect x="0" y="0" width="{pw}" height="{ph}" fill="white"/><rect x="{xl}" y="{yt}" width="{strip_w}" height="{strip_h}" fill="black" filter="url(#prefix_beam_strip_blur)"/></mask>"#).unwrap();
        write!(&mut s,
            r#"<mask id="prefix_beam_emit_mask_dash_{si}" maskUnits="userSpaceOnUse" maskContentUnits="userSpaceOnUse" x="0" y="0" width="{pw}" height="{ph}"><rect x="0" y="0" width="{pw}" height="{ph}" fill="black"/><rect x="{xl}" y="{yt}" width="{strip_w}" height="{strip_h}" fill="white" filter="url(#prefix_beam_strip_blur)"/></mask>
"#).unwrap();
    }
    s
}

fn svg_path_stroke_emit_band_clipped(
    d: &str, r: u8, g: u8, b: u8, stroke_opacity: f64, stroke_width: u32,
    si: usize, ghost_dash_alpha_mul: f64, dash_on: u32, dash_off: u32,
) -> String {
    let dim = stroke_opacity * ghost_dash_alpha_mul;
    format!(
        r#"<path mask="url(#prefix_beam_emit_mask_solid_{si})" d="{d}" fill="none" stroke="rgb({r},{g},{b})" stroke-opacity="{a0:.3}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>
<path mask="url(#prefix_beam_emit_mask_dash_{si})" d="{d}" fill="none" stroke="rgb({r},{g},{b})" stroke-opacity="{a1:.3}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="{d_on} {d_off}"/>
"#,
        si = si, d = d, r = r, g = g, b = b, a0 = stroke_opacity, a1 = dim,
        sw = stroke_width, d_on = dash_on, d_off = dash_off,
    )
}

fn inject_svg_before_close_svg(svg: &mut String, fragment: &str) {
    let Some(j) = svg.rfind("</svg>") else {
        svg.push_str(fragment);
        return;
    };
    svg.insert_str(j, fragment);
}

fn svg_patch_rect_corner_radius(svg: &mut String, skip_rects: usize, patch_count: usize, rx: u32) {
    if patch_count == 0 { return; }
    let attr = format!(r#" rx="{}" ry="{}""#, rx, rx);
    let mut rect_index = 0usize;
    let mut patched = 0usize;
    let mut cursor = 0usize;
    while patched < patch_count && cursor < svg.len() {
        let Some(rel) = svg[cursor..].find("<rect ") else { break; };
        let open = cursor + rel;
        let Some(slash_rel) = svg[open..].find("/>") else { break; };
        let slash_abs = open + slash_rel;
        rect_index += 1;
        if rect_index > skip_rects {
            svg.insert_str(slash_abs, &attr);
            patched += 1;
            cursor = slash_abs + attr.len() + 2;
        } else {
            cursor = slash_abs + 2;
        }
    }
}

fn cubic_vertical_end_tangents_f(p0: (f64, f64), p3: (f64, f64)) -> SvgCubicF {
    let dy = p3.1 - p0.1;
    SvgCubicF { p0, p1: (p0.0, p0.1 + dy / 3.0), p2: (p3.0, p3.1 - dy / 3.0), p3 }
}

fn cubic_down_to_chip_top_f(p_bot: (i32, i32), chip_cx: i32, chip_top_y: i32, _lane: i32) -> SvgCubicF {
    cubic_vertical_end_tangents_f((p_bot.0 as f64, p_bot.1 as f64), (chip_cx as f64, chip_top_y as f64))
}

fn cubic_chip_bottom_to_top_f(chip_cx: i32, chip_bot_y: i32, c_top: (i32, i32), _lane: i32) -> SvgCubicF {
    cubic_vertical_end_tangents_f((chip_cx as f64, chip_bot_y as f64), (c_top.0 as f64, c_top.1 as f64))
}

fn cubic_vertical_span_f(p_bot: (i32, i32), c_top: (i32, i32), _lane: i32) -> SvgCubicF {
    cubic_vertical_end_tangents_f((p_bot.0 as f64, p_bot.1 as f64), (c_top.0 as f64, c_top.1 as f64))
}

fn write_prefix_beam_svg(
    trace: &PrefixBeamTrace,
    path: &std::path::Path,
    token_map: Option<&TokenMap>,
    theme: &PrefixBeamSvgTheme,
) -> Result<(), String> {
    use plotters::element::{Rectangle, Text};
    use plotters::prelude::*;
    use plotters::style::text_anchor::{HPos, Pos, VPos};
    use plotters::style::Color;

    let layout = GraphLayout::from_trace(trace, token_map);
    let sn = layout.steps_n;
    let w = layout.beam_width;
    if sn == 0 { return Err("empty beam trace".to_string()); }

    let grid = beam_rank_char_layout(&layout);
    let char_sx = theme.char_sx;
    let node_row_h = theme.node_row_h;
    let node_ry = theme.node_ry;
    let label_fs = theme.node_label_fs;
    let h_pad = theme.node_h_pad;
    let min_half_w = theme.node_min_half_w;
    let chip_w = theme.chip_w;
    let chip_h = theme.chip_h;
    let chip_h_gap = theme.chip_h_gap;

    let emission_h = theme.cand_margin_v + chip_h + theme.cand_margin_v;
    let top_stack_min = theme.axes_to_heatmap_margin
        .saturating_add(theme.x_axis_tick_band_h)
        .saturating_add(theme.title_h)
        .saturating_add(theme.title_cluster_lift_px);
    let heat_y0 = theme.margin_top.max(top_stack_min);

    let mut node_half_w: Vec<Vec<u32>> = vec![vec![0; w]; sn];
    for si in 0..sn {
        for r in 0..w {
            if layout.labels[si][r].is_empty() { continue; }
            node_half_w[si][r] = svg_node_half_width(&layout.labels[si][r], layout.scores[si][r], label_fs, h_pad, min_half_w);
        }
    }

    let mut y_node_centers: Vec<i32> = Vec::with_capacity(sn);
    let mut y_emission_centers: Vec<i32> = Vec::with_capacity(sn.saturating_sub(1));
    let mut y_acc = heat_y0 as i32;
    for si in 0..sn {
        y_node_centers.push(y_acc + node_row_h as i32 / 2);
        y_acc += node_row_h as i32;
        if si + 1 < sn {
            y_emission_centers.push(y_acc + emission_h as i32 / 2);
            y_acc += emission_h as i32;
        }
    }

    let timestep_col_px = (grid.margin_w as f64 * char_sx).round() as i32;
    let origin_x = theme.margin_left as i32 - timestep_col_px;
    let beam_left_x = svg_char_col_to_x(origin_x, grid.col_x[0], char_sx);
    let beam_right_x = if w > 0 {
        svg_char_col_to_x(origin_x, grid.col_x[w - 1] + grid.col_w[w - 1], char_sx)
    } else { beam_left_x };
    let x_dag_center = svg_beam_center_x(origin_x, &grid, w, char_sx);
    let pw = (beam_right_x + theme.margin_right as i32).max(1) as u32;
    let ph = y_acc.max(1) as u32 + theme.margin_bottom;

    let col_half_min = theme.col_half_min;
    let col_half_px = |r: usize, si_step: usize| -> i32 {
        let want = node_half_w[si_step][r].max(1) as i32;
        let cap = ((grid.col_w[r] as f64 * char_sx) / 2.0).floor() as i32 - 1;
        want.min(cap.max(col_half_min))
    };

    let axes_m = theme.axes_to_heatmap_margin as i32;
    let r0_cx = svg_rank_column_center_x(origin_x, &grid, 0, char_sx);
    let rank0_prefix_left_edge = (0..sn).map(|si| r0_cx - col_half_px(0, si)).min().unwrap_or(beam_left_x);
    let time_label_x = rank0_prefix_left_edge - axes_m;

    let steps_snap = trace.steps();
    let palette_seed: u64 = trace.decoded.iter().enumerate().fold(0x517CC1B727220A95_u64, |acc, (i, &id)| {
        acc.wrapping_add((id as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(i as u64))
    });
    let lineage_palette = build_lineage_palette(w + 3, theme.edge_rgb_best, theme.lineage_min_rgb_dist_from_best_sq, Some(palette_seed));
    let lineage_node_rgb = compute_lineage_rgb_per_node(
        steps_snap, &layout.edges, w, &lineage_palette,
        theme.edge_rgb_normal, theme.edge_rgb_best, theme.lineage_min_rgb_dist_from_best_sq,
    );
    let blank_id = trace.blank_id;
    let has_emit: Vec<bool> = (0..sn).map(|si| {
        si + 1 < sn && trace.top_k_emissions_per_t.get(si).is_some_and(|v| !v.is_empty())
    }).collect();

    let mut chip_cx_per_step: Vec<Vec<i32>> = vec![Vec::new(); sn];
    for si in 0..sn {
        if !has_emit[si] { continue; }
        let Some(ids) = trace.top_k_emissions_per_t.get(si).map(|v| v.as_slice()) else { continue; };
        if ids.is_empty() { continue; }
        chip_cx_per_step[si] = svg_emission_chip_centers_x(x_dag_center, ids.len(), chip_w, chip_h_gap);
    }

    let mut svg_buf = String::new();
    let emit_band_defs = svg_emit_band_mask_defs(
        pw as i32, ph as i32, &y_emission_centers, chip_h, chip_w,
        &has_emit, &chip_cx_per_step, sn, theme.ghost_token_strip_pad, theme.ghost_strip_feather_sigma,
    );
    let mut edge_paths = format!("<defs>\n{}</defs>\n<g shape-rendering=\"geometricPrecision\">\n", emit_band_defs);

    let root = SVGBackend::with_string(&mut svg_buf, (pw, ph)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| format!("plotters fill: {e}"))?;

    let text_rgb = RGBColor(theme.text_rgb.0, theme.text_rgb.1, theme.text_rgb.2);
    let style_title = TextStyle::from((SVG_UI_FONT, theme.title_fs).into_font().color(&text_rgb))
        .pos(Pos::new(HPos::Center, VPos::Center));
    let subtitle_fs = (theme.title_fs * 0.75).clamp(13.0, 18.0);
    let style_sub = TextStyle::from((SVG_UI_FONT, subtitle_fs).into_font().color(&text_rgb))
        .pos(Pos::new(HPos::Center, VPos::Center));
    let style_node_label = TextStyle::from((SVG_UI_FONT, theme.node_label_fs).into_font().color(&text_rgb))
        .pos(Pos::new(HPos::Center, VPos::Center));
    let style_node_score = TextStyle::from((SVG_UI_FONT, label_fs).into_font()
        .color(&RGBColor(theme.node_score_rgb.0, theme.node_score_rgb.1, theme.node_score_rgb.2)))
        .pos(Pos::new(HPos::Center, VPos::Center));
    let style_timestep = TextStyle::from((SVG_UI_FONT, theme.time_label_fs).into_font().color(&text_rgb))
        .pos(Pos::new(HPos::Right, VPos::Center));
    let style_rank_lab = TextStyle::from((SVG_UI_FONT, theme.rank_header_fs).into_font().color(&text_rgb))
        .pos(Pos::new(HPos::Center, VPos::Top));
    let style_chip = TextStyle::from((SVG_UI_FONT, label_fs).into_font().color(&text_rgb))
        .pos(Pos::new(HPos::Center, VPos::Center));

    let title_lift = theme.title_cluster_lift_px as i32;
    let y_title_1 = (heat_y0 as i32 - axes_m - theme.x_axis_tick_band_h as i32 - (theme.title_h / 2) as i32 - title_lift).max(1);
    let y_title_2 = (heat_y0 as i32 - axes_m - (theme.x_axis_tick_band_h / 2) as i32 - title_lift).max(1);

    root.draw(&Text::new("CTC Decode: Prefix Beam DAG", (x_dag_center, y_title_1), &style_title).into_dyn())
        .map_err(|e| format!("plotters title: {e}"))?;
    let dec_s = decoded_display(trace.decoded_ids(), token_map);
    root.draw(&Text::new(format!("Decoded Sequence: \"{dec_s}\""), (x_dag_center, y_title_2), &style_sub).into_dyn())
        .map_err(|e| format!("plotters subtitle: {e}"))?;

    let dag_top_y = heat_y0 as i32 + node_row_h as i32 / 2 - node_ry as i32;
    let rank_hdr_est_h = (theme.rank_header_fs * 1.2).ceil() as i32;
    let y_rank_hdr = (dag_top_y - axes_m - rank_hdr_est_h).max(1);

    for r in 0..w {
        let rcx = svg_rank_column_center_x(origin_x, &grid, r, char_sx);
        root.draw(&Text::new(format!("r = {r}"), (rcx, y_rank_hdr), &style_rank_lab).into_dyn())
            .map_err(|e| format!("plotters rank hdr: {e}"))?;
    }
    for si in 0..sn {
        let lab = if si == 0 { "init".to_string() } else { format!("t = {}", si - 1) };
        root.draw(&Text::new(lab, (time_label_x, y_node_centers[si]), &style_timestep).into_dyn())
            .map_err(|e| format!("plotters time label: {e}"))?;
    }

    let v_sz = trace.vocab_size;
    let k_legend = w.min(v_sz.saturating_sub(1)).max(1);
    let rank_legend_n = w.max(1);

    // legend: centered between subtitle and rank headers
    // left column = swatch gradients, right column = edge samples
    {
        let sw = theme.legend_swatch_size;
        let row_gap = 8_i32;
        let col_gap = 24_i32;
        let swatch_label_gap = 4_i32;
        let edge_sample_w = 24_i32;
        let style_legend_r = TextStyle::from((SVG_UI_FONT, theme.legend_fs).into_font().color(&text_rgb))
            .pos(Pos::new(HPos::Right, VPos::Center));

        let legend_top = y_title_2 + (subtitle_fs * 0.6).ceil() as i32 + 4;
        let legend_bot = y_rank_hdr;
        let legend_content_h = 2 * sw as i32 + row_gap;
        let avail = legend_bot - legend_top - legend_content_h;
        let row1_y = legend_top + (avail / 2).max(2);
        let row2_y = row1_y + sw as i32 + row_gap;
        let row1_cy = row1_y + sw as i32 / 2;
        let row2_cy = row2_y + sw as i32 / 2;

        let strip_w_px = LEGEND_STRIP_CELLS * sw as i32;
        let max_swatch_strip_w = strip_w_px;

        let chip_label = "Top-K tokens";
        let rank_label = "Top-W prefixes";
        let fan_label = "Token candidate";
        let decode_label = "Decode path";
        let swatch_label_max_w = svg_sans_text_width_px(
            chip_label.len().max(rank_label.len()), theme.legend_fs) as i32;
        let edge_label_max_w = svg_sans_text_width_px(
            fan_label.len().max(decode_label.len()), theme.legend_fs) as i32;

        let left_col_w = swatch_label_max_w + swatch_label_gap + max_swatch_strip_w;
        let right_col_w = edge_label_max_w + swatch_label_gap + edge_sample_w;
        let total_w = left_col_w + col_gap + right_col_w;

        let legend_left = x_dag_center - total_w / 2;
        let left_col_right = legend_left + left_col_w;
        let right_col_right = left_col_right + col_gap + right_col_w;

        // left column: swatch gradients
        // row 1: top-W prefixes
        let rank_strip_x0 = left_col_right - strip_w_px;
        let rank_label_x = rank_strip_x0 - swatch_label_gap;
        for j in 0..rank_legend_n {
            let x0 = rank_strip_x0 + (j as i32 * strip_w_px) / rank_legend_n as i32;
            let x1 = rank_strip_x0 + ((j as i32 + 1) * strip_w_px) / rank_legend_n as i32;
            let (fill, border) = svg_rank_node_colors(j, w);
            root.draw(&Rectangle::new([(x0, row1_y), (x1, row1_y + sw as i32)], fill.filled()))
                .map_err(|e| format!("plotters legend rank fill: {e}"))?;
            root.draw(&Rectangle::new([(x0, row1_y), (x1, row1_y + sw as i32)], ShapeStyle::from(&border).stroke_width(1)))
                .map_err(|e| format!("plotters legend rank border: {e}"))?;
        }
        root.draw(&Text::new(rank_label, (rank_label_x, row1_cy), &style_legend_r).into_dyn())
            .map_err(|e| format!("plotters legend rank label: {e}"))?;

        // row 2: top-K tokens
        let chip_strip_x0 = left_col_right - strip_w_px;
        let chip_label_x = chip_strip_x0 - swatch_label_gap;
        for j in 0..k_legend {
            let x0 = chip_strip_x0 + (j as i32 * strip_w_px) / k_legend as i32;
            let x1 = chip_strip_x0 + ((j as i32 + 1) * strip_w_px) / k_legend as i32;
            let (fill, border) = svg_token_chip_colors(j, k_legend);
            root.draw(&Rectangle::new([(x0, row2_y), (x1, row2_y + sw as i32)], fill.filled()))
                .map_err(|e| format!("plotters legend chip fill: {e}"))?;
            root.draw(&Rectangle::new([(x0, row2_y), (x1, row2_y + sw as i32)], ShapeStyle::from(&border).stroke_width(1)))
                .map_err(|e| format!("plotters legend chip border: {e}"))?;
        }
        root.draw(&Text::new(chip_label, (chip_label_x, row2_cy), &style_legend_r).into_dyn())
            .map_err(|e| format!("plotters legend chip label: {e}"))?;

        // right column: edge samples
        // row 1: decode path
        let decode_line_x1 = right_col_right;
        let decode_line_x0 = decode_line_x1 - edge_sample_w;
        let decode_label_x = decode_line_x0 - swatch_label_gap;
        let (br, bg, bb) = theme.edge_rgb_best;
        let best_color = RGBColor(br, bg, bb);
        root.draw(&plotters::element::PathElement::new(
            vec![(decode_line_x0, row1_cy), (decode_line_x1, row1_cy)],
            ShapeStyle::from(&best_color).stroke_width(theme.edge_stroke_w_best),
        )).map_err(|e| format!("plotters legend decode line: {e}"))?;
        root.draw(&Text::new(decode_label, (decode_label_x, row1_cy), &style_legend_r).into_dyn())
            .map_err(|e| format!("plotters legend decode label: {e}"))?;

        // row 2: token candidate
        let fan_line_x1 = right_col_right;
        let fan_line_x0 = fan_line_x1 - edge_sample_w;
        let fan_label_x = fan_line_x0 - swatch_label_gap;
        let (fr, fg, fb) = theme.fan_edge_rgb;
        let fan_color = RGBColor(fr, fg, fb).mix(theme.fan_edge_alpha.max(0.35));
        root.draw(&plotters::element::PathElement::new(
            vec![(fan_line_x0, row2_cy), (fan_line_x1, row2_cy)],
            fan_color.stroke_width(theme.fan_edge_stroke_w.max(1)),
        )).map_err(|e| format!("plotters legend fan line: {e}"))?;
        root.draw(&Text::new(fan_label, (fan_label_x, row2_cy), &style_legend_r).into_dyn())
            .map_err(|e| format!("plotters legend fan label: {e}"))?;
    }

    let edge_color_best = RGBColor(theme.edge_rgb_best.0, theme.edge_rgb_best.1, theme.edge_rgb_best.2);

    // fan edges (faint curves from each parent to every chip)
    for (si, step_edges) in layout.edges.iter().enumerate() {
        if si + 1 >= sn || !has_emit[si] { continue; }
        let Some(ids) = trace.top_k_emissions_per_t.get(si).map(|v| v.as_slice()) else { continue; };
        if ids.is_empty() { continue; }
        let cx_list = svg_emission_chip_centers_x(x_dag_center, ids.len(), chip_w, chip_h_gap);
        let chip_top_y = y_emission_centers[si] - chip_h as i32 / 2;
        let mut has_out = vec![false; w];
        for e in step_edges { has_out[e.parent_rank] = true; }

        for pr in 0..w {
            if !has_out[pr] || layout.labels[si][pr].is_empty() { continue; }
            let pr_cx = svg_rank_column_center_x(origin_x, &grid, pr, char_sx);
            let p_bot = (pr_cx, y_node_centers[si] + node_ry as i32);
            for (j, &ccx) in cx_list.iter().enumerate() {
                let lane = (pr as i32 * 3 + j as i32) % 5 - 2;
                let c = cubic_down_to_chip_top_f(p_bot, ccx, chip_top_y, lane);
                let d = svg_path_d_chain_cubics(std::slice::from_ref(&c));
                edge_paths.push_str(&svg_path_stroke_element(
                    &d, theme.fan_edge_rgb.0, theme.fan_edge_rgb.1, theme.fan_edge_rgb.2,
                    theme.fan_edge_alpha, theme.fan_edge_stroke_w,
                ));
                edge_paths.push('\n');
            }
        }
    }

    // real beam edges
    for (si, step_edges) in layout.edges.iter().enumerate() {
        let emit_art: Option<StepEmitArt<'_>> = if has_emit[si] {
            trace.top_k_emissions_per_t.get(si).map(|v| v.as_slice()).and_then(|ids| {
                if ids.is_empty() { return None; }
                let y_em = y_emission_centers[si];
                Some(StepEmitArt {
                    ids,
                    cx_list: svg_emission_chip_centers_x(x_dag_center, ids.len(), chip_w, chip_h_gap),
                    chip_top: y_em - chip_h as i32 / 2,
                    chip_bot: y_em + chip_h as i32 / 2,
                })
            })
        } else { None };

        for edge in step_edges {
            let on_best = layout.is_edge_on_best_path(si, edge);
            let color = if on_best {
                edge_color_best
            } else {
                let (r, g, b) = lineage_node_rgb
                    .get(si + 1)
                    .and_then(|row| row.get(edge.child_rank).copied().flatten())
                    .unwrap_or(theme.edge_rgb_normal);
                RGBColor(r, g, b)
            };
            let sw = if on_best { theme.edge_stroke_w_best } else { theme.edge_stroke_w_normal };
            let alpha = if on_best {
                theme.edge_alpha_best
            } else {
                (theme.edge_alpha_normal * theme.lineage_edge_alpha_mul).clamp(0.0, 1.0)
            };

            let p_cx = svg_rank_column_center_x(origin_x, &grid, edge.parent_rank, char_sx);
            let c_cx = svg_rank_column_center_x(origin_x, &grid, edge.child_rank, char_sx);
            let p_bot = (p_cx, y_node_centers[si] + node_ry as i32);
            let c_top = (c_cx, y_node_centers[si + 1] - node_ry as i32);

            let emit = svg_edge_emission_token(&steps_snap[si][edge.parent_rank], &steps_snap[si + 1][edge.child_rank], blank_id);
            let lane = (edge.parent_rank as i32 * 5 + edge.child_rank as i32) % 7 - 3;

            let d = match emit_art.as_ref() {
                Some(row) => {
                    let span_cubic = |pb: (i32, i32), ct: (i32, i32), ln: i32| {
                        let c = cubic_vertical_span_f(pb, ct, ln);
                        svg_path_d_chain_cubics(std::slice::from_ref(&c))
                    };
                    if let Some(ki) = row.ids.iter().position(|&x| x as usize == emit) {
                        let ccx = row.cx_list[ki];
                        let lo = p_cx.min(ccx).min(c_cx);
                        let hi = p_cx.max(ccx).max(c_cx);
                        if hi - lo <= 1 {
                            let x = (p_cx + ccx + c_cx) as f64 / 3.0;
                            format!("M {x:.2} {:.2} L {x:.2} {:.2}", p_bot.1 as f64, c_top.1 as f64)
                        } else {
                            let a = cubic_down_to_chip_top_f(p_bot, ccx, row.chip_top, lane);
                            let b = cubic_chip_bottom_to_top_f(ccx, row.chip_bot, c_top, lane);
                            svg_path_d_parent_chip_child(a, b)
                        }
                    } else {
                        span_cubic(p_bot, c_top, lane)
                    }
                }
                None => {
                    let c = cubic_vertical_span_f(p_bot, c_top, lane);
                    svg_path_d_chain_cubics(std::slice::from_ref(&c))
                }
            };

            let (r, g, b) = color.rgb();
            let pins_chip = emit_art.as_ref()
                .and_then(|row| row.ids.iter().position(|&x| x as usize == emit))
                .is_some();
            let ghost_emit_band = has_emit[si] && !pins_chip;
            let paths = if ghost_emit_band {
                svg_path_stroke_emit_band_clipped(
                    &d, r, g, b, alpha, sw, si,
                    theme.ghost_dash_alpha_mul, theme.ghost_stroke_dash.0, theme.ghost_stroke_dash.1,
                )
            } else {
                svg_path_stroke_element(&d, r, g, b, alpha, sw)
            };
            edge_paths.push_str(&paths);
            edge_paths.push('\n');
        }
    }

    edge_paths.push_str("</g>\n");

    // node boxes
    for si in 0..sn {
        for r in 0..w {
            if layout.labels[si][r].is_empty() { continue; }
            let cx = svg_rank_column_center_x(origin_x, &grid, r, char_sx);
            let cy = y_node_centers[si];
            let on_best = layout.is_on_best_path(si, r);
            let nhw = col_half_px(r, si);
            let (fill, border) = svg_rank_node_colors(r, w);
            let x0 = cx - nhw;
            let y0 = cy - node_ry as i32;
            let x1 = cx + nhw;
            let y1 = cy + node_ry as i32;
            root.draw(&Rectangle::new([(x0, y0), (x1, y1)], fill.filled()))
                .map_err(|e| format!("plotters node fill: {e}"))?;
            root.draw(&Rectangle::new([(x0, y0), (x1, y1)],
                ShapeStyle::from(&border).stroke_width(if on_best { theme.node_border_stroke_w_best } else { theme.node_border_stroke_w_normal })))
                .map_err(|e| format!("plotters node border: {e}"))?;
            let label = layout.labels[si][r].as_str();
            root.draw(&Text::new(format!("\"{label}\""), (cx, cy + theme.node_label_dy), &style_node_label).into_dyn())
                .map_err(|e| format!("plotters node label: {e}"))?;
            root.draw(&Text::new(format!("{:.2}", layout.scores[si][r]), (cx, cy + theme.node_score_dy), &style_node_score).into_dyn())
                .map_err(|e| format!("plotters node score: {e}"))?;
        }
    }

    // token chips
    for si in 0..sn {
        if si + 1 >= sn || !has_emit[si] { continue; }
        let Some(ids) = trace.top_k_emissions_per_t.get(si) else { continue; };
        if ids.is_empty() { continue; }
        let k_chips = ids.len();
        let cx_list = svg_emission_chip_centers_x(x_dag_center, k_chips, chip_w, chip_h_gap);
        let y_em = y_emission_centers[si];
        for (j, tid) in ids.iter().enumerate() {
            let (chip_fill, chip_border) = svg_token_chip_colors(j, k_chips);
            let cci = cx_list[j];
            let left = cci - chip_w as i32 / 2;
            let c1 = left + chip_w as i32;
            let d0 = y_em - chip_h as i32 / 2;
            let d1 = y_em + chip_h as i32 / 2;
            root.draw(&Rectangle::new([(left, d0), (c1, d1)], chip_fill.filled()))
                .map_err(|e| format!("plotters chip fill: {e}"))?;
            root.draw(&Rectangle::new([(left, d0), (c1, d1)], ShapeStyle::from(&chip_border).stroke_width(theme.chip_border_stroke_w)))
                .map_err(|e| format!("plotters chip border: {e}"))?;
            let mut sym = token_id_display(*tid as usize, blank_id, token_map);
            if sym.chars().count() > theme.chip_symbol_max_chars {
                sym = sym.chars().take(theme.chip_symbol_max_chars).collect();
            }
            root.draw(&Text::new(format!("\"{sym}\""), (cci, y_em), &style_chip).into_dyn())
                .map_err(|e| format!("plotters chip text: {e}"))?;
        }
    }

    root.present().map_err(|e| format!("plotters present: {e}"))?;
    std::mem::drop(root);

    let legend_rects = 2 * (rank_legend_n + k_legend);
    let prefix_nodes_drawn: usize = (0..sn)
        .flat_map(|si| (0..w).map(move |r| (si, r)))
        .filter(|&(si, r)| !layout.labels[si][r].is_empty())
        .count();
    let chips_drawn: usize = (0..sn)
        .filter(|&si| si + 1 < sn && has_emit[si])
        .filter_map(|si| trace.top_k_emissions_per_t.get(si))
        .filter(|ids| !ids.is_empty())
        .map(|ids| ids.len())
        .sum();
    let node_rx = theme.node_corner_rx;
    let chip_rx = theme.chip_corner_rx;
    let legend_rx = node_rx.min(theme.legend_swatch_size / 3);
    svg_patch_rect_corner_radius(&mut svg_buf, 1, legend_rects, legend_rx);
    let prefix_rects = 2 * prefix_nodes_drawn;
    svg_patch_rect_corner_radius(&mut svg_buf, 1 + legend_rects, prefix_rects, node_rx);
    svg_patch_rect_corner_radius(&mut svg_buf, 1 + legend_rects + prefix_rects, 2 * chips_drawn, chip_rx);

    inject_svg_before_close_svg(&mut svg_buf, &edge_paths);
    std::fs::write(path, svg_buf.as_bytes()).map_err(|e| format!("write svg: {e}"))?;
    Ok(())
}



// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

/// Capture and render prefix beam traces.
pub struct CtcPrefixBeamViz;

impl CtcPrefixBeamViz {
    /// Requires [`crate::ctc::ctc_decode::CtcDecodeType::BeamSearch`] and batch **`N == 1`**.
    pub fn beam_trace_sample0<B: Backend>(
        decoder: &CtcDecoder,
        logits: Tensor<B, 3>,
    ) -> Result<PrefixBeamTrace, &'static str> {
        if decoder.search_type != CtcDecodeType::BeamSearch {
            return Err("beam_trace_sample0 requires CtcDecodeType::BeamSearch");
        }
        let [n, _t, v] = logits.dims();
        if n != 1 { return Err("beam_trace_sample0 requires batch size N == 1"); }
        assert!(decoder.blank_id < v, "blank ID out of bounds");
        assert!((1..=15).contains(&decoder.beam_width), "beam width must be in [1, 15]");
        if decoder.lm.is_some() {
            assert!((0.2..=3.0).contains(&decoder.lm_alpha), "LM alpha out of range");
            assert!((1.5..=5.0).contains(&decoder.lm_beta), "LM beta out of range");
        }
        Ok(trace_beam_sample0_inner(decoder, logits))
    }

    /// Vertical ASCII DAG: time flows downward, beam rank horizontal, box-drawing edge routing.
    pub fn render_ascii(trace: &PrefixBeamTrace, token_map: Option<&TokenMap>) -> String {
        render_prefix_beam_ascii(trace, token_map)
    }

    /// SVG DAG: time downward, beam rank left→right (same grid as ASCII), Bézier edges.
    ///
    /// Layout and colors follow [`PrefixBeamSvgTheme::default`]; use [`Self::write_svg_with_theme`] to override.
    pub fn write_svg(trace: &PrefixBeamTrace, path: &std::path::Path, token_map: Option<&TokenMap>) -> Result<(), String> {
        write_prefix_beam_svg(trace, path, token_map, &PrefixBeamSvgTheme::default())
    }

    /// Same as [`Self::write_svg`], using a custom [`PrefixBeamSvgTheme`].
    pub fn write_svg_with_theme(
        trace: &PrefixBeamTrace,
        path: &std::path::Path,
        token_map: Option<&TokenMap>,
        theme: &PrefixBeamSvgTheme,
    ) -> Result<(), String> {
        write_prefix_beam_svg(trace, path, token_map, theme)
    }
}



// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ctc::ctc_decode::{CtcDecodeType, CtcDecoderConfig};
    use crate::ctc::ctc_loss::{CtcLoss, CtcLossConfig};
    use crate::vocab::{TokenMap, BLANK_ID, VOCAB, VOCAB_SIZE};
    use burn::{
        backend::ndarray::NdArray,
        nn::loss::Reduction,
        prelude::Int,
        tensor::{backend::Backend, Tensor, TensorData},
    };
    use rand::{rngs::StdRng, Rng, SeedableRng};

    type B = NdArray<f32>;

    const FIXTURE_SEQS: &[&str] = &["cat", "gobbledygook", "hippopotomonstrosesquippedaliophobia"];
    const FIXTURE_LOGITS_SEED: u64 = 12345u64;
    const FIXTURE_INTENDED_LOGIT_BIAS: f32 = 5.5;
    const FIXTURE_LOGIT_NOISE: f32 = 0.42;
    const FIXTURE_T_BUFFER: usize = 20;

    fn fixture_timesteps(target_token_len: usize) -> usize {
        let interleaved_len = target_token_len.saturating_mul(2).saturating_add(1);
        let buf = FIXTURE_T_BUFFER;
        if interleaved_len <= buf { buf }
        else {
            let mut t = buf;
            while t < interleaved_len { t += buf; }
            t
        }
    }

    fn outputs_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("outputs")
    }

    fn synthetic_logits_sweep_interleaved<Bk: Backend>(
        loss: &CtcLoss, device: &Bk::Device, targets: Tensor<Bk, 2, Int>,
        t: usize, vocab_size: usize, seed: u64,
    ) -> Tensor<Bk, 3> {
        let targets_intr = loss.interleave_targets_with_blanks(targets.clone(), device);
        let intr_ids: Vec<i32> = targets_intr.clone().into_data().convert::<i32>().into_vec::<i32>().unwrap();
        let intr_len = intr_ids.len();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut buf = vec![0f32; t * vocab_size];
        for t_idx in 0..t {
            let row = t_idx * vocab_size;
            for v in 0..vocab_size { buf[row + v] = rng.random_range(-FIXTURE_LOGIT_NOISE..FIXTURE_LOGIT_NOISE); }
            let s = ((t_idx * intr_len) / t).min(intr_len - 1);
            let sym = intr_ids[s] as usize;
            buf[row + sym] += FIXTURE_INTENDED_LOGIT_BIAS;
        }
        Tensor::<Bk, 3>::from_data(TensorData::new(buf, vec![1, t, vocab_size]), device)
    }

    fn logits_and_decoder_for_word(word: &str, beam_width: usize, seed: u64) -> (Tensor<B, 3>, CtcDecoder) {
        let device = Default::default();
        let token_map = TokenMap::new(VOCAB);
        let chars: Vec<char> = word.chars().collect();
        let ids_usize = token_map.chars_to_ids(&chars).unwrap_or_else(|| {
            panic!("fixture word {word:?} must be encodable in VOCAB={VOCAB:?}");
        });
        let l_max = ids_usize.len();
        let t = fixture_timesteps(l_max);
        let loss = CtcLossConfig::new().with_blank_id(BLANK_ID).with_reduction(Reduction::Mean).init();
        let ids_i64: Vec<i64> = ids_usize.iter().map(|&x| x as i64).collect();
        let targets = Tensor::<B, 2, Int>::from_data(TensorData::new(ids_i64, vec![1, l_max]), &device);
        let logits = synthetic_logits_sweep_interleaved(&loss, &device, targets, t, VOCAB_SIZE, seed);
        let decoder = CtcDecoderConfig::new()
            .with_blank_id(BLANK_ID)
            .with_search_type(CtcDecodeType::BeamSearch)
            .with_beam_width(beam_width)
            .init();
        (logits, decoder)
    }

    #[test]
    fn edge_derivation_sanity() {
        for (i, word) in FIXTURE_SEQS.iter().enumerate() {
            let (logits, decoder) = logits_and_decoder_for_word(word, 5, FIXTURE_LOGITS_SEED ^ i as u64);
            let trace = CtcPrefixBeamViz::beam_trace_sample0(&decoder, logits).expect("trace");
            let all_edges = derive_all_edges(trace.steps());
            assert_eq!(all_edges.len(), trace.timesteps, "word={word:?}: one edge set per transition");
            for (si, edges) in all_edges.iter().enumerate() {
                assert!(!edges.is_empty(), "word={word:?}: edges at transition {si} must be non-empty");
                for e in edges {
                    assert!(e.parent_rank < trace.steps()[si].len(), "parent_rank out of bounds");
                    assert!(e.child_rank < trace.steps()[si + 1].len(), "child_rank out of bounds");
                }
            }
        }
    }

    #[test]
    fn lineage_merge_picks_distinct_color_from_two_parent_lineages() {
        let edge_best = (32u8, 130u8, 72u8);
        let fallback = (160u8, 160u8, 180u8);
        let min_d = 35 * 35;
        let palette = build_lineage_palette(14, edge_best, min_d, None);
        assert!(palette.len() >= 3);

        let snap = |seq: Vec<usize>| BeamHypothesisSnapshot {
            sequence: seq, log_prob_blank: 0.0, log_prob_non_blank: 0.0, combined_log_prob: -1.0,
        };
        // rue merge: two parents both with seq [7] (different ranks → different palette colors)
        // extend to child [7,9]. Neither parent has the child's sequence, so stay-inherit
        // does not fire and the merge branch picks a fresh color
        let step0 = vec![snap(vec![7]), snap(vec![7])];
        let step1 = vec![snap(vec![7, 9])];
        let steps = vec![step0, step1];
        let edges = derive_all_edges(&steps);
        let lineage = compute_lineage_rgb_per_node(&steps, &edges, 2, &palette, fallback, edge_best, min_d);

        let c0 = lineage[0][0].expect("rank0 color");
        let c1 = lineage[0][1].expect("rank1 tcolor");
        assert_ne!(c0, c1);
        let merged = lineage[1][0].expect("merged child");
        assert_ne!(merged, edge_best);
        assert_ne!(merged, c0);
        assert_ne!(merged, c1);
    }

    #[test]
    fn lineage_fanout_assigns_distinct_colors_to_child_nodes() {
        let edge_best = (32u8, 130u8, 72u8);
        let fallback = (160u8, 160u8, 180u8);
        let min_d = 35 * 35;
        let palette = build_lineage_palette(14, edge_best, min_d, None);

        let snap = |seq: Vec<usize>| BeamHypothesisSnapshot {
            sequence: seq, log_prob_blank: 0.0, log_prob_non_blank: 0.0, combined_log_prob: -1.0,
        };
        // parent [1] fans out: child rank 0 stays [1], child rank 1 extends to [1,2]
        let step0 = vec![snap(vec![1])];
        let step1 = vec![snap(vec![1]), snap(vec![1, 2])];
        let steps = vec![step0, step1];
        let edges = derive_all_edges(&steps);
        let lineage = compute_lineage_rgb_per_node(&steps, &edges, 3, &palette, fallback, edge_best, min_d);

        let parent_col = lineage[0][0].expect("parent rank0");
        let c_stay = lineage[1][0].expect("child stay");
        let c_ext = lineage[1][1].expect("child extend");
        // stay child inherits parent color; extend child gets a distinct fan-out color
        assert_eq!(c_stay, parent_col, "stay child must inherit parent color");
        assert_ne!(c_ext, parent_col, "extend child must differ from parent");
        assert_ne!(c_ext, c_stay, "fan-out children must keep separate lineages");
    }

    #[test]
    fn lineage_stay_preserves_color() {
        let edge_best = (32u8, 130u8, 72u8);
        let fallback = (160u8, 160u8, 180u8);
        let min_d = 35 * 35;
        let palette = build_lineage_palette(14, edge_best, min_d, None);

        let snap = |seq: Vec<usize>| BeamHypothesisSnapshot {
            sequence: seq, log_prob_blank: 0.0, log_prob_non_blank: 0.0, combined_log_prob: -1.0,
        };
        // "ct" has both a stay parent "ct" and extend parent "c" — stay color wins
        let step0 = vec![snap(vec![3]), snap(vec![3, 4])];
        let step1 = vec![snap(vec![3, 4])];
        let steps = vec![step0, step1];
        let edges = derive_all_edges(&steps);
        let w = 2;
        let lineage = compute_lineage_rgb_per_node(&steps, &edges, w, &palette, fallback, edge_best, min_d);

        let stay_parent_col = lineage[0][1].expect("parent [3,4] color");
        let child_col = lineage[1][0].expect("child [3,4] color");
        assert_eq!(child_col, stay_parent_col, "stay-edge child must inherit parent's color");
    }

    #[test]
    fn prefix_beam_trace_matches_forward_decode() {
        let token_map = TokenMap::new(VOCAB);
        for (i, word) in FIXTURE_SEQS.iter().enumerate() {
            let (logits, decoder) = logits_and_decoder_for_word(word, 5, FIXTURE_LOGITS_SEED ^ i as u64);
            let trace = CtcPrefixBeamViz::beam_trace_sample0(&decoder, logits.clone())
                .unwrap_or_else(|e| panic!("trace failed: {e}"));
            let forward = decoder.forward(logits);
            assert_eq!(trace.decoded_ids().to_vec(), forward[0], "word={word:?}: trace decode must match decoder.forward");
            assert_eq!(trace.steps().len(), trace.timesteps + 1, "word={word:?}: snapshot count");
            assert_eq!(trace.greedy_argmax_per_t.len(), trace.timesteps, "word={word:?}: per-t argmax count");
            assert_eq!(trace.top_k_emissions_per_t.len(), trace.timesteps, "word={word:?}: top-k emissions rows");
            let expect_k = decoder.beam_width.min(trace.vocab_size.saturating_sub(1));
            for row in &trace.top_k_emissions_per_t {
                assert_eq!(row.len(), expect_k, "word={word:?}: top-k width should match beam extend K");
            }
            let ascii = CtcPrefixBeamViz::render_ascii(&trace, Some(&token_map));
            assert!(ascii.contains("Decoded Sequence:"), "word={word:?}: expected decoded line");
            assert!(ascii.contains("best path"), "word={word:?}: legend");
            assert!(ascii.contains("Greedy Argmax Logits:"), "word={word:?}: greedy preview line");
        }
    }

    #[test]
    fn fixture_noisy_logits_top_k_order_varies_across_timesteps() {
        let (logits, decoder) = logits_and_decoder_for_word("cat", 5, FIXTURE_LOGITS_SEED);
        let trace = CtcPrefixBeamViz::beam_trace_sample0(&decoder, logits).expect("trace");
        let rows = &trace.top_k_emissions_per_t;
        assert!(rows.len() >= 2, "need multiple frames");
        let first = &rows[0];
        assert!(rows.iter().any(|r| r != first), "expected different top-K ordering on at least one timestep");
    }

    fn fixture_beam_hypothesis(seq: Vec<usize>) -> BeamHypothesisSnapshot {
        BeamHypothesisSnapshot {
            sequence: seq, log_prob_blank: -1.0, log_prob_non_blank: -1.0, combined_log_prob: 0.0,
        }
    }

    fn fixture_prefix_beam_trace_reversed_ranks(beam_width: usize) -> PrefixBeamTrace {
        assert!(beam_width >= 2 && beam_width <= BLANK_ID,
            "beam_width must be in [2, BLANK_ID) for distinct token ids 0..beam_width-1");
        let k = beam_width;
        let step0: Vec<_> = (0..k).map(|_| fixture_beam_hypothesis(Vec::new())).collect();
        let step1: Vec<_> = (0..k).map(|i| fixture_beam_hypothesis(vec![i])).collect();
        let step2: Vec<_> = (0..k).map(|r| fixture_beam_hypothesis(vec![k - 1 - r])).collect();
        let dummy_k = (k + 1).min(VOCAB_SIZE);
        let dummy_ids: Vec<i64> = (0..dummy_k).map(|i| i as i64).collect();
        PrefixBeamTrace {
            blank_id: BLANK_ID, beam_width: k, timesteps: 2, vocab_size: VOCAB_SIZE,
            decoded: vec![0], greedy_argmax_per_t: vec![BLANK_ID as i64; 2],
            top_k_emissions_per_t: vec![dummy_ids.clone(), dummy_ids],
            steps: vec![step0, step1, step2],
        }
    }

    fn fixture_beam_edges_rank_reversal(k: usize) -> Vec<BeamEdge> {
        (0..k).map(|i| BeamEdge { parent_rank: i, child_rank: k - 1 - i }).collect()
    }

    #[test]
    fn prefix_beam_derive_edges_reversed_ranks() {
        let k = 8usize;
        let trace = fixture_prefix_beam_trace_reversed_ranks(k);
        let steps = trace.steps();
        let e01 = derive_edges(&steps[0], &steps[1]);
        assert_eq!(e01.len(), k * k, "ε at every rank extends to every singleton");
        let e12 = derive_edges(&steps[1], &steps[2]);
        let manual = fixture_beam_edges_rank_reversal(k);
        assert_eq!(e12.len(), manual.len());
        let mut a: Vec<_> = e12.iter().map(|e| (e.parent_rank, e.child_rank)).collect();
        let mut b: Vec<_> = manual.iter().map(|e| (e.parent_rank, e.child_rank)).collect();
        a.sort();
        b.sort();
        assert_eq!(a, b);
    }

    #[test]
    fn prefix_beam_compute_transition_reversed_ranks() {
        let k = 8usize;
        let edges = fixture_beam_edges_rank_reversal(k);
        let td = compute_transition(&edges, k);
        assert!(td.same_ranks.is_empty(), "pure permutation has no vertical stays");
        assert_eq!(td.cross.len(), k);
        assert!(td.num_layers >= 2, "expected layered routing, got num_layers={}", td.num_layers);
        let max_x = *td.max_xoff.iter().max().unwrap();
        assert!(max_x <= 3, "unexpected max_xoff={max_x:?} for K={k} reversal: {:?}", td.max_xoff);
    }

    #[test]
    fn prefix_beam_ascii_reversed_ranks_smoke() {
        let k = 8usize;
        let token_map = TokenMap::new(VOCAB);
        let trace = fixture_prefix_beam_trace_reversed_ranks(k);
        let ascii = CtcPrefixBeamViz::render_ascii(&trace, Some(&token_map));
        assert!(ascii.contains("t = 0"), "ascii:\n{ascii}");
        assert!(ascii.contains("t = 1"), "ascii:\n{ascii}");
        assert!(ascii.contains('│') || ascii.contains('├') || ascii.contains('┬'), "expected box-drawing routing");
        assert!(ascii.contains("[a]") || ascii.contains("(a)"), "ascii:\n{ascii}");
    }

    #[test]
    fn prefix_beam_ascii_reversed_ranks_printout() {
        let token_map = TokenMap::new(VOCAB);
        let trace = fixture_prefix_beam_trace_reversed_ranks(8);
        let graph = CtcPrefixBeamViz::render_ascii(&trace, Some(&token_map));
        println!("\n=== CTC Decode: ASCII reversed-rank routing fixture (beam=8) ===\n\n{graph}\n");
        assert!(graph.contains("Decoded Sequence:"));
        assert!(graph.contains("t = 0"));
        assert!(graph.contains("t = 1"));
    }

    #[test]
    fn prefix_beam_trace_errors_on_greedy() {
        let (logits, _) = logits_and_decoder_for_word("hi", 3, FIXTURE_LOGITS_SEED);
        let greedy = CtcDecoderConfig::new()
            .with_blank_id(BLANK_ID)
            .with_search_type(CtcDecodeType::GreedySearch)
            .init();
        let err = CtcPrefixBeamViz::beam_trace_sample0(&greedy, logits).unwrap_err();
        assert!(err.contains("BeamSearch"), "{err}");
    }

    #[test]
    fn ascii_graph_structure() {
        let token_map = TokenMap::new(VOCAB);
        for (i, word) in FIXTURE_SEQS.iter().enumerate() {
            let (logits, decoder) = logits_and_decoder_for_word(word, 4, FIXTURE_LOGITS_SEED ^ i as u64);
            let trace = CtcPrefixBeamViz::beam_trace_sample0(&decoder, logits).expect("trace");
            let graph = CtcPrefixBeamViz::render_ascii(&trace, Some(&token_map));
            assert!(graph.contains("Decoded Sequence:"), "word={word:?}: expected decoded line");
            assert!(graph.contains("best path"), "word={word:?}: legend");
            assert!(graph.contains('['), "word={word:?}: best-path brackets");
            assert!(graph.contains('('), "word={word:?}: other-hypothesis parens");
        }
    }

    #[test]
    fn ctc_decode_beam_ascii_printout() {
        let token_map = TokenMap::new(VOCAB);
        for (i, word) in FIXTURE_SEQS.iter().enumerate() {
            let (logits, decoder) = logits_and_decoder_for_word(word, 4, FIXTURE_LOGITS_SEED ^ i as u64);
            let trace = CtcPrefixBeamViz::beam_trace_sample0(&decoder, logits).expect("trace");
            let graph = CtcPrefixBeamViz::render_ascii(&trace, Some(&token_map));
            println!("\n=== CTC Decode: ASCII Prefix Beam Search Graph ===\n\n{graph}\n");
            assert!(graph.contains("Decoded Sequence:"), "word={word:?}: expected decoded line");
            assert!(graph.contains("best path"), "word={word:?}: expected bracket legend");
        }
    }

    #[test]
    #[ignore = "writes outputs/ctc_decode_beam_*.svg; run with --ignored to regenerate"]
    fn ctc_decode_beam_svg_export() {
        let token_map = TokenMap::new(VOCAB);
        let out_dir = outputs_dir();
        std::fs::create_dir_all(&out_dir).expect("create outputs/");
        for (seq_idx, word) in FIXTURE_SEQS.iter().enumerate() {
            let (logits, decoder) = logits_and_decoder_for_word(word, 5, FIXTURE_LOGITS_SEED ^ seq_idx as u64);
            let trace = CtcPrefixBeamViz::beam_trace_sample0(&decoder, logits).expect("trace");
            let filename = format!("ctc_decode_beam_{seq_idx:02}.svg");
            let out = out_dir.join(&filename);
            CtcPrefixBeamViz::write_svg(&trace, &out, Some(&token_map))
                .unwrap_or_else(|e| panic!("write svg {}: {e}", out.display()));
            let shown = out.canonicalize().unwrap_or_else(|_| out.clone());
            println!("wrote {}  (seq[{seq_idx:02}] = {:?})", shown.display(), word);
        }
    }
}
