//! CTC visualization: helpers (trace capture, ASCII printout, and SVG export for both loss and decode)
//!
//! - [`forward_lattice_viz`]: loss-time forward lattice heatmap and DP arrows.
//! - [`prefix_beam_viz`]: decode-time prefix beam DAG and emission chips.

pub mod forward_lattice_viz;
pub mod prefix_beam_viz;

/// Native system stack for SVG `font-family` (no embedded fonts; matches modern UI surfaces).
pub(crate) const SVG_UI_FONT: &str = r#"system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif"#;

pub use forward_lattice_viz::ForwardLatticeSvgTheme;
pub use prefix_beam_viz::PrefixBeamSvgTheme;
