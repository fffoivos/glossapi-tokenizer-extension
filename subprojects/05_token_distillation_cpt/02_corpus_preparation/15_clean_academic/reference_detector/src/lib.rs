//! Greek academic reference/citation detector — library root.
//!
//! See `reference_module` for the contract. This crate is standalone now but its
//! two core files (`reference_signals`, `reference_module`) are written to be lifted
//! into `glossapi_rs_cleaner` as a `reference_module.rs` consumer with no behavioral
//! change (same split-counter convention as `CleanStats`).

pub mod reference_signals;
pub mod reference_module;
pub mod beta_gate_model;
pub mod span_line_model;
pub mod toc_line_model;

pub use reference_module::{
    detect_doc, detect_sections, Counters, DetectConfig, DocResult, SectionRow, Span,
};
pub use span_line_model::span_detect;
pub use toc_line_model::toc_detect;
