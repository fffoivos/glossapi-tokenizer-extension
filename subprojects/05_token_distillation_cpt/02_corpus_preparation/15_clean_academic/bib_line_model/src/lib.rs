//! Greek bibliography LINE classifier — Rust port of the deployed `heading_lexgate`
//! stack (`line_hist_v3` at threshold 0.9, no citation-grammar features).
//!
//! Contract: **decision-equivalent**. The emitted line mask must match the Python
//! pipeline document-for-document; probabilities may differ in the last bits.
//! Fitted parameters come from `eval/sequence_models/export_line_model_artifacts.py`
//! — this crate never re-fits anything.
//!
//! Chain:
//!   deterministic per-line features
//!     -> P0D entry (HistGB x5)
//!     -> signal TCN (x5)          [pending]
//!     -> heading bundle (TF-IDF + logistic)
//!     -> connector bundle
//!     -> bidirectional context features
//!     -> line model (HistGB x5) -> threshold

pub mod artifacts;
pub mod chain;
pub mod connector;
pub mod features;
pub mod gaps;
pub mod patterns;
pub mod predict;
pub mod roles;
pub mod shape;
pub mod structure;
pub mod table;
pub mod tcn;
pub mod unicode;

pub use artifacts::{Artifacts, Manifest};
