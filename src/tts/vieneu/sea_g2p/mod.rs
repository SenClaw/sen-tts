//! Vendored Rust core of **sea-g2p** — Vietnamese/English text normalization +
//! grapheme-to-phoneme, by Phạm Nguyễn Ngọc Bảo.
//!
//! Source: <https://github.com/pnnbao97/sea-g2p> (v0.7.18, Apache-2.0).
//! Vendored because upstream ships only as a pyo3 `cdylib` (not on crates.io);
//! the copy here is byte-identical except the thin pyo3 wrapper is removed
//! (`vi_normalizer/mod.rs`: `#[pyclass]/#[pymethods]` attributes stripped and
//! `normalize_batch` loses its `Python<'_>` token). Keep in sync with upstream
//! when bumping — this is the exact frontend VieNeu-TTS was trained with.
//!
//! Runtime data: [`g2p::G2PEngine`] memory-maps the `sea_g2p.bin` phoneme
//! dictionary (~50 MB, shipped in the sea-g2p wheel; our downloader extracts it
//! into the VieNeu model dir).

pub mod g2p;
pub mod punc;
pub mod vi_normalizer;

pub use g2p::G2PEngine;
pub use punc::apply_punc_norm;
pub use vi_normalizer::Normalizer;

/// `SEAPipeline.run(text, punc_norm=True)` equivalent: normalize (with terminal
/// punctuation normalization) then phonemize.
pub struct SeaPipeline {
    normalizer: Normalizer,
    g2p: G2PEngine,
}

impl SeaPipeline {
    pub fn new(dict_path: &str) -> std::io::Result<Self> {
        Ok(Self {
            normalizer: Normalizer::new("vi"),
            g2p: G2PEngine::new(dict_path)?,
        })
    }

    /// Normalize + G2P. `punc_norm` mirrors the Python kwarg.
    pub fn run(&self, text: &str, punc_norm: bool) -> String {
        if text.is_empty() {
            return String::new();
        }
        let normalized = self.normalizer.normalize(text, punc_norm);
        self.g2p.phonemize(&normalized)
    }
}
