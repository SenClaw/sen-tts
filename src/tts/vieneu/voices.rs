//! Preset voices for VieNeu-TTS v3 Turbo (`voices_v3_turbo.json`).
//!
//! Each preset carries the enrollment pair the engine needs: a 192-d speaker
//! embedding plus in-context MOSS reference codes `(T, n_vq)`. The JSON ships
//! in the upstream GitHub repo (Apache-2.0); our downloader drops it into the
//! model directory.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct VoicesFile {
    #[serde(default)]
    pub default_voice: Option<String>,
    pub presets: HashMap<String, Preset>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Preset {
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub gender: String,
    #[serde(default)]
    pub style: Option<String>,
    pub speaker_emb: Vec<f32>,
    /// Reference codes, row-major `(T, n_vq)`.
    pub codes: Vec<Vec<i64>>,
}

pub struct Voices {
    pub default_voice: String,
    pub presets: HashMap<String, Preset>,
    /// casefolded+diacritic-insensitive name → canonical key
    lookup: HashMap<String, String>,
}

/// Lowercase + strip combining marks so "pham tuyen" matches "Phạm Tuyên".
fn fold(name: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    name.nfd()
        .filter(|c| !unicode_normalization::char::is_combining_mark(*c))
        .collect::<String>()
        .to_lowercase()
        .replace('đ', "d")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

impl Voices {
    pub fn load(path: &Path) -> Result<Self> {
        let s =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let f: VoicesFile = serde_json::from_str(&s).context("parsing voices_v3_turbo.json")?;
        if f.presets.is_empty() {
            return Err(anyhow!("voices file has no presets"));
        }
        let default_voice = f
            .default_voice
            .clone()
            .filter(|d| f.presets.contains_key(d))
            .unwrap_or_else(|| f.presets.keys().next().unwrap().clone());
        let lookup = f.presets.keys().map(|k| (fold(k), k.clone())).collect();
        Ok(Self {
            default_voice,
            presets: f.presets,
            lookup,
        })
    }

    /// Resolve a user-supplied voice name (or empty → default). Falls back to
    /// an error listing available names so the UI can show them.
    pub fn get(&self, name: Option<&str>) -> Result<(&str, &Preset)> {
        let want = match name.map(str::trim).filter(|s| !s.is_empty()) {
            None => self.default_voice.as_str(),
            Some(n) => self
                .lookup
                .get(&fold(n))
                .map(String::as_str)
                .ok_or_else(|| {
                    let mut names: Vec<_> = self.presets.keys().cloned().collect();
                    names.sort();
                    anyhow!(
                        "unknown VieNeu voice `{n}` — available: {}",
                        names.join(", ")
                    )
                })?,
        };
        Ok((want, &self.presets[want]))
    }

    pub fn names(&self) -> Vec<&str> {
        self.presets.keys().map(String::as_str).collect()
    }

    /// Like [`Self::get`], but an unknown name falls back to the default
    /// preset instead of failing. Settings can carry a voice that belongs to
    /// another model (e.g. macOS "Linh" after switching to VieNeu) — playback
    /// must keep working, just with the default speaker.
    pub fn get_or_default(&self, name: Option<&str>) -> (&str, &Preset) {
        match self.get(name) {
            Ok(v) => v,
            Err(_) => {
                crate::safe_eprintln!(
                    "[vieneu] voice {:?} not found — using default `{}`",
                    name,
                    self.default_voice
                );
                (
                    self.default_voice.as_str(),
                    &self.presets[&self.default_voice],
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_diacritics_for_lookup() {
        assert_eq!(fold("Phạm  Tuyên"), "pham tuyen");
        assert_eq!(fold("Trúc Ly"), "truc ly");
        assert_eq!(fold("Đoan"), "doan");
    }

    #[test]
    fn loads_and_resolves_names() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("voices_v3_turbo.json");
        std::fs::write(&p, r#"{
            "default_voice": "Phạm Tuyên",
            "presets": {
                "Phạm Tuyên": {"description":"", "gender":"male", "speaker_emb":[0.1,0.2], "codes":[[1,2],[3,4]]},
                "Trúc Ly": {"description":"", "gender":"female", "speaker_emb":[0.3], "codes":[[5,6]]}
            }
        }"#).unwrap();
        let v = Voices::load(&p).unwrap();
        assert_eq!(v.get(None).unwrap().0, "Phạm Tuyên");
        assert_eq!(v.get(Some("truc ly")).unwrap().0, "Trúc Ly");
        assert_eq!(v.get(Some("PHAM TUYEN")).unwrap().0, "Phạm Tuyên");
        assert!(v
            .get(Some("nobody"))
            .unwrap_err()
            .to_string()
            .contains("available"));
    }
}
