//! TTS UI settings: selected model, voice preset, speed, language.
//!
//! Persisted as `<data_dir>/settings.json`, seeded once from the daemon's old
//! `ttsConfig` key ([`sen_runtime_sdk::legacy::load_or_import`]) and read per
//! request so a save applies to the next call without a restart. Unlike the
//! old daemon file this one is private to `sen-tts`, so there are no other
//! keys to preserve on a save.

use std::path::Path;

use sen_runtime_sdk::env::LaunchEnv;
use serde::{Deserialize, Serialize};

const LEGACY_KEY: &str = "ttsConfig";

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct TtsSettings {
    /// HuggingFace model id of the selected TTS model.
    #[serde(rename = "modelId", default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// Voice preset (model-specific string, e.g. speaker id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    /// Playback speed multiplier (0.25-4.0). `None` = model default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
    /// Language code: `"vi"` | `"en"`. `None` = model default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

pub fn load(env: &LaunchEnv) -> TtsSettings {
    match sen_runtime_sdk::legacy::load_or_import(&env.data_dir, &env.config_path, LEGACY_KEY) {
        Some(v) => serde_json::from_value(v).unwrap_or_default(),
        None => TtsSettings::default(),
    }
}

pub fn save(env: &LaunchEnv, settings: &TtsSettings) -> std::io::Result<()> {
    write_atomic(&env.data_dir, settings)
}

fn write_atomic(data_dir: &Path, settings: &TtsSettings) -> std::io::Result<()> {
    std::fs::create_dir_all(data_dir)?;
    let body = serde_json::to_string_pretty(settings).map_err(std::io::Error::other)?;
    let tmp = data_dir.join("settings.json.tmp");
    std::fs::write(&tmp, body)?;
    std::fs::rename(tmp, data_dir.join("settings.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_at(dir: &Path) -> LaunchEnv {
        LaunchEnv::from_lookup("sen-tts", "0.0.0-test", |k| match k {
            "SENCLAW_RUNTIME_DATA_DIR" => Some(dir.join("data").to_string_lossy().into_owned()),
            "SENCLAW_CONFIG_PATH" => Some(dir.join("config.json").to_string_lossy().into_owned()),
            "SENCLAW_HOME" => Some(dir.to_string_lossy().into_owned()),
            _ => None,
        })
    }

    #[test]
    fn imports_the_legacy_key_once_then_reads_its_own_file() {
        let tmp = tempfile::tempdir().unwrap();
        let env = env_at(tmp.path());
        std::fs::write(&env.config_path, r#"{"ttsConfig": {"modelId": "macos-speech", "voice": "Linh"}}"#).unwrap();

        let first = load(&env);
        assert_eq!(first.model_id.as_deref(), Some("macos-speech"));

        let mut updated = first;
        updated.speed = Some(1.25);
        save(&env, &updated).unwrap();
        std::fs::write(&env.config_path, r#"{"ttsConfig": {"modelId": "other"}}"#).unwrap();
        let second = load(&env);
        assert_eq!(second.model_id.as_deref(), Some("macos-speech"));
        assert_eq!(second.speed, Some(1.25));
    }

    #[test]
    fn no_legacy_file_means_defaults() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(load(&env_at(tmp.path())), TtsSettings::default());
    }
}
