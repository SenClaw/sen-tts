//! Native macOS TTS backend — drives `/usr/bin/say` from pure Rust.
//!
//! Pure-Rust in the sense that matters: no Python, no third-party native
//! library, no FFI. macOS ships `say(1)` as part of the OS, and we shell out
//! to it through `std::process::Command`, asking for WAVE/Int16 output so we
//! can stream the result straight back to the caller.
//!
//! A single [`MacosSpeech`] type backs **multiple catalog entries**, one per
//! language preset. The dispatcher in [`super::select_backend`] picks the
//! preset by `model_id` (`"macos-speech"` → Vietnamese, `"macos-speech-en"` →
//! English, …). Adding a new language is just adding a `const` here + a
//! dispatch arm + a catalog row.
//!
//! On non-macOS builds the backend compiles but always returns
//! [`TtsError::Unavailable`] — the dispatcher exposes that as `503`.

use super::{SynthesisRequest, TtsBackend, TtsError};

/// Default speaking rate for `say(1)`, in words per minute. macOS's stock
/// baseline; the caller's `speed` multiplier scales this.
const SAY_BASE_WPM: f32 = 175.0;

/// One catalog-visible flavour of the macOS speech backend.
///
/// Constants below ([`MacosSpeech::VIETNAMESE`], [`MacosSpeech::ENGLISH`]) cover
/// the languages the user can pick today; extend by adding another `const`.
pub struct MacosSpeech {
    /// Stable public id (also the dispatcher key + catalog row).
    pub id: &'static str,
    /// Human label shown to ops/logs.
    pub label: &'static str,
    /// Voice picked when the caller doesn't override.
    pub default_voice: &'static str,
    /// Language assumed when the caller doesn't override.
    pub default_language: &'static str,
}

impl MacosSpeech {
    /// Vietnamese preset — voice `Linh` (ships with macOS).
    pub const VIETNAMESE: Self = Self {
        id: "macos-speech",
        label: "macOS native speech — Vietnamese (Linh)",
        default_voice: "Linh",
        default_language: "vi",
    };

    /// English preset — voice `Samantha` (always installed).
    pub const ENGLISH: Self = Self {
        id: "macos-speech-en",
        label: "macOS native speech — English (Samantha)",
        default_voice: "Samantha",
        default_language: "en",
    };

    /// Pick the macOS preset that best fits a language hint. Used by the
    /// auto-fallback in [`super::synthesize_with_fallback`] so an English
    /// request that falls back from a stub backend gets an English voice,
    /// not the Vietnamese default.
    pub fn for_language(lang: &str) -> Self {
        match lang {
            "en" | "en-US" | "en-GB" => Self::ENGLISH,
            _ => Self::VIETNAMESE,
        }
    }
}

impl TtsBackend for MacosSpeech {
    fn id(&self) -> &str {
        self.id
    }

    fn label(&self) -> &str {
        self.label
    }

    fn synthesize(&self, req: &SynthesisRequest<'_>) -> Result<Vec<u8>, TtsError> {
        #[cfg(target_os = "macos")]
        {
            synthesize_macos(self, req)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = req;
            Err(TtsError::Unavailable(
                "macos-speech is only available on macOS hosts".into(),
            ))
        }
    }
}

#[cfg(target_os = "macos")]
fn synthesize_macos(preset: &MacosSpeech, req: &SynthesisRequest<'_>) -> Result<Vec<u8>, TtsError> {
    // Unique temp output path — multiple synth calls in flight must not collide.
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let tmp = std::env::temp_dir().join(format!("senclaw-tts-{}-{nonce}.wav", std::process::id()));

    // Voice precedence: explicit request → preset default. We trust the preset
    // to match the catalog row the user picked.
    let effective_voice = req.voice.unwrap_or(preset.default_voice);
    let _ = preset.default_language; // currently informational; reserved for `-l`
    let rate = (SAY_BASE_WPM * req.speed) as u32;

    let mut cmd = std::process::Command::new("/usr/bin/say");
    cmd.args([
        "-o",
        &tmp.to_string_lossy(),
        "--file-format=WAVE",
        "--data-format=LEI16",
        "-v",
        effective_voice,
        "-r",
        &rate.to_string(),
        req.text,
    ]);

    let output = cmd
        .output()
        .map_err(|e| TtsError::Internal(format!("failed to execute /usr/bin/say: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(TtsError::Internal(format!(
            "macOS say synthesis failed: {stderr}"
        )));
    }

    let wav = std::fs::read(&tmp)
        .map_err(|e| TtsError::Internal(format!("failed to read synthesized WAV: {e}")))?;
    let _ = std::fs::remove_file(&tmp);

    if wav.is_empty() {
        return Err(TtsError::Internal(
            "macOS say produced an empty WAV file".into(),
        ));
    }
    Ok(wav)
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    fn looks_like_wav(b: &[u8]) -> bool {
        b.len() >= 12 && &b[0..4] == b"RIFF" && &b[8..12] == b"WAVE"
    }

    fn req<'a>(
        text: &'a str,
        lang: &'a str,
        voice: Option<&'a str>,
        speed: f32,
    ) -> SynthesisRequest<'a> {
        SynthesisRequest {
            text,
            language: lang,
            voice,
            speed,
            model_dir: None,
        }
    }

    #[test]
    fn macos_speech_produces_valid_wav() {
        let wav = MacosSpeech::VIETNAMESE
            .synthesize(&req(
                "Xin chào, đây là một kiểm tra.",
                "vi",
                Some("Linh"),
                1.0,
            ))
            .expect("macos-speech synthesis should succeed");
        assert!(
            wav.len() > 1024,
            "wav suspiciously small: {} bytes",
            wav.len()
        );
        assert!(looks_like_wav(&wav), "output is not a RIFF/WAVE file");
    }

    /// English preset must also produce valid audio — proves the second
    /// catalog row actually works on this host.
    #[test]
    fn macos_speech_english_preset_produces_valid_wav() {
        let wav = MacosSpeech::ENGLISH
            .synthesize(&req("Hello, this is a test.", "en", None, 1.0))
            .expect("macos-speech-en synthesis should succeed");
        assert!(wav.len() > 1024);
        assert!(looks_like_wav(&wav));
    }

    /// Slower rate ⇒ longer audio ⇒ bigger WAV.
    #[test]
    fn macos_speech_speed_changes_output_size() {
        let fast = MacosSpeech::VIETNAMESE
            .synthesize(&req("Một hai ba bốn năm sáu bảy.", "vi", None, 1.5))
            .expect("fast synth");
        let slow = MacosSpeech::VIETNAMESE
            .synthesize(&req("Một hai ba bốn năm sáu bảy.", "vi", None, 0.75))
            .expect("slow synth");
        assert!(
            slow.len() > fast.len(),
            "expected slow ({}) > fast ({}) bytes",
            slow.len(),
            fast.len()
        );
    }

    #[test]
    fn presets_have_stable_ids() {
        assert_eq!(MacosSpeech::VIETNAMESE.id(), "macos-speech");
        assert_eq!(MacosSpeech::ENGLISH.id(), "macos-speech-en");
        assert!(MacosSpeech::VIETNAMESE.label().contains("Vietnamese"));
        assert!(MacosSpeech::ENGLISH.label().contains("English"));
    }

    #[test]
    fn for_language_picks_english_for_en_codes() {
        assert_eq!(MacosSpeech::for_language("en").id(), "macos-speech-en");
        assert_eq!(MacosSpeech::for_language("en-US").id(), "macos-speech-en");
        assert_eq!(MacosSpeech::for_language("vi").id(), "macos-speech");
        // Unknown language → VN default (it's the project's primary locale).
        assert_eq!(MacosSpeech::for_language("xx").id(), "macos-speech");
    }
}
