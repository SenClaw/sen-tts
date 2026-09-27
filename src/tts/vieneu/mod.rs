//! **VieNeu-TTS v3 Turbo** backend — 48 kHz Vietnamese/English TTS by
//! Phạm Nguyễn Ngọc Bảo (`pnnbao-ump/VieNeu-TTS-v3-Turbo`, Apache-2.0).
//!
//! Runs the author's official torch-free path: the int8 ONNX transformer
//! graphs + the MOSS-Audio-Tokenizer-Nano ONNX codec, with the vendored
//! [`sea_g2p`] Rust frontend. CPU-only (ONNX Runtime) — unlike the MLX
//! backends this also works on Windows/Linux daemons.
//!
//! Enabled by the `tts-vieneu` build feature; without it the backend returns
//! [`TtsError::NotImplemented`] and the dispatcher falls back to the macOS
//! voice (X-TTS-Fallback header, never silent).
//!
//! ## Model directory layout (created by the composite downloader)
//! ```text
//! <tts-models>/pnnbao-ump__VieNeu-TTS-v3-Turbo/
//!   onnx_int8/            # prefill / decode_step / acoustic graphs + heads + tokenizer
//!   codec/                # moss_audio_tokenizer_decode_full.onnx + .data
//!   voices_v3_turbo.json  # 14 preset voices (speaker_emb + ref codes)
//!   sea_g2p.bin           # phoneme dictionary (from the sea-g2p wheel)
//! ```

#[cfg(feature = "tts-vieneu")]
pub mod engine;
#[cfg(feature = "tts-vieneu")]
pub mod npz;
#[cfg(feature = "tts-vieneu")]
pub mod phonemize;
#[cfg(feature = "tts-vieneu")]
pub mod sea_g2p;
#[cfg(feature = "tts-vieneu")]
pub mod voices;

use super::{SynthesisRequest, TtsBackend, TtsError};

/// Canonical HF id of the supported checkpoint.
pub const MODEL_ID: &str = "pnnbao-ump/VieNeu-TTS-v3-Turbo";

/// Status message for builds without the ONNX runtime path.
pub const STUB_MESSAGE: &str =
    "VieNeu-TTS requires the `tts-vieneu` build feature (ONNX Runtime). This build \
     lacks it, so the request transparently falls back to the macOS native voice \
     (see X-TTS-Fallback header).";

/// Files that must exist for the model to count as installed.
pub const REQUIRED_FILES: &[&str] = &[
    "onnx_int8/vieneu_prefill.onnx",
    "onnx_int8/vieneu_decode_step.onnx",
    "onnx_int8/vieneu_acoustic_cached.onnx",
    "onnx_int8/vieneu_backbone_shared.data",
    "onnx_int8/vieneu_v3_heads.npz",
    "onnx_int8/config.json",
    "onnx_int8/tokenizer.json",
    "codec/moss_audio_tokenizer_decode_full.onnx",
    "codec/moss_audio_tokenizer_decode_shared.data",
    "voices_v3_turbo.json",
    "sea_g2p.bin",
];

/// True when every artifact of the composite layout is present.
pub fn dir_is_installed(dir: &std::path::Path) -> bool {
    REQUIRED_FILES.iter().all(|f| dir.join(f).exists())
}

pub struct VieNeuBackend;

impl TtsBackend for VieNeuBackend {
    fn id(&self) -> &str {
        MODEL_ID
    }

    fn label(&self) -> &str {
        "VieNeu-TTS v3 Turbo (48 kHz, ONNX)"
    }

    #[cfg(not(feature = "tts-vieneu"))]
    fn synthesize(&self, _req: &SynthesisRequest<'_>) -> Result<Vec<u8>, TtsError> {
        Err(TtsError::NotImplemented(STUB_MESSAGE.into()))
    }

    #[cfg(feature = "tts-vieneu")]
    fn synthesize(&self, req: &SynthesisRequest<'_>) -> Result<Vec<u8>, TtsError> {
        native::synthesize(req)
    }
}

#[cfg(feature = "tts-vieneu")]
mod native {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::{Mutex, Once};
    use std::time::{Duration, Instant};

    use once_cell::sync::Lazy;

    use super::engine::{VieNeuEngine, SAMPLE_RATE};
    use super::{SynthesisRequest, TtsError};
    use crate::tts::encode_wav_pcm16;

    /// Drop an idle engine after this long. Kept SHORT: even with the ORT
    /// arena off + malloc purges, a live engine pins several hundred MB (its
    /// weights) plus fragmented pages from the KV churn — reloading costs only
    /// ~1 s, so idle RAM wins.
    const IDLE_TTL: Duration = Duration::from_secs(60);
    const REAP_EVERY: Duration = Duration::from_secs(20);
    /// VieNeu chunks: max_new_frames=300 ≈ 24 s of audio — keep chunks well under.
    const MAX_CHUNK_CHARS: usize = 220;

    struct CacheEntry {
        engine: VieNeuEngine,
        last_used: Instant,
    }

    static CACHE: Lazy<Mutex<HashMap<PathBuf, CacheEntry>>> =
        Lazy::new(|| Mutex::new(HashMap::new()));
    static REAPER: Once = Once::new();

    /// Ask macOS malloc to return pooled pages to the OS. The generation loop
    /// frees thousands of small KV tensors per request; with ORT's arena off
    /// they land in malloc's magazines, which keep the pages resident (multi-GB
    /// RSS high-water). This is Apple's sanctioned "purge now" hook.
    #[cfg(target_os = "macos")]
    fn malloc_purge() {
        extern "C" {
            fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
        }
        unsafe {
            malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
        }
    }
    #[cfg(not(target_os = "macos"))]
    fn malloc_purge() {}

    fn spawn_reaper() {
        REAPER.call_once(|| {
            std::thread::Builder::new()
                .name("vieneu-reaper".into())
                .spawn(|| loop {
                    std::thread::sleep(REAP_EVERY);
                    let mut cache = CACHE.lock().unwrap_or_else(|p| p.into_inner());
                    let before = cache.len();
                    cache.retain(|_, e| e.last_used.elapsed() < IDLE_TTL);
                    let evicted = before - cache.len();
                    drop(cache);
                    if evicted > 0 {
                        malloc_purge(); // hand the engine's pages back to the OS
                    }
                })
                .ok();
        });
    }

    pub(super) fn synthesize(req: &SynthesisRequest<'_>) -> Result<Vec<u8>, TtsError> {
        let dir = req.model_dir.ok_or_else(|| {
            TtsError::BadInput("VieNeu-TTS requires a downloaded model directory".into())
        })?;
        if !super::dir_is_installed(dir) {
            let missing: Vec<&str> = super::REQUIRED_FILES
                .iter()
                .copied()
                .filter(|f| !dir.join(f).exists())
                .collect();
            return Err(TtsError::BadInput(format!(
                "VieNeu-TTS model is incomplete in {} — missing: {}",
                dir.display(),
                missing.join(", ")
            )));
        }
        spawn_reaper();

        let mut cache = CACHE.lock().unwrap_or_else(|p| p.into_inner());
        if !cache.contains_key(dir) {
            let engine = VieNeuEngine::load(dir)
                .map_err(|e| TtsError::Internal(format!("loading VieNeu-TTS: {e:#}")))?;
            cache.insert(
                dir.to_path_buf(),
                CacheEntry {
                    engine,
                    last_used: Instant::now(),
                },
            );
        }
        let entry = cache.get_mut(dir).expect("just inserted");
        entry.last_used = Instant::now();
        let engine = &mut entry.engine;

        // Clone the preset out so the immutable voices borrow ends before the
        // mutable synthesis calls (ort sessions need &mut). Unknown voices
        // (e.g. a stale setting from another model) use the default speaker.
        let preset = engine.voices.get_or_default(req.voice).1.clone();

        let gap = vec![0.0f32; (SAMPLE_RATE as usize) * 15 / 100]; // 150 ms
        let mut samples: Vec<f32> = Vec::new();
        let mut spoke_any = false;
        for chunk in crate::tts::chunk::chunk_text(req.text, MAX_CHUNK_CHARS) {
            let wav = engine
                .infer_text(&chunk, &preset)
                .map_err(|e| TtsError::Internal(format!("VieNeu synthesis: {e:#}")))?;
            if wav.is_empty() {
                continue;
            }
            if spoke_any {
                samples.extend_from_slice(&gap);
            }
            samples.extend(wav);
            spoke_any = true;
            // Return each chunk's KV churn to the OS before the next one so
            // fragmented pages don't stack across chunks.
            malloc_purge();
        }
        // The frame loop churned thousands of short-lived KV tensors — purge
        // malloc's magazines so RSS returns toward baseline after the request.
        malloc_purge();
        if !spoke_any {
            return Err(TtsError::BadInput(
                "text contains no speakable characters".into(),
            ));
        }
        Ok(encode_wav_pcm16(&samples, SAMPLE_RATE))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "tts-vieneu"))]
    #[test]
    fn stub_identifies_vieneu() {
        let r = VieNeuBackend.synthesize(&SynthesisRequest {
            text: "Xin chào.",
            language: "vi",
            voice: None,
            speed: 1.0,
            model_dir: None,
        });
        match r {
            Err(TtsError::NotImplemented(msg)) => assert!(msg.contains("VieNeu")),
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    #[cfg(feature = "tts-vieneu")]
    #[test]
    fn native_requires_model_dir() {
        let r = VieNeuBackend.synthesize(&SynthesisRequest {
            text: "Xin chào.",
            language: "vi",
            voice: None,
            speed: 1.0,
            model_dir: None,
        });
        assert!(matches!(r, Err(TtsError::BadInput(_))));
    }

    /// Real synthesis + voice listing. Run with:
    /// `cargo test --features tts-vieneu -- --ignored vieneu_native --test-threads=1`
    #[cfg(feature = "tts-vieneu")]
    #[test]
    #[ignore = "needs the downloaded VieNeu model dir"]
    fn vieneu_native_synthesizes_wav() {
        let dir = dirs::home_dir()
            .unwrap()
            .join(".senclaw/tts-models/pnnbao-ump__VieNeu-TTS-v3-Turbo");
        if !dir_is_installed(&dir) {
            eprintln!("VieNeu model not downloaded; skipping");
            return;
        }
        let wav = VieNeuBackend
            .synthesize(&SynthesisRequest {
                text: "Xin chào, đây là giọng đọc bốn tám kilô héc.",
                language: "vi",
                voice: None,
                speed: 1.0,
                model_dir: Some(&dir),
            })
            .expect("vieneu synthesis");
        assert!(&wav[0..4] == b"RIFF" && &wav[8..12] == b"WAVE");
        assert!(wav.len() > 96_000, "wav suspiciously small: {}", wav.len());
        std::fs::write("/tmp/vieneu_test.wav", &wav).ok();
    }
}
