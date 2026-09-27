//! `sen-tts`'s own HTTP surface: the old daemon namespace `/api/tts/*`, served
//! **verbatim** — same paths, request and response bodies, status codes —
//! plus OpenAI-compatible `POST /v1/audio/speech`. Ported from the daemon's
//! `src/gateway/ui_server/tts.rs` and the TTS half of `ui_server/hf_validate.rs`;
//! only the plumbing that read the daemon's `Config`/`UiState` changed, none
//! of the behaviour.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::{
    body::Body,
    extract::{Path as AxumPath, State},
    http::{HeaderValue, StatusCode},
    response::{IntoResponse, Json, Response},
    routing::get,
    Router,
};
use futures::StreamExt;
use once_cell::sync::Lazy;
use sen_runtime_sdk::api::ErrorBody;
use sen_runtime_sdk::env::LaunchEnv;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

use crate::settings_store::{self, TtsSettings};

const HF_BASE: &str = "https://huggingface.co";

/// Everything a handler needs besides the request itself.
pub struct AppState {
    pub env: LaunchEnv,
}

/// `SENCLAW_TTS_MODELS_DIR`, else `<SENCLAW_HOME>/tts-models` — engine-private,
/// so (unlike the shared `local-models` root) it is read directly rather than
/// through a field on [`LaunchEnv`].
fn models_root(env: &LaunchEnv) -> PathBuf {
    std::env::var("SENCLAW_TTS_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| env.home.join("tts-models"))
}

pub struct AppError(pub StatusCode, pub String);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.0, Json(ErrorBody::new(self.1))).into_response()
    }
}

// ── Catalog ──────────────────────────────────────────────────────────────────

struct TtsCatalogEntry {
    /// Public HuggingFace repo id (also the weights repo).
    id: &'static str,
    label: &'static str,
    approx_size_gb: f32,
    /// Supported language codes.
    languages: &'static [&'static str],
    default_language: &'static str,
    /// Short description shown in the UI.
    description: &'static str,
}

static CATALOG: &[TtsCatalogEntry] = &[
    TtsCatalogEntry {
        id: "macos-speech",
        label: "System Speech (macOS) — Vietnamese (Linh)",
        approx_size_gb: 0.0,
        languages: &["vi"],
        default_language: "vi",
        description: "Zero-dependency macOS native voice. Vietnamese (Linh).",
    },
    TtsCatalogEntry {
        id: "macos-speech-en",
        label: "System Speech (macOS) — English (Samantha)",
        approx_size_gb: 0.0,
        languages: &["en"],
        default_language: "en",
        description: "Zero-dependency macOS native voice. English (Samantha).",
    },
    TtsCatalogEntry {
        id: "pnnbao-ump/VieNeu-TTS-v3-Turbo",
        label: "VieNeu-TTS v3 Turbo (48 kHz, 14 giọng)",
        approx_size_gb: 0.31,
        languages: &["vi", "en"],
        default_language: "vi",
        description: "VieNeu-TTS v3 Turbo (Phạm Nguyễn Ngọc Bảo) — 48 kHz, 14 preset Vietnamese voices, En–Vi code-switching, emotion cues ([cười], [thở dài]). Runs the official ONNX path on CPU (built with the tts-vieneu feature). Set the Voice field to a preset name (default: Phạm Tuyên). Composite download: ONNX graphs + MOSS codec + voices + phoneme dictionary.",
    },
    // NOTE: the MLX voices are gone, not hidden. ZipVoice never synthesized
    // anything and had already been dropped from this catalog; MMS-VITS worked
    // but was the last thing keeping `mlx-rs` on the TTS path, for a voice no
    // machine had selected. VieNeu covers Vietnamese at higher quality, and
    // TTS needs ONNX on the CPU and nothing else — which is why this runtime
    // compiles no MLX at all.
];

fn catalog_get(id: &str) -> Option<&'static TtsCatalogEntry> {
    CATALOG.iter().find(|e| e.id == id)
}

/// Selectable voices for a model, if it exposes any: `(voices, default)`.
/// VieNeu reads its preset list from the downloaded `voices_v3_turbo.json`
/// (name + description + gender per voice); the macOS presets are static.
fn model_voices(id: &str, dir: &std::path::Path) -> (Vec<Value>, Option<String>) {
    match id {
        "macos-speech" => (
            vec![json!({"name": "Linh", "description": "Giọng nữ tiếng Việt (macOS)"})],
            Some("Linh".to_string()),
        ),
        "macos-speech-en" => (
            vec![json!({"name": "Samantha", "description": "English female (macOS)"})],
            Some("Samantha".to_string()),
        ),
        _ if id == crate::tts::vieneu::MODEL_ID => {
            let Ok(s) = std::fs::read_to_string(dir.join("voices_v3_turbo.json")) else {
                return (Vec::new(), None);
            };
            let Ok(v) = serde_json::from_str::<Value>(&s) else {
                return (Vec::new(), None);
            };
            let default_voice = v["default_voice"].as_str().map(str::to_string);
            let mut voices: Vec<Value> = v["presets"]
                .as_object()
                .map(|m| {
                    m.iter()
                        .map(|(name, p)| {
                            json!({
                                "name": name,
                                "description": p["description"].as_str().unwrap_or(""),
                                "gender": p["gender"].as_str().unwrap_or(""),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            voices.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
            (voices, default_voice)
        }
        _ => (Vec::new(), None),
    }
}

fn safe_dirname(id: &str) -> String {
    id.replace('/', "__")
}

fn unsafe_dirname(name: &str) -> Option<String> {
    let (org, repo) = name.split_once("__")?;
    if org.is_empty() || repo.is_empty() {
        return None;
    }
    Some(format!("{org}/{repo}"))
}

fn model_dir(state: &AppState, id: &str) -> PathBuf {
    models_root(&state.env).join(safe_dirname(id))
}

/// A TTS model is considered installed if it is a built-in system voice
/// (any `macos-speech*` preset) or if the directory contains weights.
fn is_installed(state: &AppState, id: &str) -> bool {
    if id.starts_with("macos-speech") {
        return true;
    }
    let dir = model_dir(state, id);
    if id == crate::tts::vieneu::MODEL_ID {
        return crate::tts::vieneu::dir_is_installed(&dir);
    }
    dir.join("config.json").exists()
        && (dir.join("model.safetensors").exists()
            || dir.join("weights.npz").exists()
            || dir.join("model.npz").exists()
            || dir.join("model.safetensors.index.json").exists())
}

// ── Download progress ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DownloadStatus {
    Queued,
    Listing,
    Downloading,
    Done,
    Error,
    Cancelled,
}

#[derive(Debug, Clone, Serialize)]
struct DownloadState {
    model_id: String,
    status: DownloadStatus,
    total_bytes: u64,
    downloaded_bytes: u64,
    current_file: Option<String>,
    files_total: u32,
    files_done: u32,
    error: Option<String>,
}

#[derive(Clone)]
struct DownloadHandle {
    state: Arc<Mutex<DownloadState>>,
    cancel: CancellationToken,
}

static DOWNLOADS: Lazy<Mutex<HashMap<String, DownloadHandle>>> = Lazy::new(|| Mutex::new(HashMap::new()));

// ── Routes: model listing ─────────────────────────────────────────────────────

async fn tts_models_list(State(state): State<Arc<AppState>>) -> Result<impl IntoResponse, AppError> {
    let downloads = DOWNLOADS.lock().unwrap();
    let mut models = Vec::new();

    for e in CATALOG {
        let dir = model_dir(&state, e.id);
        let download = downloads.get(e.id).map(|h| h.state.lock().unwrap().clone());
        let (voices, default_voice) = model_voices(e.id, &dir);
        models.push(json!({
            "id": e.id,
            "label": e.label,
            "approx_size_gb": e.approx_size_gb,
            "languages": e.languages,
            "default_language": e.default_language,
            "description": e.description,
            "installed": is_installed(&state, e.id),
            "on_disk_path": dir.to_string_lossy(),
            "custom": false,
            "download": download,
            "voices": voices,
            "default_voice": default_voice,
        }));
    }

    if let Ok(entries) = std::fs::read_dir(models_root(&state.env)) {
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else { continue };
            if !file_type.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            let Some(id) = unsafe_dirname(&name) else { continue };
            if catalog_get(&id).is_some() || models.iter().any(|m| m["id"] == id) {
                continue;
            }
            let dir = entry.path();
            let download = downloads.get(&id).map(|h| h.state.lock().unwrap().clone());
            if is_installed(&state, &id) || download.is_some() {
                models.push(json!({
                    "id": id,
                    "label": format!("TTS custom ({id})"),
                    "approx_size_gb": 0.0,
                    "languages": ["vi", "en"],
                    "default_language": "vi",
                    "description": "",
                    "installed": is_installed(&state, &id),
                    "on_disk_path": dir.to_string_lossy(),
                    "custom": true,
                    "download": download,
                }));
            }
        }
    }

    for (id, handle) in downloads.iter() {
        if catalog_get(id).is_some() || models.iter().any(|m| m["id"] == *id) {
            continue;
        }
        let dir = model_dir(&state, id);
        models.push(json!({
            "id": id,
            "label": format!("TTS custom ({id})"),
            "approx_size_gb": 0.0,
            "languages": ["vi", "en"],
            "default_language": "vi",
            "description": "",
            "installed": is_installed(&state, id),
            "on_disk_path": dir.to_string_lossy(),
            "custom": true,
            "download": handle.state.lock().unwrap().clone(),
        }));
    }

    Ok(Json(json!({ "models": models })))
}

// ── Routes: download ──────────────────────────────────────────────────────────

/// Normalize a HuggingFace `org/repo` id from bare id or full URL.
fn normalize_hf_id(raw: &str) -> Result<String, String> {
    let s = raw.trim();
    if s.is_empty() {
        return Err("empty model id".into());
    }
    let stripped = s
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_start_matches("huggingface.co/")
        .trim_start_matches("hf.co/")
        .trim_end_matches('/');
    let parts: Vec<&str> = stripped.split('/').collect();
    if parts.len() < 2 {
        return Err(format!("expected `org/repo` form, got `{s}`"));
    }
    let org = parts[0];
    let repo = parts[1];
    if org.is_empty() || repo.is_empty() {
        return Err(format!("invalid `org/repo` in `{s}`"));
    }
    for seg in [org, repo] {
        if seg.contains("..") || seg.contains('\\') {
            return Err(format!("unsafe path segment in `{s}`"));
        }
    }
    Ok(format!("{org}/{repo}"))
}

async fn tts_download(
    State(state): State<Arc<AppState>>,
    AxumPath(id): AxumPath<String>,
) -> Result<impl IntoResponse, AppError> {
    let id = normalize_hf_id(&id).map_err(|e| AppError(StatusCode::BAD_REQUEST, e))?;

    {
        let downloads = DOWNLOADS.lock().unwrap();
        if let Some(h) = downloads.get(&id) {
            let s = h.state.lock().unwrap();
            if matches!(s.status, DownloadStatus::Queued | DownloadStatus::Listing | DownloadStatus::Downloading) {
                return Err(AppError(StatusCode::CONFLICT, format!("download for {id} already in progress")));
            }
        }
    }

    let dir = model_dir(&state, &id);
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let progress = Arc::new(Mutex::new(DownloadState {
        model_id: id.clone(),
        status: DownloadStatus::Queued,
        total_bytes: 0,
        downloaded_bytes: 0,
        current_file: None,
        files_total: 0,
        files_done: 0,
        error: None,
    }));
    let cancel = CancellationToken::new();
    DOWNLOADS.lock().unwrap().insert(id.clone(), DownloadHandle { state: progress.clone(), cancel: cancel.clone() });

    let weights_repo = id.clone();
    tokio::spawn(async move {
        let result = if weights_repo == crate::tts::vieneu::MODEL_ID {
            run_vieneu_download(&dir, progress.clone(), cancel).await
        } else {
            run_tts_download(&weights_repo, &dir, progress.clone(), cancel).await
        };
        let mut s = progress.lock().unwrap();
        match result {
            Ok(()) if s.status != DownloadStatus::Cancelled => s.status = DownloadStatus::Done,
            Ok(()) => {}
            Err(e) => {
                s.status = DownloadStatus::Error;
                s.error = Some(e.to_string());
            }
        }
    });

    Ok(Json(json!({ "ok": true, "id": id })))
}

async fn tts_status(AxumPath(id): AxumPath<String>) -> Result<impl IntoResponse, AppError> {
    let downloads = DOWNLOADS.lock().unwrap();
    let progress = downloads.get(&id).map(|h| h.state.lock().unwrap().clone());
    Ok(Json(json!({ "id": id, "download": progress })))
}

async fn tts_cancel(AxumPath(id): AxumPath<String>) -> Result<impl IntoResponse, AppError> {
    let downloads = DOWNLOADS.lock().unwrap();
    if let Some(h) = downloads.get(&id) {
        h.cancel.cancel();
        h.state.lock().unwrap().status = DownloadStatus::Cancelled;
    }
    Ok(Json(json!({ "ok": true })))
}

async fn tts_delete(State(state): State<Arc<AppState>>, AxumPath(id): AxumPath<String>) -> Result<impl IntoResponse, AppError> {
    let dir = model_dir(&state, &id);
    if dir.exists() {
        tokio::fs::remove_dir_all(&dir)
            .await
            .map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }
    DOWNLOADS.lock().unwrap().remove(&id);
    Ok(Json(json!({ "ok": true })))
}

// ── Routes: settings ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct TtsSettingsBody {
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    speed: Option<f32>,
    #[serde(default)]
    language: Option<String>,
}

async fn tts_settings_get(State(state): State<Arc<AppState>>) -> Result<impl IntoResponse, AppError> {
    let s = settings_store::load(&state.env);
    Ok(Json(json!({
        "model_id": s.model_id.unwrap_or_else(|| "macos-speech".to_string()),
        "voice": s.voice.unwrap_or_else(|| "Linh".to_string()),
        "speed": s.speed.unwrap_or(1.0),
        "language": s.language.unwrap_or_else(|| "vi".to_string()),
    })))
}

async fn tts_settings_put(
    State(state): State<Arc<AppState>>,
    Json(body): Json<TtsSettingsBody>,
) -> Result<impl IntoResponse, AppError> {
    if let Some(spd) = body.speed {
        if !(0.25..=4.0).contains(&spd) {
            return Err(AppError(StatusCode::BAD_REQUEST, "speed must be between 0.25 and 4.0".into()));
        }
    }
    let settings = TtsSettings { model_id: body.model_id, voice: body.voice, speed: body.speed, language: body.language };
    settings_store::save(&state.env, &settings).map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(json!({ "ok": true })))
}

// ── Routes: HF pre-download validation ────────────────────────────────────────

#[derive(Debug, Serialize)]
struct ValidateReport {
    id: String,
    supported: bool,
    reason: String,
    architecture: Option<String>,
    total_size_bytes: u64,
    gated: bool,
    inconclusive: bool,
}

async fn tts_validate(AxumPath(id): AxumPath<String>) -> Result<impl IntoResponse, AppError> {
    let id = normalize_hf_id(&id).map_err(|e| AppError(StatusCode::BAD_REQUEST, e))?;
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let info_url = format!("{HF_BASE}/api/models/{id}");
    let info = match client.get(&info_url).send().await {
        // HF answers 401 (not 404) for unknown repos so private repo names
        // don't leak — treat all three as "not there for us".
        Ok(r) if matches!(r.status(), StatusCode::NOT_FOUND | StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) => {
            return Ok(Json(ValidateReport {
                id,
                supported: false,
                reason: "model not found on Hugging Face (or it is private) — check the org/repo id".into(),
                architecture: None,
                total_size_bytes: 0,
                gated: false,
                inconclusive: false,
            }));
        }
        Ok(r) if r.status().is_success() => r.json::<Value>().await.ok(),
        _ => None,
    };
    let Some(info) = info else {
        return Ok(Json(ValidateReport {
            id,
            supported: false,
            reason: "could not reach the Hugging Face API — verdict unavailable; you may still try downloading".into(),
            architecture: None,
            total_size_bytes: 0,
            gated: false,
            inconclusive: true,
        }));
    };
    let gated = !matches!(info.get("gated"), None | Some(Value::Bool(false)));
    let private = info.get("private").and_then(Value::as_bool).unwrap_or(false);
    if gated || private {
        return Ok(Json(ValidateReport {
            id,
            supported: false,
            reason: "repo is gated/private — the built-in downloader has no Hugging Face login".into(),
            architecture: None,
            total_size_bytes: 0,
            gated: true,
            inconclusive: false,
        }));
    }

    let tree_url = format!("{HF_BASE}/api/models/{id}/tree/main?recursive=true");
    let tree: Vec<Value> = match client.get(&tree_url).send().await {
        Ok(r) if r.status().is_success() => r.json().await.unwrap_or_default(),
        _ => Vec::new(),
    };
    let files: Vec<(String, u64)> = tree
        .iter()
        .filter(|e| e["type"] == "file")
        .map(|e| (e["path"].as_str().unwrap_or_default().to_string(), e["size"].as_u64().unwrap_or(0)))
        .collect();
    let total_size_bytes: u64 = files.iter().map(|f| f.1).sum();
    let has_file = |name: &str| files.iter().any(|(p, _)| p == name);
    let has_safetensors = has_file("model.safetensors") || has_file("model.safetensors.index.json");

    let cfg_url = format!("{HF_BASE}/{id}/resolve/main/config.json");
    let cfg: Option<Value> = match client.get(&cfg_url).send().await {
        Ok(r) if r.status().is_success() => r.json().await.ok(),
        _ => None,
    };

    let (supported, reason, architecture) = match &cfg {
        None => (false, "repo has no readable config.json — cannot determine the architecture".to_string(), None),
        Some(cfg) => check_tts(cfg, has_safetensors, &has_file),
    };

    Ok(Json(ValidateReport {
        id,
        supported,
        reason,
        architecture,
        total_size_bytes,
        gated: false,
        inconclusive: cfg.is_none() && files.is_empty(),
    }))
}

/// TTS rule — the HF `VitsModel` shape, plus VieNeu's own architecture name.
/// Mirrors the daemon's old `hf_validate::check_tts` (kept after the MLX
/// voices were removed: "Add model from Hugging Face" still accepts any
/// VitsModel repo, and validating the config before spending a download is
/// worth more than the backend that used to consume it).
fn check_tts(cfg: &Value, has_safetensors: bool, has_file: &dyn Fn(&str) -> bool) -> (bool, String, Option<String>) {
    let arch = cfg["architectures"][0].as_str().map(str::to_string);
    if arch.as_deref().is_some_and(|a| a.starts_with("VieNeu")) {
        return (
            true,
            "VieNeu-TTS v3 Turbo — supported via the ONNX runtime backend (built with tts-vieneu); download is composite (ONNX graphs + MOSS codec + voices + phoneme dictionary)".into(),
            arch,
        );
    }
    let is_vits = arch.as_deref().is_some_and(|a| a.starts_with("VitsModel"));
    if !is_vits {
        let a = arch.clone().unwrap_or_else(|| "unknown".into());
        return (
            false,
            format!(
                "architecture `{a}` is not supported for TTS — only HF VitsModel checkpoints \
                 (facebook/mms-tts-* and finetunes) run natively today"
            ),
            arch,
        );
    }
    let speakers = cfg["num_speakers"].as_u64().unwrap_or(1);
    let spk_embed = cfg["speaker_embedding_size"].as_u64().unwrap_or(0);
    if speakers > 1 || spk_embed > 0 {
        return (false, "multi-speaker VITS checkpoints are not supported yet (single-speaker MMS only)".into(), arch);
    }
    if !has_safetensors {
        return (false, "no model.safetensors in the repo (only .bin/.onnx?) — the native loader needs safetensors".into(), arch);
    }
    if !has_file("vocab.json") {
        return (false, "missing vocab.json (tokenizer)".into(), arch);
    }
    (true, "HF VitsModel (MMS family) — runs on the native pure-Rust VITS backend".into(), arch)
}

// ── Routes: synthesize ────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct SynthesizeBody {
    pub text: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default)]
    pub speed: Option<f32>,
    /// Model id override; if omitted uses the persisted settings model_id.
    #[serde(default)]
    pub model_id: Option<String>,
}

async fn tts_synthesize(State(state): State<Arc<AppState>>, Json(body): Json<SynthesizeBody>) -> Result<Response, AppError> {
    if body.text.trim().is_empty() {
        return Err(AppError(StatusCode::BAD_REQUEST, "text is empty".into()));
    }
    synthesize_response(&state, body.model_id, body.text, body.language, body.voice, body.speed).await
}

/// OpenAI-compatible `POST /v1/audio/speech`:
/// `{model?, input, voice?, response_format: "wav", speed?}` → WAV bytes.
/// Shares the fallback/header behaviour of `/api/tts/synthesize` — the wire
/// shape is OpenAI's, the engine underneath is the same.
#[derive(Deserialize)]
struct AudioSpeechBody {
    #[serde(default)]
    model: Option<String>,
    input: String,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    speed: Option<f32>,
}

async fn audio_speech(State(state): State<Arc<AppState>>, Json(body): Json<AudioSpeechBody>) -> Result<Response, AppError> {
    if body.input.trim().is_empty() {
        return Err(AppError(StatusCode::BAD_REQUEST, "input is empty".into()));
    }
    if let Some(fmt) = &body.response_format {
        if fmt != "wav" {
            return Err(AppError(StatusCode::BAD_REQUEST, format!("response_format `{fmt}` is not supported — only \"wav\"")));
        }
    }
    synthesize_response(&state, body.model, body.input, None, body.voice, body.speed).await
}

/// Resolve settings + backend, synthesize (with honest auto-fallback to
/// `macos-speech`), and build the WAV response both synthesis routes share.
async fn synthesize_response(
    state: &AppState,
    model_id: Option<String>,
    text: String,
    language: Option<String>,
    voice: Option<String>,
    speed: Option<f32>,
) -> Result<Response, AppError> {
    let settings = settings_store::load(&state.env);
    let model_id = model_id.or_else(|| settings.model_id.clone()).unwrap_or_else(|| "macos-speech".to_string());

    if !is_installed(state, &model_id) {
        return Err(AppError(StatusCode::BAD_REQUEST, format!("TTS model `{model_id}` is not installed")));
    }

    let language = language.or_else(|| settings.language.clone()).unwrap_or_else(|| "vi".to_string());
    let speed = speed.or(settings.speed).unwrap_or(1.0);
    // Voice must fall back to the persisted setting like language/speed do —
    // chat read-aloud (and a bare OpenAI call) sends only the text, and
    // without this it always spoke with the model's default voice instead of
    // the one picked in Settings.
    let voice = voice
        .clone()
        .filter(|v| !v.trim().is_empty())
        .or_else(|| settings.voice.clone().filter(|v| !v.trim().is_empty()));

    let model_path = if model_id.starts_with("macos-speech") { None } else { Some(model_dir(state, &model_id)) };

    let outcome = tokio::task::spawn_blocking(move || {
        crate::tts::synthesize_with_fallback(&model_id, model_path.as_deref(), &text, &language, voice.as_deref(), speed)
    })
    .await
    .map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| AppError(e.0, e.1))?;

    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", HeaderValue::from_static("audio/wav"))
        .header("Content-Disposition", HeaderValue::from_static("inline; filename=\"speech.wav\""))
        .header("Content-Length", outcome.wav.len().to_string())
        .header(
            "X-TTS-Backend",
            HeaderValue::from_str(&outcome.used_backend).unwrap_or_else(|_| HeaderValue::from_static("unknown")),
        );
    if let Some(reason) = &outcome.fallback_reason {
        // Strip control chars / non-ASCII so the header value stays valid.
        let ascii: String = reason.chars().map(|c| if c.is_ascii_graphic() || c == ' ' { c } else { '?' }).collect();
        if let Ok(v) = HeaderValue::from_str(&ascii) {
            builder = builder.header("X-TTS-Fallback", v);
        }
    }
    builder
        .body(Body::from(outcome.wav))
        .map_err(|e| AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

// ── HF download worker ────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct HfTreeEntry {
    #[serde(rename = "type")]
    entry_type: String,
    path: String,
    #[serde(default)]
    size: u64,
}

fn should_skip(name: &str) -> bool {
    let lower = name.to_lowercase();
    matches!(lower.as_str(), ".gitattributes" | "readme.md" | "license" | "license.md" | "license.txt")
        || lower.ends_with(".png")
        || lower.ends_with(".jpg")
        || lower.ends_with(".jpeg")
        || lower.ends_with(".gif")
        || lower.ends_with(".svg")
}

/// Composite download for VieNeu-TTS v3 Turbo — four sources into one model dir:
///   1. int8 ONNX graphs + config + tokenizer (HF `pnnbao-ump/VieNeu-TTS-v3-Turbo`,
///      subfolder `onnx_int8/`)
///   2. MOSS codec decoder (HF `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX`)
///   3. preset voices JSON (upstream GitHub, Apache-2.0)
///   4. `sea_g2p.bin` phoneme dictionary, extracted from the pinned sea-g2p
///      wheel on PyPI (the dictionary is platform-independent data)
async fn run_vieneu_download(dir: &PathBuf, progress: Arc<Mutex<DownloadState>>, cancel: CancellationToken) -> anyhow::Result<()> {
    use anyhow::Context;

    const SEA_G2P_VERSION: &str = "0.7.18";
    let vieneu = crate::tts::vieneu::MODEL_ID;
    let codec = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX";
    let voices_url = "https://raw.githubusercontent.com/pnnbao97/VieNeu-TTS/main/src/vieneu/assets/voices_v3_turbo.json";

    let mut files: Vec<(String, PathBuf)> = Vec::new();
    for f in [
        "vieneu_prefill.onnx",
        "vieneu_decode_step.onnx",
        "vieneu_acoustic_cached.onnx",
        "vieneu_backbone_shared.data",
        "vieneu_v3_heads.npz",
        "config.json",
        "tokenizer.json",
    ] {
        files.push((format!("{HF_BASE}/{vieneu}/resolve/main/onnx_int8/{f}"), dir.join("onnx_int8").join(f)));
    }
    for f in ["moss_audio_tokenizer_decode_full.onnx", "moss_audio_tokenizer_decode_shared.data"] {
        files.push((format!("{HF_BASE}/{codec}/resolve/main/{f}"), dir.join("codec").join(f)));
    }
    files.push((voices_url.to_string(), dir.join("voices_v3_turbo.json")));

    let client = reqwest::Client::builder().connect_timeout(std::time::Duration::from_secs(30)).build()?;

    progress.lock().unwrap().status = DownloadStatus::Listing;
    // Resolve the sea-g2p wheel URL (any platform wheel carries the same .bin).
    let pypi: Value = client
        .get(format!("https://pypi.org/pypi/sea-g2p/{SEA_G2P_VERSION}/json"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let wheel_url = pypi["urls"]
        .as_array()
        .and_then(|urls| urls.iter().find(|u| u["filename"].as_str().is_some_and(|f| f.ends_with(".whl"))))
        .and_then(|u| u["url"].as_str())
        .context("no sea-g2p wheel found on PyPI")?
        .to_string();

    {
        let mut s = progress.lock().unwrap();
        s.files_total = (files.len() + 1) as u32;
        s.status = DownloadStatus::Downloading;
    }

    for (url, dst) in &files {
        if cancel.is_cancelled() {
            progress.lock().unwrap().status = DownloadStatus::Cancelled;
            return Ok(());
        }
        progress.lock().unwrap().current_file = Some(dst.file_name().unwrap_or_default().to_string_lossy().into_owned());
        if let Some(parent) = dst.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        download_url_streaming(&client, url, dst, &progress, &cancel).await?;
        if cancel.is_cancelled() {
            progress.lock().unwrap().status = DownloadStatus::Cancelled;
            return Ok(());
        }
        progress.lock().unwrap().files_done += 1;
    }

    // sea_g2p.bin: download the wheel to a temp file, extract the dictionary.
    let bin_dst = dir.join("sea_g2p.bin");
    if !bin_dst.exists() {
        progress.lock().unwrap().current_file = Some("sea_g2p.bin (wheel)".into());
        let tmp = dir.join(".sea_g2p.whl.part");
        download_url_streaming(&client, &wheel_url, &tmp, &progress, &cancel).await?;
        if cancel.is_cancelled() {
            let _ = tokio::fs::remove_file(&tmp).await;
            progress.lock().unwrap().status = DownloadStatus::Cancelled;
            return Ok(());
        }
        let bin_dst2 = bin_dst.clone();
        let tmp2 = tmp.clone();
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let f = std::fs::File::open(&tmp2)?;
            let mut zip = zip::ZipArchive::new(f).context("sea-g2p wheel is not a zip")?;
            let mut entry = zip.by_name("sea_g2p/sea_g2p.bin").context("sea_g2p.bin missing from wheel")?;
            let mut out = std::fs::File::create(&bin_dst2)?;
            std::io::copy(&mut entry, &mut out)?;
            Ok(())
        })
        .await??;
        let _ = tokio::fs::remove_file(&tmp).await;
    }
    progress.lock().unwrap().files_done += 1;
    Ok(())
}

/// Stream one URL to a file with resume-by-size skip + progress accounting.
async fn download_url_streaming(
    client: &reqwest::Client,
    url: &str,
    dst: &std::path::Path,
    progress: &Arc<Mutex<DownloadState>>,
    cancel: &CancellationToken,
) -> anyhow::Result<()> {
    let resp = client.get(url).send().await?.error_for_status()?;
    if let (Some(len), Ok(meta)) = (resp.content_length(), std::fs::metadata(dst)) {
        if meta.len() == len {
            progress.lock().unwrap().downloaded_bytes += len;
            return Ok(()); // already complete
        }
    }
    let mut stream = resp.bytes_stream();
    let mut file = tokio::fs::File::create(dst).await?;
    while let Some(chunk) = stream.next().await {
        if cancel.is_cancelled() {
            drop(file);
            let _ = tokio::fs::remove_file(dst).await;
            return Ok(());
        }
        let bytes = chunk?;
        file.write_all(&bytes).await?;
        progress.lock().unwrap().downloaded_bytes += bytes.len() as u64;
    }
    file.flush().await?;
    Ok(())
}

async fn run_tts_download(repo: &str, dir: &PathBuf, progress: Arc<Mutex<DownloadState>>, cancel: CancellationToken) -> anyhow::Result<()> {
    let client = reqwest::Client::builder().connect_timeout(std::time::Duration::from_secs(30)).build()?;

    progress.lock().unwrap().status = DownloadStatus::Listing;

    let tree_url = format!("{HF_BASE}/api/models/{repo}/tree/main?recursive=true");
    let tree: Vec<HfTreeEntry> = client.get(&tree_url).send().await?.error_for_status()?.json().await?;

    let files: Vec<(String, u64)> =
        tree.into_iter().filter(|e| e.entry_type == "file" && !should_skip(&e.path)).map(|e| (e.path, e.size)).collect();

    {
        let mut s = progress.lock().unwrap();
        s.files_total = files.len() as u32;
        s.total_bytes = files.iter().map(|f| f.1).sum();
        s.status = DownloadStatus::Downloading;
    }

    for (path, size) in files {
        if cancel.is_cancelled() {
            progress.lock().unwrap().status = DownloadStatus::Cancelled;
            return Ok(());
        }
        progress.lock().unwrap().current_file = Some(path.clone());

        let dst = dir.join(&path);
        if let Some(parent) = dst.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        if size > 0 {
            if let Ok(meta) = tokio::fs::metadata(&dst).await {
                if meta.len() == size {
                    let mut s = progress.lock().unwrap();
                    s.files_done += 1;
                    s.downloaded_bytes += size;
                    continue;
                }
            }
        }

        let url = format!("{HF_BASE}/{repo}/resolve/main/{path}");
        let resp = client.get(&url).send().await?.error_for_status()?;
        let mut stream = resp.bytes_stream();
        let mut file = tokio::fs::File::create(&dst).await?;

        while let Some(chunk) = stream.next().await {
            if cancel.is_cancelled() {
                drop(file);
                let _ = tokio::fs::remove_file(&dst).await;
                progress.lock().unwrap().status = DownloadStatus::Cancelled;
                return Ok(());
            }
            let bytes = chunk?;
            file.write_all(&bytes).await?;
            progress.lock().unwrap().downloaded_bytes += bytes.len() as u64;
        }
        file.flush().await?;
        progress.lock().unwrap().files_done += 1;
    }

    Ok(())
}

/// The full router this runtime serves.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/tts/models", get(tts_models_list))
        .route("/api/tts/models/:id/download", axum::routing::post(tts_download))
        .route("/api/tts/models/:id/validate", get(tts_validate))
        .route("/api/tts/models/:id/status", get(tts_status))
        .route("/api/tts/models/:id/cancel", axum::routing::post(tts_cancel))
        .route("/api/tts/models/:id", axum::routing::delete(tts_delete))
        .route("/api/tts/settings", get(tts_settings_get).put(tts_settings_put))
        .route("/api/tts/synthesize", axum::routing::post(tts_synthesize))
        .route("/v1/audio/speech", axum::routing::post(audio_speech))
        .with_state(state)
}

#[cfg(test)]
mod synth_tests {
    use super::*;

    fn state_at(dir: &std::path::Path) -> Arc<AppState> {
        let env = LaunchEnv::from_lookup("sen-tts", "0.0.0-test", |k| match k {
            "SENCLAW_RUNTIME_DATA_DIR" => Some(dir.join("data").to_string_lossy().into_owned()),
            "SENCLAW_TTS_MODELS_DIR" => Some(dir.join("tts-models").to_string_lossy().into_owned()),
            "SENCLAW_CONFIG_PATH" => Some(dir.join("config.json").to_string_lossy().into_owned()),
            "SENCLAW_HOME" => Some(dir.to_string_lossy().into_owned()),
            _ => None,
        });
        Arc::new(AppState { env })
    }

    fn looks_like_wav(bytes: &[u8]) -> bool {
        bytes.len() > 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE"
    }

    #[tokio::test]
    async fn models_and_settings_answer_with_the_old_shapes() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let tmp = tempfile::tempdir().unwrap();
        let app = router(state_at(tmp.path()));

        let resp = app.clone().oneshot(Request::get("/api/tts/models").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        let ids: Vec<&str> = v["models"].as_array().unwrap().iter().map(|m| m["id"].as_str().unwrap()).collect();
        assert!(ids.contains(&"macos-speech"));
        assert!(ids.contains(&"macos-speech-en"));

        let resp = app.oneshot(Request::get("/api/tts/settings").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["model_id"], "macos-speech");
        assert_eq!(v["language"], "vi");
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn synthesize_route_speaks_macos_speech() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let tmp = tempfile::tempdir().unwrap();
        let app = router(state_at(tmp.path()));
        let resp = app
            .oneshot(
                Request::post("/api/tts/synthesize")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"text": "Xin chào.", "model_id": "macos-speech"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers().get("x-tts-backend").unwrap(), "macos-speech");
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        assert!(looks_like_wav(&body));
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn openai_audio_speech_route_speaks_macos_speech() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let tmp = tempfile::tempdir().unwrap();
        let app = router(state_at(tmp.path()));
        let resp = app
            .oneshot(
                Request::post("/v1/audio/speech")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model": "macos-speech-en", "input": "Hello there.", "response_format": "wav"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers().get("x-tts-backend").unwrap(), "macos-speech-en");
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        assert!(looks_like_wav(&body));
    }

    #[tokio::test]
    async fn openai_route_rejects_a_non_wav_format() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let tmp = tempfile::tempdir().unwrap();
        let app = router(state_at(tmp.path()));
        let resp = app
            .oneshot(
                Request::post("/v1/audio/speech")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"input": "hi", "response_format": "mp3"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// A removed/unknown voice must degrade to macOS speech, never 400 — the
    /// contract the daemon relied on for stale `facebook/mms-tts-vie` configs.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn an_unsupported_model_falls_back_with_the_header_set() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let tmp = tempfile::tempdir().unwrap();
        // `is_installed` only special-cases macos-speech* and VieNeu; anything
        // else needs a real directory with weights to pass the install check —
        // simulate one so the request reaches the `Unsupported` backend.
        let dir = tmp.path().join("tts-models").join("facebook__mms-tts-vie");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        std::fs::write(dir.join("model.safetensors"), []).unwrap();

        let app = router(state_at(tmp.path()));
        let resp = app
            .oneshot(
                Request::post("/api/tts/synthesize")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"text": "Xin chào.", "model_id": "facebook/mms-tts-vie", "language": "vi"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "a removed voice must degrade, never 400");
        assert_eq!(resp.headers().get("x-tts-backend").unwrap(), "macos-speech");
        assert!(resp.headers().get("x-tts-fallback").is_some());
    }

    #[test]
    fn normalize_hf_id_accepts_urls_and_rejects_garbage() {
        assert_eq!(normalize_hf_id("https://huggingface.co/facebook/mms-tts-vie/").unwrap(), "facebook/mms-tts-vie");
        assert_eq!(normalize_hf_id("facebook/mms-tts-vie").unwrap(), "facebook/mms-tts-vie");
        assert!(normalize_hf_id("nonsense").is_err());
        assert!(normalize_hf_id("a/../b").is_err());
    }

    #[test]
    fn hf_validate_accepts_vieneu_and_vits_finetune_wrapper() {
        let vieneu = json!({"architectures": ["VieNeuModel"]});
        let (ok, reason, _) = check_tts(&vieneu, false, &|_| false);
        assert!(ok, "{reason}");

        let cfg = json!({"architectures": ["VitsModelForPreTraining"], "num_speakers": 1, "speaker_embedding_size": 0});
        let (ok, reason, arch) = check_tts(&cfg, true, &|f| f == "vocab.json");
        assert!(ok, "{reason}");
        assert_eq!(arch.as_deref(), Some("VitsModelForPreTraining"));
    }

    #[test]
    fn hf_validate_rejects_multispeaker_and_foreign_arch() {
        let multi = json!({"architectures": ["VitsModel"], "num_speakers": 4});
        let (ok, reason, _) = check_tts(&multi, true, &|_| true);
        assert!(!ok && reason.contains("multi-speaker"), "{reason}");

        let xtts = json!({"architectures": ["XttsModel"]});
        let (ok, reason, _) = check_tts(&xtts, true, &|_| true);
        assert!(!ok && reason.contains("XttsModel"), "{reason}");
    }
}
