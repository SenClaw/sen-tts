//! VieNeu-TTS v3 Turbo inference engine — ONNX Runtime (CPU), torch-free.
//!
//! Rust mirror of `vieneu/_v3_turbo_engine/onnx_runtime_lite.py`
//! (`OnnxV3LiteEngine`): the transformer forwards and the MOSS codec run in
//! ONNX Runtime; embeddings, the speaker anchor, output heads, sampling and the
//! prompt build are plain Rust over the tied tables in `vieneu_v3_heads.npz`.
//!
//! Per synthesized frame:
//!   1. `vieneu_acoustic_cached.onnx` runs 1 + (n_vq−1) cached steps to sample
//!      the 16 RVQ codebooks (logits = hidden · audio_embᵀ per channel);
//!   2. EOS is `argmax(slot0 · text_embᵀ) == speech_generation_end`;
//!   3. `vieneu_decode_step.onnx` advances the 12-layer backbone KV cache.
//! Frames then decode to a 48 kHz waveform via
//! `moss_audio_tokenizer_decode_full.onnx` (stereo → mono mean).

use std::collections::HashSet;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{DynValue, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::npz::{load_npz, NpyArray};
use super::phonemize::phonemize_with_emotions;
use super::sea_g2p::SeaPipeline;
use super::voices::{Preset, Voices};

/// Generation defaults — the upstream engine's documented sweet spot.
pub const TEMPERATURE: f32 = 0.8;
pub const TOP_K: usize = 25;
pub const TOP_P: f32 = 0.95;
pub const REPETITION_PENALTY: f32 = 1.2;
pub const MAX_NEW_FRAMES: usize = 300;

pub const SAMPLE_RATE: u32 = 48_000;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct EngineCfg {
    pub n_vq: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    #[serde(default = "one")]
    pub local_num_hidden_layers: usize,
    #[serde(default = "eight")]
    pub local_num_attention_heads: usize,
    pub audio_pad_token_id: i64,
    pub text_prompt_start_token_id: i64,
    pub text_prompt_end_token_id: i64,
    pub speech_generation_start_token_id: i64,
    pub speech_generation_end_token_id: i64,
    pub audio_ref_slot_token_id: i64,
    #[serde(default)]
    pub use_speaker_embedding: bool,
    #[serde(default = "sixteen")]
    pub default_style_token_id: i64,
    #[serde(default)]
    pub style_labels: std::collections::HashMap<String, i64>,
}
fn one() -> usize {
    1
}
fn eight() -> usize {
    8
}
fn sixteen() -> i64 {
    16
}

struct Xvec {
    w: NpyArray,    // (H, spk_dim)
    b: NpyArray,    // (H,)
    ln_w: NpyArray, // (H,)
    ln_b: NpyArray, // (H,)
    eps: f32,
}

/// Backbone KV state: `past[i]` = layer-i key, `past[L+i]` = layer-i value.
type Past = Vec<DynValue>;

/// A session plus its ordered output names (ort removes outputs by NAME; the
/// graph's positional order is [hidden, present_k_0.., present_v_0..]).
struct Sess {
    sess: Session,
    out_names: Vec<String>,
}

impl Sess {
    fn load(p: &Path, threads: usize) -> Result<Self> {
        // ort's builder errors carry the builder itself (not Send+Sync), so
        // they can't ride `?` into anyhow — stringify at each step.
        //
        // Memory: the CPU arena + memory-pattern are DISABLED. The generation
        // loop churns thousands of variable-shaped tensors (24 fresh KV
        // presents per frame × hundreds of frames); ORT's arena grows to the
        // high-water mark and never returns it to the OS — observed multi-GB
        // daemon RSS after one long synthesis. With the arena off, run
        // buffers are plain mallocs and RSS falls back after each request.
        let cpu_no_arena = ort::execution_providers::CPUExecutionProvider::default()
            .with_arena_allocator(false)
            .build();
        let sess = Session::builder()
            .map_err(|e| anyhow!("onnx session builder: {e}"))?
            .with_execution_providers([cpu_no_arena])
            .map_err(|e| anyhow!("onnx session builder: {e}"))?
            .with_memory_pattern(false)
            .map_err(|e| anyhow!("onnx session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("onnx session builder: {e}"))?
            .with_intra_threads(threads)
            .map_err(|e| anyhow!("onnx session builder: {e}"))?
            .with_inter_threads(1)
            .map_err(|e| anyhow!("onnx session builder: {e}"))?
            .commit_from_file(p)
            .map_err(|e| anyhow!("onnx load: {e}"))
            .with_context(|| format!("loading onnx {}", p.display()))?;
        let out_names = sess
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        Ok(Self { sess, out_names })
    }
}

/// Run a KV-cached graph: feed `owned` inputs plus `past_k_i`/`past_v_i` as
/// borrowed views, return (flat copy of output 0, new past moved out of the
/// outputs). All session/output borrows end inside this function.
fn run_kv(
    s: &mut Sess,
    owned: Vec<(String, DynValue)>,
    past: &[DynValue],
    n_layers: usize,
) -> Result<(Vec<f32>, Past)> {
    use ort::session::SessionInputValue;
    let mut inputs: Vec<(std::borrow::Cow<'static, str>, SessionInputValue<'_>)> =
        Vec::with_capacity(owned.len() + past.len());
    for (k, v) in owned {
        inputs.push((k.into(), v.into()));
    }
    // Prefill graphs take no past inputs — callers pass an empty slice there
    // while still asking for `n_layers` presents back.
    if !past.is_empty() {
        for i in 0..n_layers {
            inputs.push((format!("past_k_{i}").into(), (&past[i]).into()));
            inputs.push((format!("past_v_{i}").into(), (&past[n_layers + i]).into()));
        }
    }
    let mut outputs = s.sess.run(inputs)?;
    let (_, h) = outputs[0].try_extract_tensor::<f32>()?;
    let hidden = h.to_vec();
    let mut new_past = Past::with_capacity(2 * n_layers);
    for name in s.out_names.iter().skip(1).take(2 * n_layers) {
        new_past.push(
            outputs
                .remove(name.as_str())
                .ok_or_else(|| anyhow!("onnx output `{name}` missing"))?,
        );
    }
    Ok((hidden, new_past))
}

pub struct VieNeuEngine {
    pub cfg: EngineCfg,
    tokenizer: tokenizers::Tokenizer,
    text_emb: NpyArray,       // (Vt, H)
    audio_emb: Vec<NpyArray>, // n_vq × (Va, H)
    xvec: Option<Xvec>,
    sess_pre: Sess,
    sess_dec: Sess,
    sess_ac: Sess,
    sess_codec: Sess,
    pipeline: SeaPipeline,
    pub voices: Voices,
}

impl VieNeuEngine {
    /// Load everything from the model directory laid out by our downloader:
    /// `onnx_int8/…`, `codec/…`, `voices_v3_turbo.json`, `sea_g2p.bin`.
    pub fn load(dir: &Path) -> Result<Self> {
        let od = dir.join("onnx_int8");
        let cfg: EngineCfg = serde_json::from_str(
            &std::fs::read_to_string(od.join("config.json")).context("onnx config.json")?,
        )
        .context("parsing onnx config.json")?;

        let mut heads = load_npz(&od.join("vieneu_v3_heads.npz")).context("heads npz")?;
        let text_emb = heads
            .remove("text_emb")
            .ok_or_else(|| anyhow!("heads.npz: no text_emb"))?;
        let audio_all = heads
            .remove("audio_emb")
            .ok_or_else(|| anyhow!("heads.npz: no audio_emb"))?;
        if audio_all.shape.len() != 3 || audio_all.shape[0] != cfg.n_vq {
            bail!(
                "heads.npz audio_emb shape {:?} != (n_vq, Va, H)",
                audio_all.shape
            );
        }
        let (va, h) = (audio_all.shape[1], audio_all.shape[2]);
        let audio_emb: Vec<NpyArray> = (0..cfg.n_vq)
            .map(|ch| NpyArray {
                shape: vec![va, h],
                data: audio_all.data[ch * va * h..(ch + 1) * va * h].to_vec(),
            })
            .collect();
        let xvec = if cfg.use_speaker_embedding {
            Some(Xvec {
                w: heads
                    .remove("xvec_w")
                    .ok_or_else(|| anyhow!("heads.npz: no xvec_w"))?,
                b: heads
                    .remove("xvec_b")
                    .ok_or_else(|| anyhow!("heads.npz: no xvec_b"))?,
                ln_w: heads
                    .remove("xvec_ln_w")
                    .ok_or_else(|| anyhow!("heads.npz: no xvec_ln_w"))?,
                ln_b: heads
                    .remove("xvec_ln_b")
                    .ok_or_else(|| anyhow!("heads.npz: no xvec_ln_b"))?,
                eps: heads
                    .remove("xvec_ln_eps")
                    .map(|a| a.scalar())
                    .unwrap_or(1e-5),
            })
        } else {
            None
        };

        let tokenizer = tokenizers::Tokenizer::from_file(od.join("tokenizer.json"))
            .map_err(|e| anyhow!("tokenizer.json: {e}"))?;

        let threads = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).clamp(1, 8))
            .unwrap_or(4);
        let sess_pre = Sess::load(&od.join("vieneu_prefill.onnx"), threads)?;
        let sess_dec = Sess::load(&od.join("vieneu_decode_step.onnx"), threads)?;
        let sess_ac = Sess::load(&od.join("vieneu_acoustic_cached.onnx"), threads)?;
        let sess_codec = Sess::load(
            &dir.join("codec/moss_audio_tokenizer_decode_full.onnx"),
            threads,
        )?;

        let pipeline = SeaPipeline::new(
            dir.join("sea_g2p.bin")
                .to_str()
                .ok_or_else(|| anyhow!("bad model dir path"))?,
        )
        .context("loading sea_g2p.bin")?;
        let voices = Voices::load(&dir.join("voices_v3_turbo.json"))?;

        Ok(Self {
            cfg,
            tokenizer,
            text_emb,
            audio_emb,
            xvec,
            sess_pre,
            sess_dec,
            sess_ac,
            sess_codec,
            pipeline,
            voices,
        })
    }

    // ── plain-Rust pieces (mirror the numpy side) ────────────────────────────

    /// 192-d x-vector → (H,) anchor (Linear + LayerNorm, mirror `_speaker_anchor`).
    fn speaker_anchor(&self, speaker_emb: &[f32]) -> Result<Option<Vec<f32>>> {
        let Some(x) = &self.xvec else { return Ok(None) };
        if speaker_emb.iter().all(|v| *v == 0.0) {
            bail!("speaker_emb is all-zero — not a valid speaker anchor");
        }
        let h = self.cfg.hidden_size;
        let spk = x.w.shape[1];
        if speaker_emb.len() != spk {
            bail!("speaker_emb dim {} != expected {spk}", speaker_emb.len());
        }
        let mut v = vec![0.0f32; h];
        for (i, out) in v.iter_mut().enumerate() {
            let row = &x.w.data[i * spk..(i + 1) * spk];
            *out = row.iter().zip(speaker_emb).map(|(a, b)| a * b).sum::<f32>() + x.b.data[i];
        }
        let mean = v.iter().sum::<f32>() / h as f32;
        let var = v.iter().map(|a| (a - mean) * (a - mean)).sum::<f32>() / h as f32;
        let inv = 1.0 / (var + x.eps).sqrt();
        for (i, a) in v.iter_mut().enumerate() {
            *a = (*a - mean) * inv * x.ln_w.data[i] + x.ln_b.data[i];
        }
        Ok(Some(v))
    }

    /// rows (T, n_vq+1) → flat (T·H) embeddings (mirror `_embed_rows`).
    fn embed_rows(&self, rows: &[Vec<i64>], anchor: Option<&[f32]>) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let mut out = vec![0.0f32; rows.len() * h];
        for (t, row) in rows.iter().enumerate() {
            let dst = &mut out[t * h..(t + 1) * h];
            dst.copy_from_slice(self.text_emb.row(row[0] as usize));
            for ch in 0..self.cfg.n_vq {
                let id = row[ch + 1];
                if id != self.cfg.audio_pad_token_id {
                    for (d, s) in dst.iter_mut().zip(self.audio_emb[ch].row(id as usize)) {
                        *d += *s;
                    }
                }
            }
            if let Some(a) = anchor {
                for (d, s) in dst.iter_mut().zip(a) {
                    *d += *s;
                }
            }
        }
        out
    }

    /// Prompt rows `[style, tps, …phones…, tpe]` (+ ref rows), mirror `_build_rows`.
    fn build_rows(
        &self,
        phonemes: &str,
        ref_codes: &[Vec<i64>],
        style_id: i64,
    ) -> Result<Vec<Vec<i64>>> {
        let enc = self
            .tokenizer
            .encode(phonemes, false)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        let pad = self.cfg.audio_pad_token_id;
        let width = self.cfg.n_vq + 1;
        let mut rows: Vec<Vec<i64>> = Vec::new();
        let mut text_ids: Vec<i64> = vec![style_id, self.cfg.text_prompt_start_token_id];
        text_ids.extend(enc.get_ids().iter().map(|&u| u as i64));
        text_ids.push(self.cfg.text_prompt_end_token_id);
        for id in text_ids {
            let mut row = vec![pad; width];
            row[0] = id;
            rows.push(row);
        }
        for rc in ref_codes {
            if rc.len() != self.cfg.n_vq {
                bail!("ref codes row width {} != n_vq {}", rc.len(), self.cfg.n_vq);
            }
            let mut row = vec![pad; width];
            row[0] = self.cfg.audio_ref_slot_token_id;
            row[1..].copy_from_slice(rc);
            rows.push(row);
        }
        Ok(rows)
    }

    /// logits = vec · audio_emb[ch]ᵀ, then sample (mirror the `samp` closure).
    fn sample_channel(
        &self,
        ch: usize,
        vec: &[f32],
        hist: &mut [HashSet<i64>],
        rng: &mut StdRng,
    ) -> i64 {
        let va = self.audio_emb[ch].shape[0];
        let mut logits = vec![0.0f32; va];
        for (v, l) in logits.iter_mut().enumerate() {
            *l = self.audio_emb[ch]
                .row(v)
                .iter()
                .zip(vec)
                .map(|(a, b)| a * b)
                .sum();
        }
        let code = sample(
            &mut logits,
            TEMPERATURE,
            TOP_K,
            TOP_P,
            REPETITION_PENALTY,
            Some(&hist[ch]),
            rng,
        );
        hist[ch].insert(code);
        code
    }

    /// Fresh empty local-decoder past (S=0 KV tensors).
    fn empty_local_past(&self) -> Result<Past> {
        let hd = self.cfg.hidden_size;
        let nh = self.cfg.local_num_attention_heads;
        let head_dim = hd / nh;
        let n = self.cfg.local_num_hidden_layers;
        // `Tensor::from_array` rejects zero-size dims; the allocator path
        // creates a valid empty (S=0) KV tensor like numpy's zeros((1,h,0,d)).
        let alloc = ort::memory::Allocator::default();
        let mut p = Past::with_capacity(2 * n);
        for _ in 0..2 * n {
            p.push(
                Tensor::<f32>::new(&alloc, [1usize, nh, 0, head_dim])
                    .map_err(|e| anyhow!("empty past tensor: {e}"))?
                    .into_dyn(),
            );
        }
        Ok(p)
    }

    /// One full frame: acoustic 2-token prefill + (n_vq−1) cached steps.
    /// Returns the n_vq codes + EOS flag (mirror `_acoustic_frame`).
    fn frame(
        &mut self,
        h: &[f32],
        hist: &mut [HashSet<i64>],
        rng: &mut StdRng,
    ) -> Result<(Vec<i64>, bool)> {
        let hd = self.cfg.hidden_size;
        let l_loc = self.cfg.local_num_hidden_layers;

        // 2-token prefill: [cond=h, text_emb[sgs]], positions [0, 1].
        let mut tok = Vec::with_capacity(2 * hd);
        tok.extend_from_slice(h);
        tok.extend_from_slice(
            self.text_emb
                .row(self.cfg.speech_generation_start_token_id as usize),
        );
        let owned: Vec<(String, DynValue)> = vec![
            (
                "token_emb".into(),
                Tensor::from_array(([1usize, 2, hd], tok))?.into_dyn(),
            ),
            (
                "position_ids".into(),
                Tensor::from_array(([1usize, 2], vec![0i64, 1]))?.into_dyn(),
            ),
        ];
        let empty = self.empty_local_past()?;
        let (hidden, mut past) = run_kv(&mut self.sess_ac, owned, &empty, l_loc)?;
        let slot0 = hidden[..hd].to_vec();
        let cond = hidden[hd..2 * hd].to_vec();

        // EOS via text head on slot0.
        let vt = self.text_emb.shape[0];
        let mut best = (f32::NEG_INFINITY, 0usize);
        for v in 0..vt {
            let s: f32 = self
                .text_emb
                .row(v)
                .iter()
                .zip(&slot0)
                .map(|(a, b)| a * b)
                .sum();
            if s > best.0 {
                best = (s, v);
            }
        }
        let eos = best.1 as i64 == self.cfg.speech_generation_end_token_id;

        // Channel 0 from the cond position; channels 1.. via cached steps.
        let mut codes = Vec::with_capacity(self.cfg.n_vq);
        codes.push(self.sample_channel(0, &cond, hist, rng));
        for ch in 1..self.cfg.n_vq {
            let emb = self.audio_emb[ch - 1].row(codes[ch - 1] as usize).to_vec();
            let owned: Vec<(String, DynValue)> = vec![
                (
                    "token_emb".into(),
                    Tensor::from_array(([1usize, 1, hd], emb))?.into_dyn(),
                ),
                (
                    "position_ids".into(),
                    Tensor::from_array(([1usize, 1], vec![(ch + 1) as i64]))?.into_dyn(),
                ),
            ];
            let (hidden, new_past) = run_kv(&mut self.sess_ac, owned, &past, l_loc)?;
            past = new_past;
            codes.push(self.sample_channel(ch, &hidden[..hd], hist, rng));
        }
        Ok((codes, eos))
    }

    /// Synthesize one text chunk with a preset voice → mono f32 @ 48 kHz.
    pub fn infer_text(&mut self, text: &str, preset: &Preset) -> Result<Vec<f32>> {
        let phonemes = phonemize_with_emotions(&self.pipeline, text);
        if phonemes.trim().is_empty() {
            return Ok(Vec::new());
        }
        self.infer_phonemes(&phonemes, preset)
    }

    pub fn infer_phonemes(&mut self, phonemes: &str, preset: &Preset) -> Result<Vec<f32>> {
        let anchor = self.speaker_anchor(&preset.speaker_emb)?;
        let ref_codes: Vec<Vec<i64>> = preset.codes.clone();
        let style_id = preset
            .style
            .as_deref()
            .and_then(|s| self.cfg.style_labels.get(s).copied())
            .unwrap_or(self.cfg.default_style_token_id);
        let rows = self.build_rows(phonemes, &ref_codes, style_id)?;
        let t_prompt = rows.len();
        let h = self.cfg.hidden_size;
        let l = self.cfg.num_hidden_layers;
        let embeds = self.embed_rows(&rows, anchor.as_deref());
        let mut rng = StdRng::from_entropy();

        // Backbone prefill (no past inputs; presents still come back).
        let owned: Vec<(String, DynValue)> = vec![(
            "inputs_embeds".to_string(),
            Tensor::from_array(([1usize, t_prompt, h], embeds))?.into_dyn(),
        )];
        let (hidden, mut past) = run_kv(&mut self.sess_pre, owned, &[], l)?;
        let mut hvec = hidden[(t_prompt - 1) * h..t_prompt * h].to_vec();

        let mut hist: Vec<HashSet<i64>> = (0..self.cfg.n_vq).map(|_| HashSet::new()).collect();
        let mut frames: Vec<Vec<i64>> = Vec::new();
        for t in 0..MAX_NEW_FRAMES {
            let (codes, eos) = self.frame(&hvec, &mut hist, &mut rng)?;
            frames.push(codes.clone());
            if eos {
                break;
            }
            // Advance the backbone with the generated slot row.
            let mut slot = vec![vec![self.cfg.audio_pad_token_id; self.cfg.n_vq + 1]];
            slot[0][0] = self.cfg.speech_generation_start_token_id;
            slot[0][1..].copy_from_slice(&codes);
            let se = self.embed_rows(&slot, anchor.as_deref());
            let owned: Vec<(String, DynValue)> = vec![
                (
                    "inputs_embeds".into(),
                    Tensor::from_array(([1usize, 1, h], se))?.into_dyn(),
                ),
                (
                    "position_ids".into(),
                    Tensor::from_array(([1usize, 1], vec![(t_prompt + t) as i64]))?.into_dyn(),
                ),
            ];
            let (oh, new_past) = run_kv(&mut self.sess_dec, owned, &past, l)?;
            past = new_past;
            hvec.copy_from_slice(&oh[..h]);
        }
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        self.decode_codes(&frames)
    }

    /// (T, n_vq) frames → mono waveform via the MOSS full decoder.
    fn decode_codes(&mut self, frames: &[Vec<i64>]) -> Result<Vec<f32>> {
        let t = frames.len();
        let nq = self.cfg.n_vq;
        let mut flat = Vec::with_capacity(t * nq);
        for f in frames {
            flat.extend(f.iter().map(|&c| c as i32));
        }
        let outputs = self.sess_codec.sess.run(vec![
            (
                "audio_codes".to_string(),
                ort::session::SessionInputValue::from(
                    Tensor::from_array(([1usize, t, nq], flat))?.into_dyn(),
                ),
            ),
            (
                "audio_code_lengths".to_string(),
                ort::session::SessionInputValue::from(
                    Tensor::from_array(([1usize], vec![t as i32]))?.into_dyn(),
                ),
            ),
        ])?;
        let (shape, audio) = outputs[0].try_extract_tensor::<f32>()?;
        // (1, C, n) → mono mean over channels.
        if shape.len() != 3 {
            bail!("codec output shape {shape:?} unexpected");
        }
        let c = shape[1] as usize;
        let n = shape[2] as usize;
        let mut mono = vec![0.0f32; n];
        for ch in 0..c {
            for (i, m) in mono.iter_mut().enumerate() {
                *m += audio[ch * n + i];
            }
        }
        let inv = 1.0 / c as f32;
        for m in mono.iter_mut() {
            *m *= inv;
        }
        Ok(mono)
    }
}

/// Mirror `_sample`: repetition penalty → temperature → top-k → top-p → choice.
fn sample(
    logits: &mut [f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rep_pen: f32,
    prev: Option<&HashSet<i64>>,
    rng: &mut StdRng,
) -> i64 {
    if (rep_pen - 1.0).abs() > 1e-6 {
        if let Some(prev) = prev {
            for &id in prev {
                let l = &mut logits[id as usize];
                *l = if *l < 0.0 { *l * rep_pen } else { *l / rep_pen };
            }
        }
    }
    if temperature <= 0.0 {
        return argmax(logits) as i64;
    }
    for l in logits.iter_mut() {
        *l /= temperature;
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    let k = if top_k > 0 && top_k < logits.len() {
        top_k
    } else {
        logits.len()
    };
    if k < logits.len() {
        idx.select_nth_unstable_by(k - 1, |&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(k);
    }
    idx.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let m = logits[idx[0]];
    let mut probs: Vec<f32> = idx.iter().map(|&i| (logits[i] - m).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    // nucleus: keep entries whose cumulative mass BEFORE them is < top_p
    if top_p < 1.0 {
        let mut cum = 0.0f32;
        for p in probs.iter_mut() {
            let before = cum;
            cum += *p;
            if before >= top_p {
                *p = 0.0;
            }
        }
        let s: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= s;
        }
    }
    let r: f32 = rng.gen();
    let mut cum = 0.0;
    for (i, p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return idx[i] as i64;
        }
    }
    *idx.last().expect("non-empty candidates") as i64
}

fn argmax(v: &[f32]) -> usize {
    let mut best = (f32::NEG_INFINITY, 0);
    for (i, &x) in v.iter().enumerate() {
        if x > best.0 {
            best = (x, i);
        }
    }
    best.1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_greedy_at_zero_temperature() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut logits = vec![0.1, 3.0, -1.0, 2.9];
        assert_eq!(sample(&mut logits, 0.0, 25, 0.95, 1.0, None, &mut rng), 1);
    }

    #[test]
    fn sample_respects_top_k_and_penalty() {
        let mut rng = StdRng::seed_from_u64(7);
        // With top_k=1, sampling is deterministic argmax regardless of r.
        let mut logits = vec![0.0, 5.0, 1.0];
        assert_eq!(sample(&mut logits, 0.8, 1, 1.0, 1.0, None, &mut rng), 1);
        // A strong repetition penalty on id 1 flips the winner at top_k=1.
        let mut prev = HashSet::new();
        prev.insert(1i64);
        let mut logits = vec![4.9, 5.0, 1.0];
        assert_eq!(
            sample(&mut logits, 0.8, 1, 1.0, 10.0, Some(&prev), &mut rng),
            0
        );
    }
}
