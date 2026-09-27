# sen-tts

SenClaw's text-to-speech runtime: **VieNeu-TTS v3 Turbo** (ONNX Runtime, CPU
— 48 kHz, 14 Vietnamese presets, En–Vi code-switching) and native macOS
`say` presets. No Python, no MLX. Launched by the SenClaw daemon as a child
process and driven over loopback HTTP — see
[`senclaw/docs/runtime-protocol.md`](../senclaw/docs/runtime-protocol.md)
§4.6 for the wire contract this repo implements.

Ported from the SenClaw daemon's `src/tts/**`, `src/gateway/ui_server/tts.rs`,
the TTS half of `ui_server/hf_validate.rs`, and the `TtsSettings` shape from
`gateway/group_manager/{types,llm}.rs`.

## Build & run

```bash
cargo build --release            # or: make build
cargo test                       # or: make test
make package                     # dist/sen-tts-<version>-<platform>.tar.gz + .sha256
make install-local                # senclaw runtime install-local, or extract into ~/.senclaw/runtimes
make run-dev                     # standalone serve on :4964, no token, no watchdog
```

Standalone (no daemon) for development:

```bash
cargo run -- serve --host 127.0.0.1 --port 4964
```

With no `SENCLAW_RUNTIME_TOKEN` set there is no auth; with no
`SENCLAW_PARENT_PID` there is no parent watchdog — see
[`sen_runtime_sdk::env`](../senclaw/crates/sen-runtime-sdk/src/env.rs).

## Routes

Every route the daemon proxied at `/api/tts/*` before the split, served
**verbatim** (same paths, bodies, status codes), plus the common
`/health` · `/runtime/info` · `/runtime/shutdown` from the SDK server
scaffold and OpenAI-compatible `POST /v1/audio/speech`:

```
GET    /api/tts/models
POST   /api/tts/models/:id/download
GET    /api/tts/models/:id/validate
GET    /api/tts/models/:id/status
POST   /api/tts/models/:id/cancel
DELETE /api/tts/models/:id
GET    /api/tts/settings
PUT    /api/tts/settings
POST   /api/tts/synthesize
POST   /v1/audio/speech
```

A removed or unrecognized voice **degrades to the macOS preset** with
`fallback_reason` / `X-TTS-Fallback` — never a 400 — so a stale config
(e.g. an old `facebook/mms-tts-vie` selection) keeps working.

## Models on disk

`SENCLAW_TTS_MODELS_DIR`, default `<SENCLAW_HOME>/tts-models/<safe-id>/`.
VieNeu's composite download layout is unchanged:

```
onnx_int8/            prefill / decode_step / acoustic graphs + heads + tokenizer
codec/                MOSS-Audio-Tokenizer-Nano ONNX
voices_v3_turbo.json  14 preset voices
sea_g2p.bin           phoneme dictionary
```

Settings persist at `<SENCLAW_RUNTIME_DATA_DIR>/settings.json`, seeded once
from the daemon's old `config.json["ttsConfig"]`.

## Feature flags

- `tts-vieneu` (default on) — the ONNX engine (`ort` + `tokenizers` +
  `fancy-regex` + `memmap2`). Off, the binary still serves the whole API and
  the macOS presets; VieNeu answers `NotImplemented`, which
  `synthesize_with_fallback` turns into the macOS voice.
