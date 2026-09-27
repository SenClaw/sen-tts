# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

`sen-tts` is the SenClaw text-to-speech runtime — VieNeu-TTS v3 Turbo (ONNX
Runtime, CPU) plus native macOS `say` presets, ported unchanged from the
daemon's `src/tts/**` + `src/gateway/ui_server/tts.rs`. No Python, no MLX. It
is a standalone binary the SenClaw daemon installs, launches as a child
process, and talks to over loopback HTTP: see
`../senclaw/docs/runtime-protocol.md` §4.6 for the exact contract (routes,
bodies, status codes) this repo must keep serving **verbatim**.

## Rules for Claude

- **A removed or unrecognized voice must degrade, not 400.**
  `select_backend_for`'s catch-all (`src/tts/mod.rs`) is an `Unsupported`
  backend returning `NotImplemented`, which `synthesize_with_fallback` turns
  into the macOS voice plus a `fallback_reason` (`X-TTS-Fallback` header).
  Machines can still have `facebook/mms-tts-vie` selected in a config carried
  over from before the MLX voices were removed; `None` there would hard-fail
  the next play button instead of falling back. A test in `src/http.rs`
  (`an_unsupported_model_falls_back_with_the_header_set`) pins this.
- **`/api/tts/synthesize` and `/v1/audio/speech` share one code path**
  (`http::synthesize_response`) — the wire shape differs (legacy body vs
  OpenAI's `{model?, input, voice?, response_format, speed?}`), the
  fallback/header behaviour must not. Do not fork them.
- **The VieNeu engine session runs with ORT's arena allocator and memory
  pattern OFF** (`tts/vieneu/engine.rs::Sess::load`). The generation loop
  churns thousands of variable-shaped KV tensors per synthesis; with the
  arena on, ORT's high-water mark never returns to the OS and a long session
  accumulates multi-GB RSS. With it off, buffers are plain mallocs that
  actually free.
- **`tts/vieneu/mod.rs`'s idle cache purges malloc's magazines after every
  chunk and every request** (`malloc_zone_pressure_relief` on macOS) — the
  arena-off tradeoff above means freed pages sit in malloc's own pools
  otherwise. The cached engine itself drops after 60 s idle
  (`IDLE_TTL`), deliberately short: reloading costs ~1 s, and a live engine
  pins several hundred MB regardless.
- **Text is chunked before synthesis** (`tts::chunk::chunk_text`, cap 220
  chars for VieNeu — `max_new_frames=300` ≈ 24 s of audio per chunk) with a
  150 ms silence gap stitched between chunks. Never feed VieNeu a whole long
  reply unchunked.
- **`tts-vieneu` is a default-on feature, not an always-on dependency.** A
  build without it still serves the whole API — VieNeu answers
  `NotImplemented` (which degrades to macOS speech) instead of failing to
  compile — because `ort` + `tokenizers` are optional native/download deps
  some CI/dev builds want to skip.
- **The vendored `sea_g2p` module is byte-identical to upstream** (Phạm
  Nguyễn Ngọc Bảo's `sea-g2p` v0.7.18, Apache-2.0) except the pyo3 wrapper is
  stripped. Keep it that way when bumping — it is the exact frontend
  VieNeu-TTS was trained with. `g2p::G2PEngine` memory-maps `sea_g2p.bin`
  (~50 MB, extracted from the sea-g2p PyPI wheel by the composite downloader).
- **Loopback bind + bearer auth is the SDK server scaffold's job**
  (`sen_runtime_sdk::server::serve`), not this repo's.
- No plan ids, phase numbers, or finding codes in code, comments, or test
  names — explain the invariant or behaviour directly.
