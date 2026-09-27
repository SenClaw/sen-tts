//! `senclaw-runtime.json` must parse with the SDK and stay in lockstep with
//! the crate's own version — a drift here means the daemon installs a
//! package whose manifest lies about what it is.

#[test]
fn manifest_parses_and_matches_the_crate_version() {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/senclaw-runtime.json"))
        .expect("senclaw-runtime.json must exist at the package root");
    let parsed = sen_runtime_sdk::manifest::RuntimeManifest::parse(&text).expect("manifest must be valid");
    assert!(parsed.warnings.is_empty(), "unexpected warnings: {:?}", parsed.warnings);
    let m = parsed.manifest;
    assert_eq!(m.id, "sen-tts");
    assert_eq!(m.version, env!("CARGO_PKG_VERSION"));
    assert_eq!(m.slots, vec![sen_runtime_sdk::manifest::Slot::Tts]);
    assert_eq!(m.mode, sen_runtime_sdk::manifest::RunMode::Service);
    assert_eq!(m.capabilities, vec![sen_runtime_sdk::manifest::Capability::Tts]);
}
