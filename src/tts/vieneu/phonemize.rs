//! Text → phoneme front-end for VieNeu-TTS v3 Turbo.
//!
//! Port of `vieneu_utils/phonemize_text.py::phonemize_text_with_emotions`:
//! inline non-verbal cues (`[cười]`, `[thở dài]`, `[hắng giọng]`, English
//! equivalents, or explicit `<|emotion_k|>`) are preserved as emotion tokens in
//! the phoneme stream — everything else goes through the vendored sea-g2p
//! pipeline (normalize → G2P), with the chunk-level terminal punctuation
//! normalized at the end. Spacing mirrors the training data: one space before
//! an emotion token, following punctuation attached.

use super::sea_g2p::{apply_punc_norm, SeaPipeline};

/// Punctuation that attaches to the preceding token without a space.
const ATTACHING_PUNCT: &[char] = &[
    '.', ',', '!', '?', ';', ':', '…', ')', ']', '}', '"', '\'', '’', '”',
];

fn emotion_tag_token(tag: &str) -> Option<String> {
    let t = tag.trim();
    if t.starts_with("<|") {
        return Some(t.to_string()); // already explicit — pass through
    }
    let inner = t
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim()
        .to_lowercase();
    let k = match inner.as_str() {
        "chuckle" | "cười" | "cuoi" => 1,
        "sigh" | "thở dài" | "tho dai" => 2,
        "clear throat" | "hắng giọng" | "hang giong" => 3,
        _ => return None,
    };
    Some(format!("<|emotion_{k}|>"))
}

/// Split `text` into alternating (fragment, tag, fragment, …) parts, where a
/// tag is `[...]` or `<|emotion_k|>`. Mirrors Python's `re.split` with a
/// capturing group: odd indices are tags.
fn split_emotion_parts(text: &str) -> Vec<(bool, String)> {
    let re = regex::Regex::new(r"(\[[^\]]+\]|<\|emotion_\d+\|>)").expect("static regex");
    let mut out = Vec::new();
    let mut last = 0;
    for m in re.find_iter(text) {
        out.push((false, text[last..m.start()].to_string()));
        out.push((true, m.as_str().to_string()));
        last = m.end();
    }
    out.push((false, text[last..].to_string()));
    out
}

/// Phonemize with emotion-cue preservation (chunk-level punc_norm at the end).
pub fn phonemize_with_emotions(pipeline: &SeaPipeline, text: &str) -> String {
    if !text.contains('[') && !text.contains("<|emotion_") {
        return pipeline.run(text, true);
    }
    let mut out = String::new();
    for (is_tag, part) in split_emotion_parts(text) {
        let token = if is_tag {
            emotion_tag_token(&part)
        } else {
            None
        };
        match token {
            Some(tok) => {
                if out.is_empty() {
                    out = tok;
                } else {
                    out.push(' ');
                    out.push_str(&tok);
                }
            }
            None => {
                // Unrecognized [tags] phonemize as ordinary text (Python parity).
                if part.trim().is_empty() {
                    continue;
                }
                // Fragments inside a chunk: NO punc_norm (would inject "." mid-chunk).
                let ph = pipeline.run(&part, false);
                if ph.is_empty() {
                    continue;
                }
                if out.is_empty() {
                    out = ph;
                } else if ph.starts_with(ATTACHING_PUNCT) {
                    out.push_str(&ph);
                } else {
                    out.push(' ');
                    out.push_str(&ph);
                }
            }
        }
    }
    apply_punc_norm(&out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emotion_tags_resolve() {
        assert_eq!(
            emotion_tag_token("[cười]").as_deref(),
            Some("<|emotion_1|>")
        );
        assert_eq!(
            emotion_tag_token("[Thở dài]").as_deref(),
            Some("<|emotion_2|>")
        );
        assert_eq!(
            emotion_tag_token("[clear throat]").as_deref(),
            Some("<|emotion_3|>")
        );
        assert_eq!(
            emotion_tag_token("<|emotion_2|>").as_deref(),
            Some("<|emotion_2|>")
        );
        assert_eq!(emotion_tag_token("[ghi chú]"), None);
    }

    #[test]
    fn split_alternates_fragments_and_tags() {
        let parts = split_emotion_parts("xin chào [cười] nhé <|emotion_2|>.");
        let tags: Vec<_> = parts
            .iter()
            .filter(|(t, _)| *t)
            .map(|(_, s)| s.clone())
            .collect();
        assert_eq!(
            tags,
            vec!["[cười]".to_string(), "<|emotion_2|>".to_string()]
        );
    }
}
