//! Splitting text into synthesisable chunks.
//!
//! Lifted out of the MMS-VITS backend when that moved to `apps/mlx-media`: it is
//! plain string work with no engine behind it, and VieNeu — which stayed in the
//! daemon — was the other caller.

/// Split text into synthesis chunks at sentence boundaries, keeping each chunk
/// under `max_chars`. Never splits mid-word: overlong sentences fall back to
/// clause separators, then whitespace.
pub fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        cur.push(c);
        if matches!(c, '.' | '!' | '?' | '…' | '\n' | ';') {
            if !cur.trim().is_empty() {
                sentences.push(cur.trim().to_string());
            }
            cur.clear();
        }
    }
    if !cur.trim().is_empty() {
        sentences.push(cur.trim().to_string());
    }

    // Merge short sentences up to max_chars; split overlong ones on , then space.
    let mut chunks: Vec<String> = Vec::new();
    for s in sentences {
        let pieces: Vec<String> = if s.chars().count() <= max_chars {
            vec![s]
        } else {
            split_long(&s, max_chars)
        };
        for p in pieces {
            match chunks.last_mut() {
                Some(last) if last.chars().count() + 1 + p.chars().count() <= max_chars => {
                    last.push(' ');
                    last.push_str(&p);
                }
                _ => chunks.push(p),
            }
        }
    }
    chunks
}

fn split_long(s: &str, max_chars: usize) -> Vec<String> {
    for sep in [',', ' '] {
        let parts: Vec<&str> = s.split(sep).collect();
        if parts.len() < 2 {
            continue;
        }
        let mut out: Vec<String> = Vec::new();
        let mut cur = String::new();
        for p in parts {
            if !cur.is_empty() && cur.chars().count() + 1 + p.chars().count() > max_chars {
                out.push(cur.trim().to_string());
                cur = String::new();
            }
            if !cur.is_empty() {
                cur.push(sep);
            }
            cur.push_str(p);
        }
        if !cur.trim().is_empty() {
            out.push(cur.trim().to_string());
        }
        if out.iter().all(|c| c.chars().count() <= max_chars) {
            return out;
        }
    }
    // Last resort: hard character split.
    s.chars()
        .collect::<Vec<_>>()
        .chunks(max_chars)
        .map(|c| c.iter().collect())
        .collect()
}
