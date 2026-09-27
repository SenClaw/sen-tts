//! Minimal NPY/NPZ reader — just enough for `vieneu_v3_heads.npz`.
//!
//! Supports little-endian `<f4`/`<f8`/`<i4`/`<i8` C-order arrays (the only
//! dtypes the heads archive uses). Everything is materialized as `f32`.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};

/// A dense C-order f32 tensor.
#[derive(Debug, Clone)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl NpyArray {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Row `i` of a 2-D array.
    pub fn row(&self, i: usize) -> &[f32] {
        let cols = self.shape[self.shape.len() - 1];
        &self.data[i * cols..(i + 1) * cols]
    }

    /// Scalar (0-d or single-element) value.
    pub fn scalar(&self) -> f32 {
        self.data[0]
    }
}

/// Parse one `.npy` blob.
pub fn parse_npy(bytes: &[u8]) -> Result<NpyArray> {
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        bail!("not an NPY file");
    }
    let major = bytes[6];
    let (header, data_start) = if major == 1 {
        let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (&bytes[10..10 + len], 10 + len)
    } else {
        let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        (&bytes[12..12 + len], 12 + len)
    };
    let header = std::str::from_utf8(header).context("npy header not utf8")?;

    let descr = extract_quoted(header, "descr").ok_or_else(|| anyhow!("npy: no descr"))?;
    if header.contains("'fortran_order': True") {
        bail!("npy: fortran order not supported");
    }
    let shape = parse_shape(header)?;
    let count: usize = shape.iter().product::<usize>().max(1);
    let raw = &bytes[data_start..];

    let data: Vec<f32> = match descr.as_str() {
        "<f4" | "|f4" => raw[..count * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        "<f8" => raw[..count * 8]
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        "<i4" => raw[..count * 4]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        "<i8" => raw[..count * 8]
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        other => bail!("npy: unsupported dtype `{other}`"),
    };
    Ok(NpyArray { shape, data })
}

/// Load every array in an `.npz` (a zip of `.npy` members).
pub fn load_npz(path: &Path) -> Result<HashMap<String, NpyArray>> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut zip = zip::ZipArchive::new(file).context("npz is not a zip")?;
    let mut out = HashMap::new();
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        let name = entry.name().trim_end_matches(".npy").to_string();
        let mut buf = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut buf)?;
        out.insert(
            name.clone(),
            parse_npy(&buf).with_context(|| format!("parsing npz member `{name}`"))?,
        );
    }
    Ok(out)
}

fn extract_quoted(header: &str, key: &str) -> Option<String> {
    let pos = header.find(&format!("'{key}'"))?;
    let rest = &header[pos..];
    let start = rest.find(": '")? + 3;
    let end = rest[start..].find('\'')? + start;
    Some(rest[start..end].to_string())
}

fn parse_shape(header: &str) -> Result<Vec<usize>> {
    let pos = header
        .find("'shape'")
        .ok_or_else(|| anyhow!("npy: no shape"))?;
    let rest = &header[pos..];
    let open = rest.find('(').ok_or_else(|| anyhow!("npy: bad shape"))?;
    let close = rest[open..]
        .find(')')
        .ok_or_else(|| anyhow!("npy: bad shape"))?
        + open;
    Ok(rest[open + 1..close]
        .split(',')
        .filter_map(|s| {
            let t = s.trim();
            if t.is_empty() {
                None
            } else {
                t.parse::<usize>().ok()
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy_v1(descr: &str, shape: &str, data: &[u8]) -> Vec<u8> {
        let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
        let mut header = dict.into_bytes();
        while (10 + header.len() + 1) % 16 != 0 {
            header.push(b' ');
        }
        header.push(b'\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn parses_f4_and_f8_and_i8() {
        let f4 = npy_v1(
            "<f4",
            "(2, 2)",
            &[1.0f32, 2.0, 3.0, 4.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let a = parse_npy(&f4).unwrap();
        assert_eq!(a.shape, vec![2, 2]);
        assert_eq!(a.row(1), &[3.0, 4.0]);

        let f8 = npy_v1(
            "<f8",
            "(3,)",
            &[0.5f64, -1.5, 2.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let b = parse_npy(&f8).unwrap();
        assert_eq!(b.data, vec![0.5, -1.5, 2.0]);

        let i8v = npy_v1(
            "<i8",
            "(2,)",
            &[7i64, -3]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let c = parse_npy(&i8v).unwrap();
        assert_eq!(c.data, vec![7.0, -3.0]);
    }

    #[test]
    fn scalar_and_bad_magic() {
        let s = npy_v1("<f4", "()", &1.25f32.to_le_bytes());
        assert_eq!(parse_npy(&s).unwrap().scalar(), 1.25);
        assert!(parse_npy(b"nope").is_err());
    }
}
