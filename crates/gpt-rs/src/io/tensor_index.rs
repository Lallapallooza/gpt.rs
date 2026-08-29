//! Tensor index and payload layout that the `GPTRSCHK` and `GPTRSTEN` formats share. See
//! `docs/formats.md`.

use crate::tensor::{DType, Tensor};
use anyhow::{anyhow, bail, ensure, Result};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const VERSION_V2: u32 = 2;
const PAYLOAD_ALIGN: u64 = 64;

#[derive(Clone, Debug)]
pub(crate) struct IndexEntry {
    pub name: String,
    /// Stored parameter id (`0` when absent or not recorded).
    pub base_id: u128,
    pub dims: Vec<usize>,
    pub dtype: DType,
    pub offset: u64,
    pub len: u64,
}

/// Little-endian reader over a byte slice.
pub(crate) struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    pub fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| anyhow!("unexpected end of file at byte {}", self.pos))?;
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.take(N)?.try_into().expect("take returns N bytes"))
    }

    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn u128(&mut self) -> Result<u128> {
        Ok(u128::from_le_bytes(self.array()?))
    }
}

/// Reads `magic` and the format version. Returns the reader positioned after them.
pub(crate) fn read_header<'a>(bytes: &'a [u8], magic: &[u8; 8]) -> Result<ByteReader<'a>> {
    let mut reader = ByteReader::new(bytes);
    ensure!(
        reader.take(8)? == magic,
        "invalid {} header",
        String::from_utf8_lossy(magic)
    );
    match reader.u32()? {
        VERSION_V2 => Ok(reader),
        other => bail!(
            "unsupported {} version {other}",
            String::from_utf8_lossy(magic)
        ),
    }
}

/// Parses an index. Checks the byte length of each entry against its shape, and its payload range
/// against `file_len`.
pub(crate) fn decode_index(
    reader: &mut ByteReader<'_>,
    with_base_id: bool,
    file_len: u64,
) -> Result<Vec<IndexEntry>> {
    let count = reader.u32()? as usize;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let name_len = reader.u32()? as usize;
        let name = String::from_utf8(reader.take(name_len)?.to_vec())?;
        let base_id = if with_base_id { reader.u128()? } else { 0 };
        let rank = reader.u32()? as usize;
        let dims = (0..rank)
            .map(|_| {
                let dim = reader.u64()?;
                usize::try_from(dim).map_err(|_| anyhow!("tensor {name} dim {dim} overflows usize"))
            })
            .collect::<Result<Vec<_>>>()?;
        let tag = reader.u32()?;
        let dtype = DType::from_tag(tag)
            .ok_or_else(|| anyhow!("tensor {name} has unknown dtype tag {tag}"))?;
        reader.take(1)?; // reserved
        let offset = reader.u64()?;
        let len = reader.u64()?;

        let expected = dims
            .iter()
            .try_fold(dtype.size_in_bytes() as u64, |acc, &d| {
                acc.checked_mul(d as u64)
            });
        ensure!(
            expected == Some(len),
            "tensor {name} byte length {len} does not match shape {dims:?} with dtype {dtype:?}"
        );
        ensure!(
            offset.checked_add(len).is_some_and(|end| end <= file_len),
            "tensor {name} data range {offset}+{len} exceeds file size {file_len}"
        );
        entries.push(IndexEntry {
            name,
            base_id,
            dims,
            dtype,
            offset,
            len,
        });
    }
    Ok(entries)
}

fn encode_index(entries: &[IndexEntry], with_base_id: bool) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for entry in entries {
        out.extend_from_slice(&(entry.name.len() as u32).to_le_bytes());
        out.extend_from_slice(entry.name.as_bytes());
        if with_base_id {
            out.extend_from_slice(&entry.base_id.to_le_bytes());
        }
        out.extend_from_slice(&(entry.dims.len() as u32).to_le_bytes());
        for &dim in &entry.dims {
            out.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        out.extend_from_slice(&entry.dtype.tag().to_le_bytes());
        out.push(0);
        out.extend_from_slice(&entry.offset.to_le_bytes());
        out.extend_from_slice(&entry.len.to_le_bytes());
    }
    out
}

fn align_up(value: u64) -> u64 {
    value.div_ceil(PAYLOAD_ALIGN) * PAYLOAD_ALIGN
}

/// Writes `magic`, the version, `header` (format-specific fields), a `u32`-length-prefixed index,
/// and the `PAYLOAD_ALIGN`-aligned payloads of `tensors`. Each entry of `tensors` is a name, a stored base
/// id, and a tensor.
///
/// The function writes the file next to `path` and renames it into place. Readers that
/// memory-mapped the old file keep a valid mapping.
pub(crate) fn write_tensor_file(
    path: &Path,
    magic: &[u8; 8],
    header: &[u8],
    tensors: &[(&str, u128, &Tensor)],
    with_base_id: bool,
) -> Result<()> {
    let mut entries: Vec<IndexEntry> = tensors
        .iter()
        .map(|&(name, base_id, tensor)| IndexEntry {
            name: name.to_string(),
            base_id,
            dims: tensor.shape().dims().to_vec(),
            dtype: tensor.dtype(),
            offset: 0,
            len: (tensor.len() * tensor.dtype().size_in_bytes()) as u64,
        })
        .collect();
    // Offsets are fixed-width fields, so the index length does not depend on their values.
    let index_len = encode_index(&entries, with_base_id).len();
    ensure!(
        index_len <= u32::MAX as usize,
        "tensor index is larger than u32::MAX bytes"
    );
    let data_start = (magic.len() + 4 + header.len() + 4 + index_len) as u64;
    let mut offset = data_start;
    for entry in &mut entries {
        entry.offset = align_up(offset);
        offset = entry.offset + entry.len;
    }

    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let write = || -> Result<()> {
        let mut writer = BufWriter::new(File::create(&tmp)?);
        writer.write_all(magic)?;
        writer.write_all(&VERSION_V2.to_le_bytes())?;
        writer.write_all(header)?;
        writer.write_all(&(index_len as u32).to_le_bytes())?;
        writer.write_all(&encode_index(&entries, with_base_id))?;
        let mut written = data_start;
        for (entry, (_, _, tensor)) in entries.iter().zip(tensors) {
            writer.write_all(&vec![0u8; (entry.offset - written) as usize])?;
            writer.write_all(tensor.to_le_bytes())?;
            written = entry.offset + entry.len;
        }
        writer
            .into_inner()
            .map_err(|err| err.into_error())?
            .sync_all()?;
        Ok(std::fs::rename(&tmp, path)?)
    };
    write().inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}
