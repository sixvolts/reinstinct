//! Build a tiny synthetic GGUF entirely in memory and parse it back.
//!
//! This catches regressions in alignment, tensor-info layout, metadata
//! ordering, and offset arithmetic without needing a real model on disk.

use std::io::Write;

use memmap2::MmapMut;
use reinstinct_engine::gguf::{GgmlType, GgufFile};

const GGUF_MAGIC: u32 = 0x4655_4747;

fn write_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn align_up(v: usize, a: usize) -> usize {
    v.div_ceil(a) * a
}

#[test]
fn parses_synthetic_minimal_gguf() {
    let mut hdr = Vec::<u8>::new();

    // Header
    hdr.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    hdr.extend_from_slice(&3u32.to_le_bytes());      // version
    hdr.extend_from_slice(&1u64.to_le_bytes());      // tensor_count
    hdr.extend_from_slice(&2u64.to_le_bytes());      // metadata_kv_count

    // Metadata KV #0: "general.architecture" = "test-arch" (string, type=8)
    write_string(&mut hdr, "general.architecture");
    hdr.extend_from_slice(&8u32.to_le_bytes());
    write_string(&mut hdr, "test-arch");

    // Metadata KV #1: "general.alignment" = 32 (u32, type=4)
    write_string(&mut hdr, "general.alignment");
    hdr.extend_from_slice(&4u32.to_le_bytes());
    hdr.extend_from_slice(&32u32.to_le_bytes());

    // Tensor info: name="x", n_dims=1, dims=[4], type=F32, offset=0
    write_string(&mut hdr, "x");
    hdr.extend_from_slice(&1u32.to_le_bytes());
    hdr.extend_from_slice(&4u64.to_le_bytes());
    hdr.extend_from_slice(&0u32.to_le_bytes()); // F32
    hdr.extend_from_slice(&0u64.to_le_bytes());

    // Pad to alignment (32) before tensor data section.
    let data_offset = align_up(hdr.len(), 32);
    hdr.resize(data_offset, 0);

    // Tensor data: four f32s = [1.0, 2.0, 3.0, 4.0]
    for v in [1.0f32, 2.0, 3.0, 4.0] {
        hdr.extend_from_slice(&v.to_le_bytes());
    }

    // Materialize as anonymous mmap so GgufFile::from_mmap can take it.
    let mut mmap = MmapMut::map_anon(hdr.len()).unwrap();
    (&mut mmap[..]).write_all(&hdr).unwrap();
    let mmap = mmap.make_read_only().unwrap();

    let g = GgufFile::from_mmap(mmap).expect("parse");

    assert_eq!(g.header.version, 3);
    assert_eq!(g.header.tensor_count, 1);
    assert_eq!(g.alignment, 32);
    assert_eq!(g.data_section_offset as usize, data_offset);

    let arch = g.metadata_get("general.architecture").unwrap();
    assert_eq!(arch.as_str(), Some("test-arch"));

    let t = g.tensor("x").unwrap();
    assert_eq!(t.ggml_type, GgmlType::F32);
    assert_eq!(t.shape(), &[4]);
    assert_eq!(t.byte_size().unwrap(), 16);

    let bytes = g.tensor_data("x").unwrap().unwrap();
    assert_eq!(bytes.len(), 16);
    let v0 = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let v3 = f32::from_le_bytes(bytes[12..16].try_into().unwrap());
    assert_eq!(v0, 1.0);
    assert_eq!(v3, 4.0);
}

#[test]
fn rejects_bad_magic() {
    let buf = vec![0u8; 64];
    let mut mmap = MmapMut::map_anon(buf.len()).unwrap();
    (&mut mmap[..]).write_all(&buf).unwrap();
    let mmap = mmap.make_read_only().unwrap();
    match GgufFile::from_mmap(mmap) {
        Ok(_) => panic!("expected BadMagic error"),
        Err(reinstinct_engine::gguf::GgufError::BadMagic { .. }) => {}
        Err(e) => panic!("expected BadMagic, got {e:?}"),
    }
}

