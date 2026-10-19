//! Tests of `Buffer`: its views of external bytes, its owned storage and its copy on write.

use std::sync::Arc;

use gpt_rs_ir::host::{Buffer, ExternalBytes};

const VALUES: [f32; 3] = [1.0, -2.5, 3.25];

/// The bytes of `VALUES` in host byte order, `shift` bytes into a 4-byte aligned allocation.
struct Shifted {
    words: Vec<u32>,
    shift: usize,
}

impl Shifted {
    fn new(shift: usize) -> Arc<Self> {
        let mut words = vec![0u32; VALUES.len() + 1];
        let bytes: Vec<u8> = VALUES
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        bytemuck::cast_slice_mut::<u32, u8>(&mut words)[shift..shift + bytes.len()]
            .copy_from_slice(&bytes);
        Arc::new(Self { words, shift })
    }
}

impl AsRef<[u8]> for Shifted {
    fn as_ref(&self) -> &[u8] {
        &bytemuck::cast_slice(&self.words)[self.shift..self.shift + 4 * VALUES.len()]
    }
}

#[test]
fn aligned_bytes_are_read_in_place() {
    let owner = Shifted::new(0);
    let buffer = Buffer::<f32>::from_bytes(owner.clone()).unwrap();
    assert_eq!(*buffer, VALUES);
    assert_eq!(buffer.as_ptr().cast::<u8>(), (*owner).as_ref().as_ptr());
}

#[test]
fn misaligned_bytes_are_copied() {
    let owner = Shifted::new(1);
    let buffer = Buffer::<f32>::from_bytes(owner.clone()).unwrap();
    assert_eq!(*buffer, VALUES);
    assert_ne!(buffer.as_ptr().cast::<u8>(), (*owner).as_ref().as_ptr());
}

#[test]
fn writing_a_view_leaves_the_owner_unchanged() {
    let owner = Shifted::new(0);
    let mut buffer = Buffer::<f32>::from_bytes(owner.clone()).unwrap();
    buffer.make_mut()[0] = 7.0;
    assert_eq!(*buffer, [7.0, -2.5, 3.25]);
    assert_eq!(*Buffer::<f32>::from_bytes(owner).unwrap(), VALUES);
}

#[test]
fn a_partial_element_is_rejected() {
    let owner: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(vec![0u8; 5]);
    let err = Buffer::<f32>::from_bytes(owner).unwrap_err();
    assert_eq!(
        err.to_string(),
        "5 bytes do not divide into 4-byte elements"
    );
}

#[test]
fn a_view_of_misaligned_bytes_is_rejected() {
    let bytes = ExternalBytes::new(Shifted::new(1), 0, 4 * VALUES.len()).unwrap();
    let err = Buffer::<f32>::view(bytes).unwrap_err();
    assert_eq!(
        err.to_string(),
        "12 bytes at offset 0 are not aligned for 4-byte elements"
    );
}

#[test]
fn owned_elements_start_on_a_cache_line() {
    let view = Buffer::<f32>::from_bytes(Shifted::new(0)).unwrap();
    let mut copied = view.clone();
    copied.make_mut();
    let buffers = [
        Buffer::<f32>::zeroed(3),
        Buffer::from(VALUES.to_vec()),
        VALUES.iter().copied().collect(),
        Buffer::<f32>::from_bytes(Shifted::new(1)).unwrap(),
        copied,
    ];
    for buffer in &buffers {
        assert_eq!(buffer.as_ptr() as usize % 64, 0);
    }
}

#[test]
fn writing_a_shared_buffer_leaves_its_clone_unchanged() {
    let mut buffer = Buffer::from(VALUES.to_vec());
    assert!(buffer.is_unique());
    let clone = buffer.clone();
    assert!(!buffer.is_unique());
    buffer.make_mut()[0] = 7.0;
    assert!(buffer.is_unique() && clone.is_unique());
    assert_eq!(*clone, VALUES);
    let before = buffer.as_ptr();
    buffer.make_mut()[1] = 8.0;
    assert_eq!(buffer.as_ptr(), before);
    assert!(!Buffer::<f32>::from_bytes(Shifted::new(0))
        .unwrap()
        .is_unique());
}
