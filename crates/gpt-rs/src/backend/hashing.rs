use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const FNV1A_OFFSET: u64 = 0xcbf29ce484222325;
const FNV1A_PRIME: u64 = 0x100000001b3;

pub struct FingerprintHasher {
    inner: DefaultHasher,
}

impl FingerprintHasher {
    pub fn new() -> Self {
        Self {
            inner: DefaultHasher::new(),
        }
    }

    pub fn write<T: Hash>(&mut self, value: &T) {
        value.hash(&mut self.inner);
    }

    pub fn write_u8(&mut self, value: u8) {
        self.inner.write_u8(value);
    }

    pub fn finish(self) -> u64 {
        self.inner.finish()
    }
}

impl Default for FingerprintHasher {
    fn default() -> Self {
        Self::new()
    }
}

pub fn hash_value<T: Hash>(value: &T) -> u64 {
    let mut hasher = FingerprintHasher::new();
    hasher.write(value);
    hasher.finish()
}

pub fn fnv1a_init() -> u64 {
    FNV1A_OFFSET
}

pub fn fnv1a_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV1A_PRIME);
    }
    hash
}

pub fn fnv1a_hash(bytes: &[u8]) -> u64 {
    fnv1a_bytes(fnv1a_init(), bytes)
}
