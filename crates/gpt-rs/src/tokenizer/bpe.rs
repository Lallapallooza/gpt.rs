//! Byte pair encoding helpers that implement the merge priority table used by GPT-style
//! tokenizers.

use std::collections::{HashMap, HashSet};

/// Lookup table for the rank (priority) of individual byte pair merges.
#[derive(Debug, Clone)]
pub struct BpeMerges {
    /// Maps an adjacent token pair to the order in which it should be merged.
    ranks: HashMap<(String, String), usize>,
}

impl BpeMerges {
    /// Creates a merge table from the pre-computed pair ranking produced during tokenizer
    /// training. Lower ranks take precedence when collapsing tokens.
    pub fn new(ranks: HashMap<(String, String), usize>) -> Self {
        BpeMerges { ranks }
    }

    /// Returns the merge priority for a given token pair, if present.
    ///
    /// The returned value follows the GPT-2 convention where smaller numbers indicate that
    /// the pair should be merged earlier in the iterative BPE loop. An absent value means the
    /// pair never appears in the learned merge list and should therefore be left untouched.
    pub fn rank(&self, pair: &(String, String)) -> Option<usize> {
        self.ranks.get(pair).copied()
    }
}

/// Enumerates every adjacent token pair within a word to drive the BPE merge process.
///
/// The tokenizer repeatedly calls this helper to collect merge candidates, gradually
/// collapsing the most likely pairs until either a single token remains or no known merges
/// are available.
pub(crate) fn get_pairs(word: &[String]) -> HashSet<(String, String)> {
    let mut pairs = HashSet::new();
    if word.len() < 2 {
        return pairs;
    }
    for window in word.windows(2) {
        pairs.insert((window[0].clone(), window[1].clone()));
    }
    pairs
}
