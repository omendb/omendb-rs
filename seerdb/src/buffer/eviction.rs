use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};

pub type FrameId = usize;

pub trait EvictionPolicy: Send + Sync {
    fn access(&self, frame_id: FrameId);
    fn evict(&self) -> Option<FrameId>;
    fn remove(&self, frame_id: FrameId);
}

/// Clock-Pro replacement policy.
///
/// Improves on basic Clock by distinguishing hot vs cold pages:
/// - Cold pages: accessed once (may be scan/one-shot access)
/// - Hot pages: accessed multiple times (working set)
///
/// Benefits:
/// - Scan-resistant: large sequential scans don't evict frequently-used data
/// - 10-20% better hit rate than basic Clock for mixed workloads
///
/// State per frame (2 bits):
/// - 00: cold, not referenced
/// - 01: cold, referenced (in test period)
/// - 10: hot, not referenced
/// - 11: hot, referenced
pub struct ClockProPolicy {
    hand: AtomicUsize,
    capacity: usize,
    // Each entry: bits 0 = referenced, bit 1 = hot
    state: Vec<AtomicU8>,
}

// State bit masks
const REF_BIT: u8 = 0b01;
const HOT_BIT: u8 = 0b10;

impl ClockProPolicy {
    pub fn new(capacity: usize) -> Self {
        let mut state = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            state.push(AtomicU8::new(0));
        }
        Self {
            hand: AtomicUsize::new(0),
            capacity,
            state,
        }
    }
}

impl EvictionPolicy for ClockProPolicy {
    fn access(&self, frame_id: FrameId) {
        if frame_id >= self.capacity {
            return;
        }

        // Load current state
        let old = self.state[frame_id].load(Ordering::Relaxed);

        if (old & HOT_BIT) != 0 {
            // Already hot: just set reference bit
            self.state[frame_id].fetch_or(REF_BIT, Ordering::Relaxed);
        } else if (old & REF_BIT) != 0 {
            // Cold but referenced (in test period): promote to hot
            self.state[frame_id].store(HOT_BIT | REF_BIT, Ordering::Relaxed);
        } else {
            // Cold, not referenced: set reference bit (start test period)
            self.state[frame_id].fetch_or(REF_BIT, Ordering::Relaxed);
        }
    }

    fn evict(&self) -> Option<FrameId> {
        let max_iterations = self.capacity * 3;
        let mut iterations = 0;

        while iterations < max_iterations {
            let current = self.hand.fetch_add(1, Ordering::Relaxed) % self.capacity;
            iterations += 1;

            let state = self.state[current].load(Ordering::Relaxed);

            if (state & HOT_BIT) != 0 {
                // Hot page
                if (state & REF_BIT) != 0 {
                    // Hot + referenced: clear reference, keep hot
                    self.state[current].fetch_and(!REF_BIT, Ordering::Relaxed);
                } else {
                    // Hot + not referenced: demote to cold
                    self.state[current].fetch_and(!HOT_BIT, Ordering::Relaxed);
                }
            } else {
                // Cold page
                if (state & REF_BIT) != 0 {
                    // Cold + referenced (test period): clear reference, stay cold
                    // This gives it one more chance before eviction
                    self.state[current].fetch_and(!REF_BIT, Ordering::Relaxed);
                } else {
                    // Cold + not referenced: evict this page
                    return Some(current);
                }
            }
        }

        None // Everything is hot/referenced
    }

    fn remove(&self, frame_id: FrameId) {
        if frame_id < self.capacity {
            self.state[frame_id].store(0, Ordering::Relaxed);
        }
    }
}

/// Clock (Second Chance) replacement policy.
/// Efficient O(1) access and eviction.
pub struct ClockPolicy {
    hand: AtomicUsize,
    capacity: usize,
    reference_bits: Vec<AtomicBool>,
}

impl ClockPolicy {
    pub fn new(capacity: usize) -> Self {
        let mut bits = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            bits.push(AtomicBool::new(false));
        }
        Self {
            hand: AtomicUsize::new(0),
            capacity,
            reference_bits: bits,
        }
    }
}

impl EvictionPolicy for ClockPolicy {
    fn access(&self, frame_id: FrameId) {
        if frame_id < self.capacity {
            self.reference_bits[frame_id].store(true, Ordering::Relaxed);
        }
    }

    fn evict(&self) -> Option<FrameId> {
        // Spin the clock hand at most 2 times around to find a victim.
        // If everyone is referenced, we might cycle forever, so limit loops.
        let start_hand = self.hand.load(Ordering::Relaxed);
        let mut loops = 0;

        loop {
            let current_hand = self.hand.fetch_add(1, Ordering::Relaxed) % self.capacity;

            // Check reference bit
            if self.reference_bits[current_hand].load(Ordering::Relaxed) {
                // Give second chance: set to 0 and continue
                self.reference_bits[current_hand].store(false, Ordering::Relaxed);
            } else {
                // Found victim (bit was 0)
                return Some(current_hand);
            }

            // Safety valve: if we've scanned 2x capacity, everything is hot.
            // Just return the current hand to force eviction?
            // Or return None to signal "cache pressure but no victim"?
            // For now, returning None is safer, but a DB usually *needs* a page.
            if current_hand == start_hand {
                loops += 1;
                if loops >= 2 {
                    return None;
                }
            }
        }
    }

    fn remove(&self, frame_id: FrameId) {
        if frame_id < self.capacity {
            self.reference_bits[frame_id].store(false, Ordering::Relaxed);
        }
    }
}

// Alias LruPolicy to ClockPolicy for now, as it's our default
pub type LruPolicy = ClockPolicy;
