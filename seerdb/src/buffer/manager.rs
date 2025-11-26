use crate::buffer::eviction::{ClockProPolicy, EvictionPolicy, FrameId};
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

pub type FileId = u64;
pub type BlockId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId {
    pub file_id: FileId,
    pub offset: BlockId,
}

#[derive(Debug, Clone)]
pub struct BufferPoolOptions {
    pub capacity_bytes: usize,
    pub frame_size: usize,
    pub num_shards: usize, // New: Number of shards for partitioning
}

impl Default for BufferPoolOptions {
    fn default() -> Self {
        Self {
            capacity_bytes: 128 * 1024 * 1024, // 128MB
            frame_size: 16 * 1024,             // 16KB
            num_shards: 16,                    // 16 shards for multi-core systems
        }
    }
}

/// Thread-safe reference to a pinned frame.
pub struct FrameRef {
    pool: Arc<BufferPool>,
    page_id: PageId,
    frame_id: FrameId,
    // Cached data pointer for lock-free access
    // SAFETY: Valid while frame is pinned (pin_count > 0) and data is immutable
    data_ptr: *const u8,
    data_len: usize,
}

impl FrameRef {
    pub fn data(&self) -> &[u8] {
        // SAFETY:
        // 1. We hold a pin count > 0 (ensured by constructor and Drop).
        // 2. Eviction policy checks pin count before repurposing frame.
        // 3. Therefore, the data in this frame belongs to self.page_id and won't change.
        // 4. We take a read lock just to be safe against any internal mutability or race
        //    during the initial loading phase, though technically once pinned and loaded, it's stable.
        //    Actually, to return &[u8], we need to bypass the RwLock or return a Guard.
        //    Returning a Guard is safer.
        panic!("Use get_data() which returns a guard");
    }

    pub fn get_data(&self) -> parking_lot::RwLockReadGuard<'_, Vec<u8>> {
        self.pool.get_frame_data(self.frame_id)
    }

    /// Lock-free access to data slice using cached pointer.
    ///
    /// # Safety
    /// Caller must ensure that:
    /// 1. The frame is pinned (which `FrameRef` ensures).
    /// 2. The data is not being mutated concurrently (guaranteed for SSTables as they are immutable).
    /// 3. The `Vec` does not reallocate (guaranteed as we only resize on initial load).
    ///
    /// This is safe because:
    /// - Frame is pinned (pin_count > 0) so it won't be evicted
    /// - SSTable data is immutable (never modified after loading)
    /// - Vec won't reallocate (we resize() once during load, then never touch it)
    /// - data_ptr and data_len are captured after data is loaded
    pub unsafe fn data_unchecked(&self) -> &[u8] {
        std::slice::from_raw_parts(self.data_ptr, self.data_len)
    }
}

// SAFETY: FrameRef is safe to Send/Sync because:
// 1. data_ptr points to data owned by BufferPool (behind Arc)
// 2. Frame is pinned (won't be evicted) while FrameRef exists
// 3. Data is immutable (SSTable blocks never change after loading)
// 4. All BufferPool internal state is already Send + Sync
unsafe impl Send for FrameRef {}
unsafe impl Sync for FrameRef {}

impl Clone for FrameRef {
    fn clone(&self) -> Self {
        // Increment pin count to account for the new reference
        self.pool.pin_frame(self.frame_id);
        Self {
            pool: self.pool.clone(),
            page_id: self.page_id,
            frame_id: self.frame_id,
            data_ptr: self.data_ptr,
            data_len: self.data_len,
        }
    }
}

impl Drop for FrameRef {
    fn drop(&mut self) {
        self.pool.unpin(self.frame_id);
    }
}

impl fmt::Debug for FrameRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FrameRef(page={:?}, frame={})",
            self.page_id, self.frame_id
        )
    }
}

struct FrameHeader {
    pin_count: AtomicUsize,
    is_dirty: AtomicBool,
    // The page_id currently stored here.
    // We use a Lock because changing ownership (eviction) is complex.
    page_id: RwLock<Option<PageId>>,
}

struct FrameSlot {
    header: FrameHeader,
    data: RwLock<Vec<u8>>,
}

/// A shard of the buffer pool - independent partition with own page table and eviction
struct BufferShard {
    frames: Vec<FrameSlot>,
    page_table: DashMap<PageId, FrameId>,
    free_list: Mutex<Vec<FrameId>>,
    eviction: Box<dyn EvictionPolicy>,
    frame_offset: usize, // Global frame ID offset for this shard
    frames_per_shard: usize,
}

impl BufferShard {
    fn new(shard_idx: usize, frames_per_shard: usize, frame_size: usize) -> Self {
        let frame_offset = shard_idx * frames_per_shard;
        let mut frames = Vec::with_capacity(frames_per_shard);
        let mut free_list = Vec::with_capacity(frames_per_shard);

        for i in 0..frames_per_shard {
            frames.push(FrameSlot {
                header: FrameHeader {
                    pin_count: AtomicUsize::new(0),
                    is_dirty: AtomicBool::new(false),
                    page_id: RwLock::new(None),
                },
                data: RwLock::new(vec![0u8; frame_size]),
            });
            free_list.push(frame_offset + i); // Global frame ID
        }

        Self {
            frames,
            page_table: DashMap::new(),
            free_list: Mutex::new(free_list),
            eviction: Box::new(ClockProPolicy::new(frames_per_shard)),
            frame_offset,
            frames_per_shard,
        }
    }

    #[inline]
    fn local_frame_id(&self, global_id: FrameId) -> usize {
        global_id - self.frame_offset
    }

    fn allocate_frame(&self) -> Option<FrameId> {
        // 1. Free list
        {
            let mut free = self.free_list.lock();
            if let Some(id) = free.pop() {
                return Some(id);
            }
        }

        // 2. Eviction (local frame IDs)
        let max_attempts = self.frames_per_shard * 2;
        let mut attempts = 0;

        while attempts < max_attempts {
            attempts += 1;
            if let Some(local_victim_id) = self.eviction.evict() {
                let slot = &self.frames[local_victim_id];

                // Check pin count
                if slot.header.pin_count.load(Ordering::SeqCst) > 0 {
                    continue;
                }

                // Lock page_id to claim ownership
                let mut pid_guard = slot.header.page_id.write();

                // Check pin count again under lock
                if slot.header.pin_count.load(Ordering::SeqCst) > 0 {
                    continue;
                }

                // Remove from page table
                if let Some(old_pid) = *pid_guard {
                    self.page_table.remove(&old_pid);
                }

                *pid_guard = None; // Mark as invalid/being setup

                // Return global frame ID
                return Some(self.frame_offset + local_victim_id);
            }
        }

        None // Failed to find victim
    }

    fn free_frame(&self, global_frame_id: FrameId) {
        let mut free = self.free_list.lock();
        free.push(global_frame_id);
    }
}

pub struct BufferPool {
    shards: Vec<BufferShard>,
    num_shards: usize,
    #[allow(dead_code)] // Kept for future use (e.g. resizing)
    options: BufferPoolOptions,
    // Stable hasher for consistent shard selection
    // CRITICAL: Must be created once and reused, NOT per-call!
    hasher: std::collections::hash_map::RandomState,
}

impl BufferPool {
    pub fn new(options: BufferPoolOptions) -> Arc<Self> {
        let num_frames = options.capacity_bytes / options.frame_size;
        let frames_per_shard = num_frames.div_ceil(options.num_shards);

        let mut shards = Vec::with_capacity(options.num_shards);
        for shard_idx in 0..options.num_shards {
            shards.push(BufferShard::new(
                shard_idx,
                frames_per_shard,
                options.frame_size,
            ));
        }

        Arc::new(Self {
            shards,
            num_shards: options.num_shards,
            options,
            hasher: std::collections::hash_map::RandomState::new(),
        })
    }

    #[inline]
    fn hash_page_id(&self, page_id: PageId) -> usize {
        // Use stored hasher for CONSISTENT shard selection
        // CRITICAL: Same page_id must ALWAYS hash to same shard!
        use std::hash::BuildHasher;
        let hash = self.hasher.hash_one(page_id);
        (hash as usize) % self.num_shards
    }

    #[inline]
    fn get_shard(&self, page_id: PageId) -> &BufferShard {
        let shard_idx = self.hash_page_id(page_id);
        &self.shards[shard_idx]
    }

    #[inline]
    fn get_shard_by_frame(&self, frame_id: FrameId) -> &BufferShard {
        // Each shard owns a contiguous range of frame IDs
        let shard_idx = frame_id / self.shards[0].frames_per_shard;
        &self.shards[shard_idx.min(self.num_shards - 1)]
    }

    pub fn get_page<F, E>(self: &Arc<Self>, page_id: PageId, loader: F) -> Result<FrameRef, E>
    where
        F: FnOnce(&mut Vec<u8>) -> Result<(), E>,
        E: From<crate::buffer::BufferPoolError>,
    {
        let shard = self.get_shard(page_id);

        // 1. Try find in cache
        if let Some(frame_ref) = self.lookup(shard, page_id) {
            return Ok(frame_ref);
        }

        // 2. Miss - Allocate frame
        let frame_id = match shard.allocate_frame() {
            Some(id) => id,
            None => return Err(self.make_capacity_error::<E>()),
        };

        // CRITICAL: Pin frame immediately to prevent eviction from stealing it
        self.pin_frame(frame_id);

        // 3. Double check page_table (race condition check)
        if let Some(frame_ref) = self.lookup(shard, page_id) {
            shard.free_frame(frame_id);
            self.unpin(frame_id);
            return Ok(frame_ref);
        }

        // 4. Load data
        let local_frame_id = shard.local_frame_id(frame_id);
        {
            let slot = &shard.frames[local_frame_id];
            let mut data_guard = slot.data.write();

            // Execute loader with mutable access to the buffer
            match loader(&mut data_guard) {
                Ok(_) => {}
                Err(e) => {
                    drop(data_guard);
                    shard.free_frame(frame_id);
                    self.unpin(frame_id);
                    return Err(e);
                }
            }

            // Install metadata
            let mut pid_guard = slot.header.page_id.write();
            *pid_guard = Some(page_id);
            slot.header.is_dirty.store(false, Ordering::SeqCst);
        }

        // 5. Publish to page table
        shard.page_table.insert(page_id, frame_id);

        // 6. Mark access for eviction policy (local frame ID)
        shard.eviction.access(local_frame_id);

        // 7. Capture data pointer for lock-free access
        let (data_ptr, data_len) = {
            let data_guard = shard.frames[local_frame_id].data.read();
            (data_guard.as_ptr(), data_guard.len())
        };

        Ok(FrameRef {
            pool: self.clone(),
            page_id,
            frame_id,
            data_ptr,
            data_len,
        })
    }

    fn lookup(self: &Arc<Self>, shard: &BufferShard, page_id: PageId) -> Option<FrameRef> {
        if let Some(entry) = shard.page_table.get(&page_id) {
            let frame_id = *entry.value();
            let local_frame_id = shard.local_frame_id(frame_id);

            // Optimistic pin
            self.pin_frame(frame_id);

            // Verify it's still the right page (eviction race check)
            let slot = &shard.frames[local_frame_id];
            let current_pid = slot.header.page_id.read();
            if *current_pid == Some(page_id) {
                shard.eviction.access(local_frame_id);

                // Capture data pointer for lock-free access
                let (data_ptr, data_len) = {
                    let data_guard = slot.data.read();
                    (data_guard.as_ptr(), data_guard.len())
                };

                return Some(FrameRef {
                    pool: self.clone(),
                    page_id,
                    frame_id,
                    data_ptr,
                    data_len,
                });
            }

            // Wrong page (was evicted and repurposed before we pinned)
            self.unpin(frame_id);
        }
        None
    }

    fn pin_frame(&self, frame_id: FrameId) {
        let shard = self.get_shard_by_frame(frame_id);
        let local_id = shard.local_frame_id(frame_id);
        shard.frames[local_id]
            .header
            .pin_count
            .fetch_add(1, Ordering::SeqCst);
    }

    fn unpin(&self, frame_id: FrameId) {
        let shard = self.get_shard_by_frame(frame_id);
        let local_id = shard.local_frame_id(frame_id);
        shard.frames[local_id]
            .header
            .pin_count
            .fetch_sub(1, Ordering::SeqCst);
    }

    fn get_frame_data(&self, frame_id: FrameId) -> parking_lot::RwLockReadGuard<'_, Vec<u8>> {
        let shard = self.get_shard_by_frame(frame_id);
        let local_id = shard.local_frame_id(frame_id);
        shard.frames[local_id].data.read()
    }

    // Helper to create buffer pool error from loader error type
    fn make_capacity_error<E>(&self) -> E
    where
        E: From<crate::buffer::BufferPoolError>,
    {
        E::from(crate::buffer::BufferPoolError::PoolFull)
    }
}
