use rustc_hash::FxHashMap;
use std::borrow::Cow;
#[cfg(unix)]
use std::ptr::NonNull;

const UPPER_NEIGHBORS_REGION_VERSION: u32 = 1;
const UPPER_NEIGHBORS_HEADER_SIZE: usize = 16;
const UPPER_NEIGHBORS_ENTRY_SIZE: usize = 16;

#[derive(Debug)]
pub(crate) enum UpperNeighborsStorage {
    Owned(FxHashMap<u32, Vec<Vec<u32>>>),
    #[cfg(feature = "mmap")]
    Mmap(UpperNeighborsMmap),
}

impl Default for UpperNeighborsStorage {
    fn default() -> Self {
        Self::Owned(FxHashMap::default())
    }
}

impl UpperNeighborsStorage {
    pub(crate) fn upper_level_count(&self, id: u32) -> usize {
        match self {
            Self::Owned(map) => map.get(&id).map_or(0, Vec::len),
            #[cfg(feature = "mmap")]
            Self::Mmap(mmap) => mmap.upper_level_count(id),
        }
    }

    pub(crate) fn nodes_with_upper(&self) -> usize {
        match self {
            Self::Owned(map) => map.len(),
            #[cfg(feature = "mmap")]
            Self::Mmap(mmap) => mmap.nodes_with_upper,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.nodes_with_upper() == 0
    }

    pub(crate) fn memory_usage(&self) -> usize {
        match self {
            Self::Owned(map) => map
                .values()
                .map(|levels: &Vec<Vec<u32>>| {
                    levels.iter().map(|v| v.len() * 4).sum::<usize>()
                        + levels.len() * std::mem::size_of::<Vec<u32>>()
                })
                .sum(),
            #[cfg(feature = "mmap")]
            Self::Mmap(mmap) => mmap.mmap.len(),
        }
    }

    pub(crate) fn allocate_upper_levels(&mut self, id: u32, level: u8) {
        if level == 0 {
            return;
        }

        let needed_levels = level as usize;
        match self {
            Self::Owned(map) => match map.entry(id) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert((0..needed_levels).map(|_| Vec::new()).collect());
                }
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    if entry.get().len() < needed_levels {
                        entry.get_mut().resize(needed_levels, Vec::new());
                    }
                }
            },
            #[cfg(feature = "mmap")]
            Self::Mmap(_) => panic!("Cannot mutate mmap-backed upper neighbors"),
        }
    }

    pub(crate) fn set_neighbors_at_level(&mut self, id: u32, level: u8, neighbors: Vec<u32>) {
        if level == 0 {
            return;
        }

        self.allocate_upper_levels(id, level);
        match self {
            Self::Owned(map) => {
                if let Some(levels) = map.get_mut(&id) {
                    let level_idx = level as usize - 1;
                    if level_idx < levels.len() {
                        levels[level_idx] = neighbors;
                    }
                }
            }
            #[cfg(feature = "mmap")]
            Self::Mmap(_) => panic!("Cannot mutate mmap-backed upper neighbors"),
        }
    }

    pub(crate) fn add_neighbor(
        &mut self,
        id: u32,
        level: u8,
        neighbor: u32,
        max_neighbors_upper: usize,
    ) {
        if level == 0 {
            return;
        }

        self.allocate_upper_levels(id, level);
        match self {
            Self::Owned(map) => {
                if let Some(levels) = map.get_mut(&id) {
                    let level_idx = level as usize - 1;
                    if level_idx < levels.len() && levels[level_idx].len() < max_neighbors_upper {
                        levels[level_idx].push(neighbor);
                    }
                }
            }
            #[cfg(feature = "mmap")]
            Self::Mmap(_) => panic!("Cannot mutate mmap-backed upper neighbors"),
        }
    }

    pub(crate) fn neighbors_at_level_cow(
        &self,
        id: u32,
        level: u8,
        max_neighbors_upper: usize,
    ) -> Cow<'_, [u32]> {
        match self {
            Self::Owned(map) => match map.get(&id) {
                Some(levels) => {
                    let level_idx = level as usize - 1;
                    if level_idx < levels.len() {
                        Cow::Borrowed(&levels[level_idx])
                    } else {
                        Cow::Borrowed(&[])
                    }
                }
                None => Cow::Borrowed(&[]),
            },
            #[cfg(feature = "mmap")]
            Self::Mmap(mmap) => mmap
                .neighbors_at_level_cow(id, level, max_neighbors_upper)
                .unwrap_or(Cow::Borrowed(&[])),
        }
    }

    pub(crate) fn take_owned(&mut self) -> FxHashMap<u32, Vec<Vec<u32>>> {
        match self {
            Self::Owned(map) => std::mem::take(map),
            #[cfg(feature = "mmap")]
            Self::Mmap(_) => panic!("Cannot take ownership of mmap-backed upper neighbors"),
        }
    }

    pub(crate) fn replace_owned(&mut self, map: FxHashMap<u32, Vec<Vec<u32>>>) {
        *self = Self::Owned(map);
    }

    pub(crate) fn serialize_region(&self, node_count: usize) -> Vec<u8> {
        if self.is_empty() {
            return Vec::new();
        }

        let payload_size: usize = (0..node_count as u32)
            .map(|node_id| {
                let levels = self.upper_level_count(node_id);
                (1..=levels)
                    .map(|level| {
                        4 + self
                            .neighbors_at_level_cow(node_id, level as u8, usize::MAX)
                            .len()
                            * 4
                    })
                    .sum::<usize>()
            })
            .sum();

        let payload_offset = UPPER_NEIGHBORS_HEADER_SIZE + node_count * UPPER_NEIGHBORS_ENTRY_SIZE;
        let mut out = Vec::with_capacity(payload_offset + payload_size);
        out.extend_from_slice(&UPPER_NEIGHBORS_REGION_VERSION.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(node_count as u64).to_le_bytes());
        out.resize(payload_offset, 0);

        for node_id in 0..node_count as u32 {
            let level_count = self.upper_level_count(node_id) as u32;
            let entry_offset =
                UPPER_NEIGHBORS_HEADER_SIZE + node_id as usize * UPPER_NEIGHBORS_ENTRY_SIZE;
            let rel_offset = if level_count == 0 {
                0
            } else {
                (out.len() - payload_offset) as u64
            };
            out[entry_offset..entry_offset + 8].copy_from_slice(&rel_offset.to_le_bytes());
            out[entry_offset + 8..entry_offset + 12].copy_from_slice(&level_count.to_le_bytes());

            for level in 1..=level_count as u8 {
                let neighbors = self.neighbors_at_level_cow(node_id, level, usize::MAX);
                out.extend_from_slice(&(neighbors.len() as u32).to_le_bytes());
                for &neighbor in neighbors.iter() {
                    out.extend_from_slice(&neighbor.to_le_bytes());
                }
            }
        }

        out
    }

    pub(crate) fn deserialize_region_owned(
        data: &[u8],
        expected_nodes: usize,
        max_neighbors_upper: usize,
    ) -> Result<Self, String> {
        let (node_count, payload_offset) = parse_region_layout(data, expected_nodes)?;
        let mut upper_neighbors = FxHashMap::default();

        for node_id in 0..node_count as u32 {
            let (entry_offset, level_count) = read_entry(data, node_id as usize)?;
            if level_count == 0 {
                continue;
            }

            let mut cursor = payload_offset
                .checked_add(entry_offset as usize)
                .ok_or_else(|| "Upper-neighbor payload offset overflow".to_string())?;
            let mut levels = Vec::with_capacity(level_count as usize);

            for _ in 0..level_count {
                let count: usize = read_u32_at(data, cursor)?
                    .try_into()
                    .map_err(|_| "Upper-neighbor count exceeds usize".to_string())?;
                if count > max_neighbors_upper {
                    return Err(format!(
                        "Upper-neighbor count {count} exceeds max_neighbors_upper {max_neighbors_upper}"
                    ));
                }
                cursor += 4;
                let bytes_len = count
                    .checked_mul(4)
                    .ok_or_else(|| "Upper-neighbor byte length overflow".to_string())?;
                let end = cursor
                    .checked_add(bytes_len)
                    .ok_or_else(|| "Upper-neighbor payload end overflow".to_string())?;
                if end > data.len() {
                    return Err("Upper-neighbor payload extends past region".to_string());
                }
                let mut neighbors = Vec::with_capacity(count);
                for offset in (cursor..end).step_by(4) {
                    neighbors.push(read_u32_at(data, offset)?);
                }
                cursor = end;
                levels.push(neighbors);
            }

            upper_neighbors.insert(node_id, levels);
        }

        Ok(Self::Owned(upper_neighbors))
    }

    #[cfg(feature = "mmap")]
    pub(crate) fn mmap_region(
        mmap: memmap2::Mmap,
        expected_nodes: usize,
        max_neighbors_upper: usize,
    ) -> Result<Self, String> {
        Ok(Self::Mmap(UpperNeighborsMmap::new(
            mmap,
            expected_nodes,
            max_neighbors_upper,
        )?))
    }

    #[cfg(all(test, feature = "mmap"))]
    pub(crate) fn is_mmap(&self) -> bool {
        matches!(self, Self::Mmap(_))
    }
}

#[cfg(feature = "mmap")]
#[derive(Debug)]
pub(crate) struct UpperNeighborsMmap {
    mmap: memmap2::Mmap,
    node_count: usize,
    payload_offset: usize,
    nodes_with_upper: usize,
    max_neighbors_upper: usize,
}

#[cfg(feature = "mmap")]
impl UpperNeighborsMmap {
    fn new(
        mmap: memmap2::Mmap,
        expected_nodes: usize,
        max_neighbors_upper: usize,
    ) -> Result<Self, String> {
        let (node_count, payload_offset) = parse_region_layout(&mmap, expected_nodes)?;
        let mut nodes_with_upper = 0;
        for node_id in 0..node_count {
            let (_, level_count) = read_entry(&mmap, node_id)?;
            if level_count > 0 {
                nodes_with_upper += 1;
            }
        }

        #[cfg(unix)]
        {
            mlock_upper_region(&mmap);
        }

        Ok(Self {
            mmap,
            node_count,
            payload_offset,
            nodes_with_upper,
            max_neighbors_upper,
        })
    }

    fn upper_level_count(&self, id: u32) -> usize {
        if id as usize >= self.node_count {
            return 0;
        }
        read_entry(&self.mmap, id as usize)
            .map(|(_, level_count)| level_count as usize)
            .unwrap_or(0)
    }

    fn neighbors_at_level_cow(
        &self,
        id: u32,
        level: u8,
        _max_neighbors_upper: usize,
    ) -> Option<Cow<'_, [u32]>> {
        let level_idx = level.checked_sub(1)? as usize;
        let (entry_offset, level_count) = read_entry(&self.mmap, id as usize).ok()?;
        if level_idx >= level_count as usize {
            return Some(Cow::Borrowed(&[]));
        }

        let mut cursor = self.payload_offset.checked_add(entry_offset as usize)?;
        for _ in 0..level_idx {
            let count = read_u32_at(&self.mmap, cursor).ok()? as usize;
            if count > self.max_neighbors_upper {
                return None;
            }
            cursor = cursor.checked_add(4 + count * 4)?;
        }

        let count = read_u32_at(&self.mmap, cursor).ok()? as usize;
        if count > self.max_neighbors_upper {
            return None;
        }
        let start = cursor.checked_add(4)?;
        let end = start.checked_add(count * 4)?;
        let bytes = self.mmap.get(start..end)?;
        if bytes.as_ptr().align_offset(std::mem::align_of::<u32>()) == 0 {
            let ptr = bytes.as_ptr() as *const u32;
            Some(Cow::Borrowed(unsafe {
                std::slice::from_raw_parts(ptr, count)
            }))
        } else {
            let mut neighbors = Vec::with_capacity(count);
            for offset in (start..end).step_by(4) {
                neighbors.push(read_u32_at(&self.mmap, offset).ok()?);
            }
            Some(Cow::Owned(neighbors))
        }
    }
}

#[cfg(unix)]
fn mlock_upper_region(mmap: &memmap2::Mmap) {
    use nix::sys::mman::mlock;

    if mmap.is_empty() {
        return;
    }

    let Some(addr) = NonNull::new(mmap.as_ptr() as *mut std::ffi::c_void) else {
        return;
    };

    if let Err(error) = unsafe { mlock(addr, mmap.len()) } {
        tracing::debug!(
            error = %error,
            bytes = mmap.len(),
            "Failed to mlock upper-neighbor region"
        );
    }
}

fn parse_region_layout(data: &[u8], expected_nodes: usize) -> Result<(usize, usize), String> {
    if data.is_empty() {
        if expected_nodes == 0 {
            return Ok((0, UPPER_NEIGHBORS_HEADER_SIZE));
        }
        return Err("Upper-neighbor region missing".to_string());
    }
    if data.len() < UPPER_NEIGHBORS_HEADER_SIZE {
        return Err("Upper-neighbor region too short".to_string());
    }

    let version = read_u32_at(data, 0)?;
    if version != UPPER_NEIGHBORS_REGION_VERSION {
        return Err(format!(
            "Unsupported upper-neighbor region version: {version}"
        ));
    }

    let node_count = read_u64_at(data, 8)?
        .try_into()
        .map_err(|_| "Upper-neighbor node count exceeds usize".to_string())?;
    if node_count != expected_nodes {
        return Err(format!(
            "Upper-neighbor node count mismatch: expected {expected_nodes}, got {node_count}"
        ));
    }

    let payload_offset = UPPER_NEIGHBORS_HEADER_SIZE
        .checked_add(node_count * UPPER_NEIGHBORS_ENTRY_SIZE)
        .ok_or_else(|| "Upper-neighbor table size overflow".to_string())?;
    if payload_offset > data.len() {
        return Err("Upper-neighbor entry table extends past region".to_string());
    }

    Ok((node_count, payload_offset))
}

fn read_entry(data: &[u8], node_id: usize) -> Result<(u64, u32), String> {
    let offset = UPPER_NEIGHBORS_HEADER_SIZE
        .checked_add(node_id * UPPER_NEIGHBORS_ENTRY_SIZE)
        .ok_or_else(|| "Upper-neighbor entry offset overflow".to_string())?;
    let rel_offset = read_u64_at(data, offset)?;
    let level_count = read_u32_at(data, offset + 8)?;
    Ok((rel_offset, level_count))
}

fn read_u32_at(data: &[u8], offset: usize) -> Result<u32, String> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or_else(|| "Upper-neighbor region truncated".to_string())?;
    Ok(u32::from_le_bytes(
        bytes
            .try_into()
            .map_err(|_| "Invalid upper-neighbor u32".to_string())?,
    ))
}

fn read_u64_at(data: &[u8], offset: usize) -> Result<u64, String> {
    let bytes = data
        .get(offset..offset + 8)
        .ok_or_else(|| "Upper-neighbor region truncated".to_string())?;
    Ok(u64::from_le_bytes(
        bytes
            .try_into()
            .map_err(|_| "Invalid upper-neighbor u64".to_string())?,
    ))
}
