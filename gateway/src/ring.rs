//! The shared-memory ring between an engine and its gateway.
//!
//! Section 5: "the engine hands hidden states to a Rust gateway over shared
//! memory, one at a time, in stream order". The ring is that hand-off. It is
//! single producer, single consumer, and its layout is fixed so the Python
//! engine (`slipstream/transport/shm.py`) and this gateway agree byte for
//! byte.
//!
//! File layout, little endian throughout:
//!
//! ```text
//! offset  size  field
//! 0       4     magic, 0x534C5031 ("SLP1")
//! 4       4     version, 1
//! 8       4     slot count
//! 12      4     slot size in bytes
//! 16      8     write index, the producer's next slot
//! 24      8     read index, the consumer's next slot
//! 32      32    reserved
//! 64      ...   slots, `slot count` of `slot size` bytes each
//! ```
//!
//! Each slot holds a 64-byte record header and then the payload:
//!
//! ```text
//! offset  size  field
//! 0       8     sequence number, nonzero once the slot is published
//! 8       4     kind: 1 hidden state, 2 verdict, 3 output
//! 12      4     flags: bit 0 end of first segment, bit 1 background class,
//!               bit 2 continue a round, bit 3 end of request
//! 16      8     request id
//! 24      8     round id
//! 32      4     position
//! 36      4     parent
//! 40      4     payload precision: 1 bf16, 2 fp8, 3 fp4
//! 44      4     output precision
//! 48      4     element count
//! 52      4     payload length in bytes
//! 56      8     round this record commits, or zero
//! ```
//!
//! A verdict carries its accepted positions in the payload as little-endian
//! `u32`s, so one record shape serves all three messages.

use anyhow::{bail, Context, Result};
use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

pub const MAGIC: u32 = 0x534C_5031;
pub const VERSION: u32 = 1;
pub const HEADER_BYTES: usize = 64;
pub const RECORD_HEADER_BYTES: usize = 64;

pub const KIND_HIDDEN_STATE: u32 = 1;
pub const KIND_VERDICT: u32 = 2;
pub const KIND_OUTPUT: u32 = 3;

pub const FLAG_FIRST_SEGMENT_END: u32 = 1 << 0;
pub const FLAG_BACKGROUND: u32 = 1 << 1;
pub const FLAG_CONTINUE: u32 = 1 << 2;
pub const FLAG_END_OF_REQUEST: u32 = 1 << 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    pub kind: u32,
    pub flags: u32,
    pub request_id: u64,
    pub round_id: u64,
    pub position: u32,
    pub parent: u32,
    pub payload_precision: u32,
    pub output_precision: u32,
    pub num_elements: u32,
    pub commits_round_id: u64,
    pub payload: Vec<u8>,
}

impl Record {
    pub fn is_first_segment(&self) -> bool {
        self.flags & FLAG_BACKGROUND == 0
    }

    pub fn ends_first_segment(&self) -> bool {
        self.flags & FLAG_FIRST_SEGMENT_END != 0
    }

    /// Accepted positions of a verdict, which travel in the payload.
    pub fn accepted_positions(&self) -> Vec<u32> {
        self.payload
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

/// One direction of the hand-off.
pub struct Ring {
    map: MmapMut,
    slots: usize,
    slot_bytes: usize,
    /// The consumer's cursor, kept locally so a reader never writes to the
    /// producer's fields.
    cursor: u64,
}

impl Ring {
    /// Create or truncate a ring file.
    pub fn create(path: impl AsRef<Path>, slots: usize, slot_bytes: usize) -> Result<Self> {
        let size = HEADER_BYTES + slots * slot_bytes;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.as_ref())
            .with_context(|| format!("creating ring {}", path.as_ref().display()))?;
        file.set_len(size as u64)?;
        let mut map = unsafe { MmapMut::map_mut(&file)? };
        map[..4].copy_from_slice(&MAGIC.to_le_bytes());
        map[4..8].copy_from_slice(&VERSION.to_le_bytes());
        map[8..12].copy_from_slice(&(slots as u32).to_le_bytes());
        map[12..16].copy_from_slice(&(slot_bytes as u32).to_le_bytes());
        map[16..24].copy_from_slice(&0u64.to_le_bytes());
        map[24..32].copy_from_slice(&0u64.to_le_bytes());
        map.flush()?;
        Ok(Self { map, slots, slot_bytes, cursor: 0 })
    }

    /// Open a ring another process created.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path.as_ref())
            .with_context(|| format!("opening ring {}", path.as_ref().display()))?;
        let map = unsafe { MmapMut::map_mut(&file)? };
        if map.len() < HEADER_BYTES {
            bail!("ring file is too small");
        }
        let magic = u32::from_le_bytes(map[0..4].try_into()?);
        let version = u32::from_le_bytes(map[4..8].try_into()?);
        if magic != MAGIC || version != VERSION {
            bail!("not a slipstream ring: magic {magic:#x} version {version}");
        }
        let slots = u32::from_le_bytes(map[8..12].try_into()?) as usize;
        let slot_bytes = u32::from_le_bytes(map[12..16].try_into()?) as usize;
        let cursor = u64::from_le_bytes(map[24..32].try_into()?);
        Ok(Self { map, slots, slot_bytes, cursor })
    }

    fn write_index(&self) -> u64 {
        let raw = unsafe { &*(self.map[16..24].as_ptr() as *const AtomicU64) };
        raw.load(Ordering::Acquire)
    }

    fn set_write_index(&mut self, value: u64) {
        let raw = unsafe { &*(self.map[16..24].as_ptr() as *const AtomicU64) };
        raw.store(value, Ordering::Release);
    }

    fn read_index(&self) -> u64 {
        let raw = unsafe { &*(self.map[24..32].as_ptr() as *const AtomicU64) };
        raw.load(Ordering::Acquire)
    }

    fn set_read_index(&mut self, value: u64) {
        let raw = unsafe { &*(self.map[24..32].as_ptr() as *const AtomicU64) };
        raw.store(value, Ordering::Release);
    }

    /// Slots in the ring, which bounds what the engine may run ahead by.
    pub fn capacity(&self) -> usize {
        self.slots
    }

    pub fn max_payload(&self) -> usize {
        self.slot_bytes - RECORD_HEADER_BYTES
    }

    /// Publish one record. Returns false when the ring is full, which is the
    /// back pressure that keeps the engine from running ahead of the link.
    pub fn push(&mut self, record: &Record) -> Result<bool> {
        if record.payload.len() > self.max_payload() {
            bail!(
                "payload of {} bytes exceeds the slot's {}",
                record.payload.len(),
                self.max_payload()
            );
        }
        let write = self.write_index();
        let read = self.read_index();
        if write - read >= self.slots as u64 {
            return Ok(false);
        }
        let slot = (write % self.slots as u64) as usize;
        let base = HEADER_BYTES + slot * self.slot_bytes;
        let header = &mut self.map[base..base + RECORD_HEADER_BYTES];
        header.fill(0);
        header[0..8].copy_from_slice(&(write + 1).to_le_bytes());
        header[8..12].copy_from_slice(&record.kind.to_le_bytes());
        header[12..16].copy_from_slice(&record.flags.to_le_bytes());
        header[16..24].copy_from_slice(&record.request_id.to_le_bytes());
        header[24..32].copy_from_slice(&record.round_id.to_le_bytes());
        header[32..36].copy_from_slice(&record.position.to_le_bytes());
        header[36..40].copy_from_slice(&record.parent.to_le_bytes());
        header[40..44].copy_from_slice(&record.payload_precision.to_le_bytes());
        header[44..48].copy_from_slice(&record.output_precision.to_le_bytes());
        header[48..52].copy_from_slice(&record.num_elements.to_le_bytes());
        header[52..56].copy_from_slice(&(record.payload.len() as u32).to_le_bytes());
        header[56..64].copy_from_slice(&record.commits_round_id.to_le_bytes());
        let start = base + RECORD_HEADER_BYTES;
        self.map[start..start + record.payload.len()].copy_from_slice(&record.payload);
        self.set_write_index(write + 1);
        Ok(true)
    }

    /// Take the next record, if the producer has published one.
    pub fn pop(&mut self) -> Result<Option<Record>> {
        let write = self.write_index();
        if self.cursor >= write {
            return Ok(None);
        }
        let slot = (self.cursor % self.slots as u64) as usize;
        let base = HEADER_BYTES + slot * self.slot_bytes;
        let header = &self.map[base..base + RECORD_HEADER_BYTES];
        let sequence = u64::from_le_bytes(header[0..8].try_into()?);
        if sequence == 0 {
            return Ok(None);
        }
        let payload_len = u32::from_le_bytes(header[52..56].try_into()?) as usize;
        let start = base + RECORD_HEADER_BYTES;
        let record = Record {
            kind: u32::from_le_bytes(header[8..12].try_into()?),
            flags: u32::from_le_bytes(header[12..16].try_into()?),
            request_id: u64::from_le_bytes(header[16..24].try_into()?),
            round_id: u64::from_le_bytes(header[24..32].try_into()?),
            position: u32::from_le_bytes(header[32..36].try_into()?),
            parent: u32::from_le_bytes(header[36..40].try_into()?),
            payload_precision: u32::from_le_bytes(header[40..44].try_into()?),
            output_precision: u32::from_le_bytes(header[44..48].try_into()?),
            num_elements: u32::from_le_bytes(header[48..52].try_into()?),
            commits_round_id: u64::from_le_bytes(header[56..64].try_into()?),
            payload: self.map[start..start + payload_len].to_vec(),
        };
        self.cursor += 1;
        self.set_read_index(self.cursor);
        Ok(Some(record))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(position: u32, payload: Vec<u8>) -> Record {
        Record {
            kind: KIND_HIDDEN_STATE,
            flags: FLAG_FIRST_SEGMENT_END,
            request_id: 7,
            round_id: 3,
            position,
            parent: position.saturating_sub(1),
            payload_precision: 1,
            output_precision: 1,
            num_elements: (payload.len() / 2) as u32,
            commits_round_id: 0,
            payload,
        }
    }

    #[test]
    fn round_trips_records_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ring");
        let mut writer = Ring::create(&path, 8, 1024).unwrap();
        let mut reader = Ring::open(&path).unwrap();
        for position in 0..5u32 {
            assert!(writer.push(&record(position, vec![position as u8; 16])).unwrap());
        }
        for position in 0..5u32 {
            let got = reader.pop().unwrap().unwrap();
            assert_eq!(got.position, position);
            assert_eq!(got.payload, vec![position as u8; 16]);
            assert!(got.ends_first_segment());
            assert!(got.is_first_segment());
        }
        assert!(reader.pop().unwrap().is_none());
    }

    #[test]
    fn reports_its_capacity() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ring");
        let ring = Ring::create(&path, 4, 512).unwrap();
        assert_eq!(ring.capacity(), 4);
        assert_eq!(ring.max_payload(), 512 - RECORD_HEADER_BYTES);
    }

    #[test]
    fn reports_back_pressure_when_full() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ring");
        let mut writer = Ring::create(&path, 2, 256).unwrap();
        assert!(writer.push(&record(0, vec![1, 2])).unwrap());
        assert!(writer.push(&record(1, vec![3, 4])).unwrap());
        assert!(!writer.push(&record(2, vec![5, 6])).unwrap());
    }

    #[test]
    fn rejects_an_oversized_payload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ring");
        let mut writer = Ring::create(&path, 2, 128).unwrap();
        assert!(writer.push(&record(0, vec![0u8; 4096])).is_err());
    }

    #[test]
    fn verdict_positions_travel_in_the_payload() {
        let mut payload = Vec::new();
        for position in [1u32, 4, 9] {
            payload.extend_from_slice(&position.to_le_bytes());
        }
        let mut verdict = record(0, payload);
        verdict.kind = KIND_VERDICT;
        assert_eq!(verdict.accepted_positions(), vec![1, 4, 9]);
    }
}
