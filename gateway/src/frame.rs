//! Records on the ring become protocol-buffer frames on the wire, and back.
//!
//! Nothing here interprets a payload: a hidden state is opaque bytes at the
//! precision the client chose, and a verdict is a list of positions. The
//! gateway carries structure, never text.

use crate::pb;
use crate::ring::{
    Record, FLAG_BACKGROUND, FLAG_CONTINUE, FLAG_END_OF_REQUEST, FLAG_FIRST_SEGMENT_END,
    KIND_HIDDEN_STATE, KIND_OUTPUT, KIND_VERDICT,
};

fn precision_to_proto(value: u32) -> i32 {
    match value {
        2 => pb::Precision::Fp8 as i32,
        3 => pb::Precision::Fp4 as i32,
        _ => pb::Precision::Bf16 as i32,
    }
}

fn precision_from_proto(value: i32) -> u32 {
    match pb::Precision::try_from(value) {
        Ok(pb::Precision::Fp8) => 2,
        Ok(pb::Precision::Fp4) => 3,
        _ => 1,
    }
}

fn class_to_proto(record: &Record) -> i32 {
    if record.is_first_segment() {
        pb::TrafficClass::FirstSegment as i32
    } else {
        pb::TrafficClass::Background as i32
    }
}

/// One upstream frame: a hidden state or a verdict.
pub fn record_to_upstream(record: &Record, now_ns: u64) -> Option<pb::Upstream> {
    match record.kind {
        KIND_HIDDEN_STATE => Some(pb::Upstream {
            frame: Some(pb::upstream::Frame::HiddenState(pb::HiddenState {
                request_id: record.request_id,
                round_id: record.round_id,
                position: record.position,
                parent: record.parent,
                first_segment_end: record.ends_first_segment(),
                traffic_class: class_to_proto(record),
                payload_precision: precision_to_proto(record.payload_precision),
                output_precision: precision_to_proto(record.output_precision),
                payload: record.payload.clone(),
                num_elements: record.num_elements,
                send_time_ns: now_ns,
                commits_round_id: record.commits_round_id,
            })),
        }),
        KIND_VERDICT => Some(pb::Upstream {
            frame: Some(pb::upstream::Frame::Verdict(pb::Verdict {
                request_id: record.request_id,
                round_id: record.round_id,
                accepted: record.accepted_positions(),
                continue_round: record.flags & FLAG_CONTINUE != 0,
                end_of_request: record.flags & FLAG_END_OF_REQUEST != 0,
                send_time_ns: now_ns,
            })),
        }),
        _ => None,
    }
}

/// An arriving hidden state becomes a record for the provider's engine.
pub fn hidden_state_to_record(message: &pb::HiddenState) -> Record {
    let mut flags = 0;
    if message.first_segment_end {
        flags |= FLAG_FIRST_SEGMENT_END;
    }
    if message.traffic_class == pb::TrafficClass::Background as i32 {
        flags |= FLAG_BACKGROUND;
    }
    Record {
        kind: KIND_HIDDEN_STATE,
        flags,
        request_id: message.request_id,
        round_id: message.round_id,
        position: message.position,
        parent: message.parent,
        payload_precision: precision_from_proto(message.payload_precision),
        output_precision: precision_from_proto(message.output_precision),
        num_elements: message.num_elements,
        commits_round_id: message.commits_round_id,
        payload: message.payload.clone(),
    }
}

pub fn verdict_to_record(message: &pb::Verdict) -> Record {
    let mut flags = 0;
    if message.continue_round {
        flags |= FLAG_CONTINUE;
    }
    if message.end_of_request {
        flags |= FLAG_END_OF_REQUEST;
    }
    let mut payload = Vec::with_capacity(message.accepted.len() * 4);
    for position in &message.accepted {
        payload.extend_from_slice(&position.to_le_bytes());
    }
    Record {
        kind: KIND_VERDICT,
        flags,
        request_id: message.request_id,
        round_id: message.round_id,
        position: 0,
        parent: 0,
        payload_precision: 1,
        output_precision: 1,
        num_elements: message.accepted.len() as u32,
        commits_round_id: 0,
        payload,
    }
}

/// An output the provider's engine produced becomes a downstream frame.
pub fn record_to_output(record: &Record, window: u32, received: u64, now_ns: u64) -> pb::Output {
    pb::Output {
        request_id: record.request_id,
        round_id: record.round_id,
        position: record.position,
        payload_precision: precision_to_proto(record.payload_precision),
        payload: record.payload.clone(),
        num_elements: record.num_elements,
        window,
        received_count: received,
        arrival_time_ns: record.commits_round_id,
        batch_start_time_ns: now_ns,
        traffic_class: class_to_proto(record),
    }
}

/// An arriving output becomes a record for the client's engine.
pub fn output_to_record(message: &pb::Output) -> Record {
    let mut flags = 0;
    if message.traffic_class == pb::TrafficClass::Background as i32 {
        flags |= FLAG_BACKGROUND;
    }
    Record {
        kind: KIND_OUTPUT,
        flags,
        request_id: message.request_id,
        round_id: message.round_id,
        position: message.position,
        parent: message.window,
        payload_precision: precision_from_proto(message.payload_precision),
        output_precision: precision_from_proto(message.payload_precision),
        num_elements: message.num_elements,
        commits_round_id: message.arrival_time_ns,
        payload: message.payload.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::FLAG_FIRST_SEGMENT_END;

    #[test]
    fn a_hidden_state_survives_the_round_trip() {
        let record = Record {
            kind: KIND_HIDDEN_STATE,
            flags: FLAG_FIRST_SEGMENT_END,
            request_id: 11,
            round_id: 5,
            position: 4,
            parent: 2,
            payload_precision: 2,
            output_precision: 3,
            num_elements: 4096,
            commits_round_id: 4,
            payload: vec![9u8; 64],
        };
        let upstream = record_to_upstream(&record, 1234).unwrap();
        let message = match upstream.frame.unwrap() {
            pb::upstream::Frame::HiddenState(m) => m,
            _ => panic!("expected a hidden state"),
        };
        assert_eq!(message.position, 4);
        assert_eq!(message.parent, 2);
        assert!(message.first_segment_end);
        assert_eq!(message.payload_precision, pb::Precision::Fp8 as i32);
        assert_eq!(message.output_precision, pb::Precision::Fp4 as i32);
        let back = hidden_state_to_record(&message);
        assert_eq!(back.payload, record.payload);
        assert_eq!(back.payload_precision, 2);
        assert_eq!(back.commits_round_id, 4);
    }

    #[test]
    fn a_verdict_carries_only_positions() {
        let verdict = pb::Verdict {
            request_id: 3,
            round_id: 9,
            accepted: vec![1, 5, 6],
            continue_round: true,
            end_of_request: false,
            send_time_ns: 0,
        };
        let record = verdict_to_record(&verdict);
        assert_eq!(record.accepted_positions(), vec![1, 5, 6]);
        assert_eq!(record.payload.len(), 12);
        assert_ne!(record.flags & FLAG_CONTINUE, 0);
    }
}
