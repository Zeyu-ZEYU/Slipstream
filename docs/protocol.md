# The wire protocol

Three messages share one persistent connection per client. `proto/slipstream.proto`
is the definition; this file is the reading of it (Section 5 and Appendix C).

## Upstream: hidden states and verdicts

A hidden state goes up with

| Field | Why |
| --- | --- |
| `request_id`, `round_id` | which round it belongs to |
| `position` | its place in the round's tree; 0 is the root |
| `parent` | its parent's position, which is the only structure the provider is told |
| `first_segment_end` | the one marker the client sets: the provider runs the segment as one batch when it arrives |
| `traffic_class` | first segment or background, so the gateway keeps the classes apart |
| `payload_precision`, `output_precision` | what this state is encoded at, and what the client wants back |
| `payload`, `num_elements` | the obfuscated hidden state |
| `commits_round_id` | the round this state closes, if any |

A verdict carries the identifiers and the newly accepted positions, plus
`continue_round` when the round goes on and `end_of_request` at the end. It
carries no token and no probability.

## Downstream: outputs

An output comes back with the identifiers, its position, the window `W`, the
count of hidden states received so far, and the provider's timestamps for the
arrival and for the batch that ran it. The client differences consecutive
arrival timestamps to estimate the uplink time per hidden state, which needs
no clock synchronization.

## Closing a round

Closing needs no verdict. The next round's first hidden state carries a new
round identifier and a parent pointer to the accepted leaf, and the provider
commits that leaf's ancestors, appending their key and value entries and
keeping the leaf's recurrent state, and releases the rest. A round's state
expires after about one round time, and hidden states of an expired round are
dropped.

## Shared memory between the engine and the gateway

The engine never touches a socket. It publishes records into a ring, one at a
time, in stream order; the gateway drains the ring and carries the frames.
`gateway/src/ring.rs` documents the layout byte for byte,
`slipstream/transport/shm.py` writes the same layout from Python, and
`tests/test_ring_layout.py` checks that the two agree on every offset.

A full ring is back pressure: it is what keeps the engine from running ahead
of the link, and it is why the scheduler hands out one hidden state at a time
and keeps the unsent buffer below one hidden state (Section 4.4).

## Precision on the wire

| Code | Name | Bytes per element | Layout |
| --- | --- | --- | --- |
| 1 | `bf16` | 2 | little endian |
| 2 | `fp8` | 1 | E4M3 |
| 3 | `fp4` | 0.5 plus a scale per block | one float16 scale per block of 32, then packed E2M1 codes, low nibble first |

`slipstream/codec.py` encodes and decodes all three, and
`payload_bytes` is what admission prices a background token by.
