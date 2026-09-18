#!/usr/bin/env bash
# The provider: the middle layers, its queues, and the gateway that serves the
# clients' connections (Sections 4.3 and 5).
#
#   deploy/provider.sh --model /path/to/checkpoint --listen 0.0.0.0:50151
#
# Two processes come up: the Rust gateway, which owns the socket, and the
# engine, which owns the GPU. They meet in two shared-memory rings.
set -euo pipefail

MODEL="tiny"
LISTEN="0.0.0.0:50151"
SHM="${SHM_DIR:-.shm/provider}"
WINDOW=24
PASSENGERS=4
POLICY="opportunistic"
DEVICE="cuda:0"
TLS_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --listen) LISTEN="$2"; shift 2 ;;
    --shm) SHM="$2"; shift 2 ;;
    --window) WINDOW="$2"; shift 2 ;;
    --passengers) PASSENGERS="$2"; shift 2 ;;
    --policy) POLICY="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --tls-dir) TLS_ARGS=(--tls-ca "$2/ca.pem" --tls-cert "$2/provider.pem" --tls-key "$2/provider.key"); shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$SHM"
GATEWAY="gateway/target/release/slipstream-gateway"
if [[ ! -x "$GATEWAY" ]]; then
  echo "building the gateway"
  (cd gateway && cargo build --release)
fi

echo "gateway on $LISTEN, rings in $SHM"
"$GATEWAY" provider \
  --listen "$LISTEN" \
  --up "$SHM/up.ring" --down "$SHM/down.ring" \
  --create-rings --window "$WINDOW" "${TLS_ARGS[@]}" &
GATEWAY_PID=$!
trap 'kill $GATEWAY_PID 2>/dev/null || true' EXIT

# Give the gateway a moment to create the rings before the engine opens them.
sleep 1

python3 -m slipstream.provider.main \
  --model "$MODEL" --device "$DEVICE" \
  --shm "$SHM" --window "$WINDOW" --passengers "$PASSENGERS" --policy "$POLICY"
