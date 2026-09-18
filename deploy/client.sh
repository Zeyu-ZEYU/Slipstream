#!/usr/bin/env bash
# One client: the first layer, the language-model head, the drafter, and the
# stream scheduler, plus the gateway that holds its one connection to the
# provider (Sections 4.2, 4.4 and 5).
#
#   deploy/client.sh --model /path/to/checkpoint --connect provider.example:50151 \
#                    --requests 50 --method slipstream
set -euo pipefail

MODEL="tiny"
CONNECT="127.0.0.1:50151"
SHM="${SHM_DIR:-.shm/client0}"
METHOD="slipstream"
REQUESTS=8
MAX_TOKENS=256
WORKLOAD="synthetic"
DEVICE="cuda:0"
TLS_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --connect) CONNECT="$2"; shift 2 ;;
    --shm) SHM="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --requests) REQUESTS="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --workload) WORKLOAD="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --tls-dir) TLS_ARGS=(--tls-ca "$2/ca.pem" --tls-cert "$2/client.pem" --tls-key "$2/client.key" --tls-domain provider); shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$SHM"
GATEWAY="gateway/target/release/slipstream-gateway"
if [[ ! -x "$GATEWAY" ]]; then
  (cd gateway && cargo build --release)
fi

"$GATEWAY" client \
  --connect "$CONNECT" \
  --up "$SHM/up.ring" --down "$SHM/down.ring" \
  --create-rings "${TLS_ARGS[@]}" &
GATEWAY_PID=$!
trap 'kill $GATEWAY_PID 2>/dev/null || true' EXIT
sleep 1

python3 -m slipstream.client.main \
  --model "$MODEL" --device "$DEVICE" --shm "$SHM" \
  --method "$METHOD" --requests "$REQUESTS" \
  --max-output-tokens "$MAX_TOKENS" --workload "$WORKLOAD"
