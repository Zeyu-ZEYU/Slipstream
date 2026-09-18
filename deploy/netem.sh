#!/usr/bin/env bash
# Shape an interface like the residential link of Section 3.1, and like the 24
# emulated links of Section 6.6: 20 Mbps up, 100 Mbps down, 20 ms each way.
#
#   sudo deploy/netem.sh up eth0          # shape
#   sudo deploy/netem.sh down eth0        # remove
#   sudo deploy/netem.sh up eth0 --uplink 20mbit --downlink 100mbit --delay 20ms
#
# Egress is shaped directly; ingress is shaped through an ifb device, which is
# the usual way to rate-limit inbound traffic with tc.
set -euo pipefail

ACTION="${1:-up}"
IFACE="${2:-eth0}"
shift 2 || true

UPLINK="20mbit"
DOWNLINK="100mbit"
DELAY="20ms"
IFB="ifb0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --uplink) UPLINK="$2"; shift 2 ;;
    --downlink) DOWNLINK="$2"; shift 2 ;;
    --delay) DELAY="$2"; shift 2 ;;
    --ifb) IFB="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

teardown() {
  tc qdisc del dev "$IFACE" root 2>/dev/null || true
  tc qdisc del dev "$IFACE" ingress 2>/dev/null || true
  tc qdisc del dev "$IFB" root 2>/dev/null || true
  ip link set dev "$IFB" down 2>/dev/null || true
}

case "$ACTION" in
  down)
    teardown
    echo "removed the shaping on $IFACE"
    ;;
  up)
    teardown
    modprobe ifb numifbs=1 2>/dev/null || true
    ip link add "$IFB" type ifb 2>/dev/null || true
    ip link set dev "$IFB" up

    # Egress: the client's uplink, the leg the first segment waits on.
    tc qdisc add dev "$IFACE" root handle 1: tbf rate "$UPLINK" burst 32kbit latency 400ms
    tc qdisc add dev "$IFACE" parent 1:1 handle 10: netem delay "$DELAY"

    # Ingress, through the ifb device: the downlink.
    tc qdisc add dev "$IFACE" handle ffff: ingress
    tc filter add dev "$IFACE" parent ffff: protocol ip u32 match u32 0 0 \
      action mirred egress redirect dev "$IFB"
    tc qdisc add dev "$IFB" root handle 1: tbf rate "$DOWNLINK" burst 32kbit latency 400ms
    tc qdisc add dev "$IFB" parent 1:1 handle 10: netem delay "$DELAY"

    echo "$IFACE shaped: $UPLINK up, $DOWNLINK down, $DELAY each way"
    tc -s qdisc show dev "$IFACE" | head -4
    ;;
  *)
    echo "usage: $0 {up|down} IFACE [options]" >&2
    exit 2
    ;;
esac
