#!/usr/bin/env bash
# The run of Section 6.6: eight providers behind a router, 24 clients on
# emulated residential links, K requests in flight per client.
#
#   deploy/cluster/launch_at_scale.sh --hosts deploy/cluster/hosts.example \
#       --model /path/to/checkpoint --k 1 2 4 8 16
#
# What it does on each machine, over ssh:
#   provider node  one gateway and one engine per GPU, ports 50151..50158
#   router node    envoy with deploy/envoy/envoy.yaml
#   client nodes   netem on the interface, then one client per GPU
#
# Every client's traffic is shaped to a residential link, so the cluster's own
# network never binds (Section 6.6).
set -euo pipefail

HOSTS="deploy/cluster/hosts.example"
MODEL="tiny"
KS=(1 2 4 8 16)
IFACE="${IFACE:-eth0}"
REQUESTS=8
MAX_TOKENS=256
METHOD="slipstream"
REMOTE_DIR="${REMOTE_DIR:-~/slipstream}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hosts) HOSTS="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --k) shift; KS=(); while [[ $# -gt 0 && "$1" != --* ]]; do KS+=("$1"); shift; done ;;
    --iface) IFACE="$2"; shift 2 ;;
    --requests) REQUESTS="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

run() {
  local host="$1"; shift
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "ssh $host -- $*"
  else
    # shellcheck disable=SC2029
    ssh "$host" "cd $REMOTE_DIR && $*"
  fi
}

providers=(); clients=(); router=""
while read -r role host gpus; do
  [[ -z "${role:-}" || "$role" == \#* ]] && continue
  case "$role" in
    provider) providers+=("$host:$gpus") ;;
    client) clients+=("$host:$gpus") ;;
    router) router="$host" ;;
  esac
done < "$HOSTS"

echo "providers: ${providers[*]}"
echo "clients:   ${clients[*]}"
echo "router:    ${router:-none}"

# 1. Providers: one gateway and one engine per GPU.
for entry in "${providers[@]}"; do
  host="${entry%%:*}"; gpus="${entry##*:}"
  for ((gpu = 0; gpu < gpus; gpu++)); do
    port=$((50151 + gpu))
    run "$host" "CUDA_VISIBLE_DEVICES=$gpu SHM_DIR=.shm/provider$gpu \
      nohup deploy/provider.sh --model '$MODEL' --listen 0.0.0.0:$port \
      --shm .shm/provider$gpu > logs/provider$gpu.log 2>&1 &"
  done
done

# 2. Router.
if [[ -n "$router" ]]; then
  run "$router" "nohup envoy -c deploy/envoy/envoy.yaml > logs/envoy.log 2>&1 &"
fi

# 3. Clients: shape the link, then one client per GPU, for each K.
for k in "${KS[@]}"; do
  echo "== K=$k"
  index=0
  for entry in "${clients[@]}"; do
    host="${entry%%:*}"; gpus="${entry##*:}"
    run "$host" "sudo deploy/netem.sh up $IFACE"
    for ((gpu = 0; gpu < gpus; gpu++)); do
      run "$host" "CUDA_VISIBLE_DEVICES=$gpu SHM_DIR=.shm/client$gpu \
        nohup deploy/client.sh --model '$MODEL' --connect ${router:-${providers[0]%%:*}}:50151 \
        --method '$METHOD' --requests $REQUESTS --max-tokens $MAX_TOKENS \
        --shm .shm/client$gpu > logs/client-k$k-$gpu.log 2>&1 &"
      index=$((index + 1))
    done
  done
  echo "started $index clients at K=$k; collect logs/client-k$k-*.log when they finish"
done

echo
echo "then: python analysis/stats.py results/at_scale/*.json"
