#!/usr/bin/env bash
# A self-signed certificate authority and one identity per side, so the
# gateways can run with mutual TLS (Section 5, "Transport").
#
#   deploy/tls/gen_certs.sh certs provider.example
#
# Then pass --tls-dir certs to deploy/provider.sh and deploy/client.sh.
set -euo pipefail

OUT="${1:-certs}"
PROVIDER_CN="${2:-provider}"
mkdir -p "$OUT"
cd "$OUT"

openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout ca.key -out ca.pem -subj "/CN=slipstream-ca" 2>/dev/null

for role in provider client; do
  cn="$PROVIDER_CN"
  [[ "$role" == client ]] && cn="client"
  openssl req -newkey rsa:2048 -nodes -keyout "$role.key" -out "$role.csr" \
    -subj "/CN=$cn" 2>/dev/null
  openssl x509 -req -in "$role.csr" -CA ca.pem -CAkey ca.key -CAcreateserial \
    -days 365 -out "$role.pem" \
    -extfile <(printf "subjectAltName=DNS:%s,DNS:localhost,IP:127.0.0.1" "$cn") 2>/dev/null
  rm -f "$role.csr"
done

echo "wrote $OUT/{ca.pem,provider.pem,provider.key,client.pem,client.key}"
