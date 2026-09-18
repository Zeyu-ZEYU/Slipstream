# Deploying it

Two machines at a minimum: one runs the middle, one runs a client. Each runs
two processes, an engine that owns the GPU and a gateway that owns the socket,
and they meet in two shared-memory rings.

## The provider

```bash
deploy/provider.sh --model /path/to/checkpoint --listen 0.0.0.0:50151 --device cuda:0 \
                   --window 24 --passengers 4 --policy opportunistic
```

What the flags mean (Section 4.3): `--window` is how many background tokens
the provider will hold per client, `--passengers` is `B`, the background
tokens a foreground batch may carry, and `--policy` is the execution policy,
`opportunistic` by default with `eager` and `lazy` as the two ablations.

With `--state-budget-gb` the provider divides that budget among the clients
with a round in progress and announces each share as the window, which is what
the paper describes; without it the window is fixed.

## A client

```bash
sudo deploy/netem.sh up eth0                # shape the link first
deploy/client.sh --model /path/to/checkpoint --connect provider.example:50151 \
                 --method slipstream --requests 50 --max-tokens 256
```

`--method` selects one of the five methods of Section 6.1. `--compression`
turns stream compression on. `--dataset` points at a jsonl of prompts and
`--workload` names which request set it is; without a path, the client
generates prompts whose lengths follow that set's distribution
(`experiments/datasets.py`).

## The link

`deploy/netem.sh` shapes an interface the way Section 3.1 measures and
Section 6.6 emulates: 20 Mbps up, 100 Mbps down, 20 ms each way. Egress is
shaped directly and ingress through an `ifb` device, which is the usual way to
rate-limit inbound traffic with `tc`. `deploy/netem.sh down eth0` removes it.

## Mutual TLS

```bash
deploy/tls/gen_certs.sh certs provider.example
deploy/provider.sh --tls-dir certs ...
deploy/client.sh   --tls-dir certs ...
```

The gateway reads the certificates; Python never sees them.

## Many clients and many providers

Section 6.6 runs eight providers behind a router and 24 clients on emulated
links. The router is Envoy with `LEAST_REQUEST` over the provider gateways,
which is "every new request to the provider with the fewest requests in
flight", and a gRPC stream that lives for the connection, which is "and never
moves it".

```bash
envoy -c deploy/envoy/envoy.yaml
deploy/cluster/launch_at_scale.sh --hosts deploy/cluster/hosts.example \
    --model /path/to/checkpoint --k 1 2 4 8 16
```

The launcher starts one gateway and one engine per provider GPU, the router,
then per client GPU one shaped link and one client, for each `K`. Pass
`--dry-run` to see the commands it would run. Logs land in `logs/` on each
machine; `python3 analysis/stats.py` reads the json the clients write.

## One machine

Everything also runs in one process, which is what the tests and the
experiment drivers use by default. `--transport loopback` shapes the links in
software (`slipstream/transport/link.py`), and `--transport direct` removes
the link entirely and makes a run deterministic.
