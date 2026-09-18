# Slipstream

Streaming speculative decoding for split LLM inference over wide-area
networks.

Split inference keeps a prompt private by cutting the model in two: the client
runs the first layer and the language-model head, the provider runs every layer
in between, and only obfuscated hidden states cross the network. Every
generated token then costs a round trip, and speculative decoding is the
natural remedy, except that in a split deployment every drafted token is an
8 KB hidden state on the client's uplink. Slipstream streams each draft tree
instead of shipping it in one piece: a round waits only for the few tokens most
likely to be accepted, the rest travel and run in idle time and catch the
misses, and one marginal rule decides what to wait for and what to send.

This repository is the code behind the paper. It holds a readable
implementation of every mechanism, the wire protocol and the Rust transport
gateway, the drivers for every measurement the paper reports, and the scripts
that draw every figure from the data behind it.

## What is here

```
slipstream/            the system
  tree.py              the round's draft tree, ranks, trunk, ancestor mask
  alpha.py             acceptance probability by path probability
  costs.py             the cost model and the one marginal rule
  scheduler.py         the client's stream scheduler: first segment, admission, priorities
  codec.py             stream compression, FP8 and block-scaled FP4
  client/              encoder, decoder, drafter, the round loop
  provider/            the middle, two queues, passengers, the window
  transport/           the protocol, shared memory, the software-shaped link
  models/              split shards, the drafter's module, the provider's cache
  methods/             the five comparison methods
proto/slipstream.proto the wire protocol
gateway/               the Rust gateway: shared memory to the engine, gRPC to the network
experiments/           one driver per measurement in the paper
figures/               one script per figure, with the data behind it
analysis/              metrics, collection, and a check against the paper's numbers
deploy/                provider, client, link shaping, mutual TLS, the router, the cluster
vllm_integration/      where the engine-side changes go
tests/                 the mechanisms, the attention merge, the ring layout
```

Every module names the section of the paper it implements, and the section
numbers in this file are the paper's.

## Install

```bash
python3 -m pip install -e .            # the package and its runtime needs
python3 -m pip install -e '.[models,transport,figures,dev]'   # everything
cd gateway && cargo build --release    # the transport gateway
```

Python 3.10 or newer, torch 2.4 or newer. The tests and a first run need
nothing else; a checkpoint and a GPU are what a full-scale deployment needs,
and the next sections say how to point the same code at them.

## Three commands to see it work

```bash
python3 -m pytest tests -q                       # 25 tests, a few seconds
python3 analysis/check_paper_numbers.py           # the paper's numbers, from the figure data
python3 experiments/evaluation/overall.py --in-flight 4 --clients 2 --requests 2
```

The first runs the test suite, which exercises every mechanism on a small
built-in model, so it needs no checkpoint. The second recomputes the numbers
Sections 3.2 and 3.3 report from the data the figures are drawn from. The
third runs all five methods end to end, four clients' worth of load on one
provider, over links shaped in software.

## Running the measurements

Each driver writes one json file and prints a table. `--help` lists every knob.

**Section 3, the measurement study.**

```bash
python3 experiments/motivation/chain_sweep.py       --budgets 1 2 3 4 5 6 7 8
python3 experiments/motivation/tree_shape_sweep.py  --budget 8
python3 experiments/motivation/budget_sweep.py      --budgets 3 4 6 8 10 14 18 22
python3 experiments/motivation/stage_breakdown.py   --budgets 3 6 9 12 --hidden-size 4096
python3 experiments/motivation/rank_overlap.py      --dumps figures/data/tree_dumps/nemotron
```

The first three measure acceptance, which needs no network: they draft, verify
and record the acceptance length. The fourth times the five stages of a round
and charges the link for a hidden state of the size you give it. The last
counts, per rank, the share of drafted and of accepted tokens, either from a
fresh run (`--live`) or from the verified-round dumps that ship in
`figures/data/tree_dumps/`.

**Section 6, the evaluation.**

```bash
python3 experiments/evaluation/overall.py            --in-flight 4 8 16 32
python3 experiments/evaluation/ablation.py           --in-flight 4 32
python3 experiments/evaluation/stream_compression.py --in-flight 32
python3 experiments/evaluation/sensitivity.py        --windows 0 4 8 16 24 32
python3 experiments/evaluation/at_scale.py           --providers 8 --clients 24 --k 1 2 4 8 16
python3 experiments/evaluation/overhead.py           --in-flight 4 32
```

The seven ablation variants are settings, not separate code paths, which is
also how the paper describes them: `scheduler.background`,
`scheduler.continuation`, `scheduler.order`, `provider.policy`,
`scheduler.admission` and `scheduler.priorities` in `slipstream/config.py`.

## Reading the results

```bash
python3 analysis/stats.py results/evaluation/overall.json
python3 analysis/collect.py overall results/evaluation/overall.json
```

`stats.py` prints goodput, the median time per output token, and the counters
that say what the mechanisms did: the acceptance length, the first segment's
length, continuations, passengers the provider carried, and uplink bytes per
output token. `collect.py` reshapes a run into what a figure script reads.

## Drawing the figures

Every figure in the paper is one script under `figures/`, and the data behind
it ships next to it:

```bash
python3 figures/acceptance_vs_draft.py --save     # Figure 3
python3 figures/tree_budget_ext.py --save         # Figure 4
python3 figures/stage_stack_vs_budget.py --save   # Figure 5
python3 figures/tree_overlap_paper.py --save      # Figure 6
python3 figures/overall_perf.py --save            # Figure 10
python3 figures/ablation.py --save                # Figure 11
python3 figures/overall_scale.py --save           # Figure 12
```

Without `--save` a script opens a window instead of writing a file. The
architecture figure is drawn by hand and `figures/design_arch.py --regenerate`
overwrites it.

## Deploying it for real

One provider, one or more clients, one persistent gRPC connection each, and a
Rust gateway on each machine that owns the socket while the engine owns the
GPU.

```bash
# the provider
deploy/provider.sh --model /path/to/checkpoint --listen 0.0.0.0:50151 --device cuda:0

# a residential link, on the client's machine
sudo deploy/netem.sh up eth0                      # 20 Mbps up, 100 Mbps down, 20 ms each way

# a client
deploy/client.sh --model /path/to/checkpoint --connect provider.example:50151 \
                 --method slipstream --requests 50
```

Mutual TLS is available: `deploy/tls/gen_certs.sh certs` and then `--tls-dir
certs` on both sides. For many clients and many providers, put the router in
front of them and give every client a shaped link:

```bash
envoy -c deploy/envoy/envoy.yaml                  # least-request over eight providers
deploy/cluster/launch_at_scale.sh --hosts deploy/cluster/hosts.example --k 1 2 4 8 16
```

`docs/deployment.md` has the details, including what each process needs and
where its logs go.

## Running it on your own hardware

The defaults are sized so that the tests and a first run need nothing but
Python. Four settings move a run onto a checkpoint, a GPU and real traffic.

**A model.** `--model /path/to/checkpoint` loads a checkpoint and splits it the
way Section 2.1 describes: the embedding, the first layer, the final norm and
the language-model head to the client, every remaining layer to the provider.
`--device cuda:0` puts each side on its GPU. The vLLM backend
(`vllm_integration/README.md`) is what carries the mixture-of-experts models of
Section 3.1, and the design above it is unchanged.

**A drafter.** `--drafter mtp --mtp /path/to/module` uses the vendor's
multi-token-prediction module, the drafter of Section 2.2: one trained
next-token layer applied recursively, running on the client, which needs only
the hidden states the provider returns and the head the client already holds.

**Request sets.** The four sets of Section 3.1 are public, and are downloaded
rather than redistributed here:

```
ShareGPT       https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
LiveCodeBench  https://huggingface.co/datasets/livecodebench/code_generation_lite
MATH           https://huggingface.co/datasets/hendrycks/competition_math
LongBench      https://huggingface.co/datasets/THUDM/LongBench
```

`--dataset prompts.jsonl --workload math` points a client or a driver at one of
them, one JSON object per line with a `prompt` or `messages` field. Without one,
the drivers generate prompts whose lengths follow the distributions in
`figures/data/dataset_lengths.json`, so a run has the shape of the workload it
names.

**The link.** On a deployment, `deploy/netem.sh` shapes the interface and the
traffic is real. On one machine, `--transport loopback` shapes the link in
software and `--wire-hidden-bytes N` sets what it charges per hidden state,
which is 8192 for a model of hidden size 4096 at 16-bit precision, the size
Section 3.1 gives. `--transport direct` removes the link entirely and makes a
run deterministic, which is what the correctness tests use.

Every setting above is also a field in `slipstream/config.py`, and a run's json
records the ones it used.

## License

MIT. See `LICENSE`.
