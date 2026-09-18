"""The four request sets of Section 3.1, and prompts to run without them.

Four request sets span the main serving domains: open-ended chat, competition
code, mathematical reasoning, and long-context question answering. They are
public and are downloaded rather than redistributed here; `load` reads
whichever ones you have, one JSON object per line with a `prompt` or
`messages` field:

    ShareGPT         https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
    LiveCodeBench    https://huggingface.co/datasets/livecodebench/code_generation_lite
    MATH             https://huggingface.co/datasets/hendrycks/competition_math
    LongBench        https://huggingface.co/datasets/THUDM/LongBench

Without a path, `load` generates prompts whose lengths follow the
distributions of the named set, which ship in
`figures/data/dataset_lengths.json` and are what Figure A1 plots, so a run has
the shape of the workload it names.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = ROOT / "figures" / "data" / "dataset_lengths.json"

WORKLOADS = ("sharegpt", "livecodebench", "math", "longbench")

# Output caps of Section 3.1: 256 tokens on chat and long context, 512 on code
# and math.
OUTPUT_CAP = {
    "sharegpt": 256,
    "longbench": 256,
    "livecodebench": 512,
    "math": 512,
    "generated": 256,
}


def _open(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path)


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    records = []
    with _open(Path(path)) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def prompt_text(record: dict) -> str:
    if "prompt" in record:
        return str(record["prompt"])
    messages = record.get("messages") or []
    for message in messages:
        if message.get("role") in (None, "user"):
            return str(message.get("content", ""))
    if messages:
        return str(messages[0].get("content", ""))
    for key in ("question", "problem", "input", "text"):
        if key in record:
            return str(record[key])
    return ""


def tokenize(texts: list[str], tokenizer_path: str) -> list[list[int]]:
    """Tokenize with the model's own tokenizer, which is what the lengths in
    Section 3.1 are counted in."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return [tokenizer(text).input_ids for text in texts]


def input_lengths(workload: str) -> list[int]:
    """Every request's prompt length in the named set, the token counts behind
    Figure 3."""
    if not LENGTHS.exists():
        return []
    data = json.loads(LENGTHS.read_text())
    entry = data.get(workload.lower()) or {}
    return [int(v) for v in entry.get("input", []) if v]


def generated_prompts(
    count: int,
    tokens: int = 24,
    vocab: int = 512,
    seed: int = 0,
    workload: str = "generated",
    spread: bool = False,
) -> list[list[int]]:
    """Prompts of the shape a run asks for.

    With `spread`, prompt lengths follow the distribution of the named request
    set, so long-context behaviour shows up; otherwise every prompt has
    `tokens` tokens, which keeps a short run quick.
    """
    rng = random.Random(seed)
    lengths = [tokens] * count
    if spread:
        real = input_lengths(workload)
        if real:
            # Draw from the set's own prompt lengths, capped so a smoke run
            # stays quick.
            cap = 8 * tokens
            lengths = [max(4, min(rng.choice(real), cap)) for _ in range(count)]
    return [
        [rng.randrange(1, vocab) for _ in range(length)] for length in lengths
    ]


def load(
    workload: str,
    count: int,
    *,
    path: str | None = None,
    tokenizer: str | None = None,
    prompt_tokens: int = 24,
    vocab: int = 512,
    seed: int = 0,
) -> list[list[int]]:
    """Prompts as token identifier lists.

    Give `path` and `tokenizer` to read one of the request sets; without a
    path, prompt lengths follow that set's own distribution.
    """
    if workload == "generated" or not path:
        return generated_prompts(
            count, tokens=prompt_tokens, vocab=vocab, seed=seed, workload=workload
        )
    records = load_jsonl(path, limit=count)
    texts = [prompt_text(record) for record in records]
    if tokenizer:
        return tokenize(texts, tokenizer)
    # Without a tokenizer, stand in with the text's byte values, which keeps
    # lengths realistic even though the identifiers are not the model's.
    return [[b % vocab for b in text.encode()[: 8 * prompt_tokens]] for text in texts]


def describe(workload: str, prompts: list[list[int]]) -> dict:
    lengths = sorted(len(p) for p in prompts)
    if not lengths:
        return {"workload": workload, "requests": 0}
    return {
        "workload": workload,
        "requests": len(lengths),
        "prompt_tokens_p50": lengths[len(lengths) // 2],
        "prompt_tokens_min": lengths[0],
        "prompt_tokens_max": lengths[-1],
        "output_cap": OUTPUT_CAP.get(workload, 256),
    }
