"""Commonsense-reasoning data pipeline: training loader + real held-out eval sets.

Why this module exists
-----------------------
Two failure modes are cheap to introduce here and expensive to discover later:

1. A wrong or slightly-off instruction template silently produces flat,
   meaningless accuracy in Phase 4 -- there's no exception to catch, the model
   just never learns the task shape. ``generate_prompt`` below matches the
   template ``commonsense_evaluate.py`` in AGI-Edgerunners/LLM-Adapters uses
   at eval time; training and eval must agree on it, or the model is asked at
   eval time to complete a shape it was never trained on.
2. Evaluating against a held-out slice of the *training* distribution instead
   of the real BoolQ/PIQA/HellaSwag test sets makes ties and small gains look
   real when they are actually just measuring how well the eval slice matches
   the training slice. ``load_eval_split`` therefore always pulls the real
   ``dataset/<name>/test.json`` files from LLM-Adapters, never a carved-out
   piece of ``commonsense_170k``.

Dataset source
--------------
Training data: ``zwhe99/commonsense_170k`` on the Hugging Face Hub, which
mirrors ``AGI-Edgerunners/LLM-Adapters``'s ``ft-training_set/commonsense_170k.json``
-- used here as a raw-JSON fallback if the Hub load fails for any reason
(offline, dataset gated/moved, etc).

Do NOT use ``tloen/alpaca-lora-commonsense`` -- that dataset path does not exist.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request
from typing import Any, Mapping, Optional, Sequence

RAW_TRAIN_JSON_URL = (
    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main/"
    "ft-training_set/commonsense_170k.json"
)
HF_TRAIN_DATASET_ID = "zwhe99/commonsense_170k"

#: Base URL for the real held-out test sets, one JSON file per task.
RAW_EVAL_JSON_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main/dataset/{name}/test.json"
)

#: Every task LLM-Adapters ships a commonsense test split for.
ALL_EVAL_DATASETS: tuple[str, ...] = (
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Challenge",
    "ARC-Easy",
    "openbookqa",
)

#: The minimum set called out in the Phase 2 plan.
DEFAULT_EVAL_DATASETS: tuple[str, ...] = ("boolq", "piqa", "hellaswag")

DEFAULT_TRAIN_SUBSET_SIZE = 15_000
DEFAULT_TRAIN_SUBSET_SEED = 42
DEFAULT_MAX_LEN = 256

#: One compiled regex per task, matching commonsense_evaluate.py's extractor.
#: A model trained on this template answers with a bare label token like
#: "answer3" or "true" (never "the answer is answer3"), so a first-match scan
#: is sufficient -- it's what the original eval script does.
_ANSWER_PATTERNS: dict[str, "re.Pattern[str]"] = {
    "boolq": re.compile(r"true|false"),
    "piqa": re.compile(r"solution1|solution2"),
    "social_i_qa": re.compile(r"answer1|answer2|answer3|answer4|answer5"),
    "ARC-Challenge": re.compile(r"answer1|answer2|answer3|answer4|answer5"),
    "ARC-Easy": re.compile(r"answer1|answer2|answer3|answer4|answer5"),
    "openbookqa": re.compile(r"answer1|answer2|answer3|answer4|answer5"),
    "hellaswag": re.compile(r"ending1|ending2|ending3|ending4"),
    "winogrande": re.compile(r"option1|option2"),
}

_REQUIRED_TRAIN_FIELDS = ("instruction", "output", "answer")


class DataPipelineError(RuntimeError):
    """Raised for problems specific to this module (vs. a bare network/IO error)."""


# --------------------------------------------------------------------------
# Instruction template (must match eval-time formatting exactly)
# --------------------------------------------------------------------------


def generate_prompt(instruction: str, input: Optional[str] = None) -> str:  # noqa: A002
    """Format one example with the Alpaca-style template LLM-Adapters uses.

    Every commonsense_170k record already folds the question and its answer
    options into ``instruction`` with ``input`` empty, but both branches are
    implemented since a falsy ``input`` (``None`` or ``""``) must route to the
    no-input branch -- a stray empty-string input would otherwise silently
    change which template is used relative to eval time.
    """
    if input:
        return (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            f"completes the request.\n\n### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input}\n\n### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that "
        f"appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )


def format_example(record: Mapping[str, Any], *, eos_token: str = "") -> str:
    """Full training text: prompt + target completion (+ optional EOS marker).

    This is what a ``formatting_func`` in Phase 4's ``SFTTrainer`` should
    return per example -- SFTTrainer does its own tokenization, so training
    examples are handed to it as text, not as pre-tokenized ids.
    """
    prompt = generate_prompt(record["instruction"], record.get("input"))
    return f"{prompt}{record['output']}{eos_token}"


def extract_answer(dataset: str, generated_text: str) -> str:
    """Pull the bare answer label out of a model's generated completion.

    Mirrors ``commonsense_evaluate.py``'s ``extract_answer``: first regex
    match, or ``""`` if the model never produced a recognizable label. Do not
    special-case a missing match into a guess -- an empty prediction should
    count as wrong, not accidentally match ``label == ""``.
    """
    if dataset not in _ANSWER_PATTERNS:
        raise DataPipelineError(f"Unknown eval dataset {dataset!r}; known: {sorted(_ANSWER_PATTERNS)}")
    match = _ANSWER_PATTERNS[dataset].search(generated_text.strip())
    return match.group(0) if match else ""


# --------------------------------------------------------------------------
# Training data: commonsense_170k
# --------------------------------------------------------------------------


def _fetch_json_url(url: str, timeout: float = 60.0) -> Any:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.load(resp)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise DataPipelineError(f"Could not fetch {url}: {exc}") from exc


def _validate_train_records(records: Sequence[Mapping[str, Any]], source: str) -> None:
    if not records:
        raise DataPipelineError(f"{source} returned zero records.")
    sample = records[0]
    missing = [f for f in _REQUIRED_TRAIN_FIELDS if f not in sample]
    if missing:
        raise DataPipelineError(
            f"{source} record is missing expected field(s) {missing}; got keys {sorted(sample)}. "
            "The dataset schema may have changed since this loader was written."
        )


def load_commonsense_170k(cache_dir: Optional[str] = None) -> list[dict[str, Any]]:
    """Load the full 170k commonsense-reasoning training set.

    Tries the Hugging Face Hub mirror first (``zwhe99/commonsense_170k``),
    which is faster and resumable; falls back to fetching the raw JSON
    straight from AGI-Edgerunners/LLM-Adapters on any failure (network,
    missing ``datasets`` package, dataset moved/gated, etc). Both paths are
    validated against the same required-field check so a caller downstream
    can't silently get a differently-shaped record depending on which path
    was taken.
    """
    try:
        from datasets import load_dataset  # local import: optional dependency at call time

        ds = load_dataset(HF_TRAIN_DATASET_ID, split="train", cache_dir=cache_dir)
        records = [dict(r) for r in ds]
        _validate_train_records(records, f"Hub dataset {HF_TRAIN_DATASET_ID!r}")
        return records
    except Exception as hub_exc:  # noqa: BLE001 -- deliberately broad, this is a fallback boundary
        try:
            records = _fetch_json_url(RAW_TRAIN_JSON_URL)
        except DataPipelineError as raw_exc:
            raise DataPipelineError(
                f"Both the Hub load of {HF_TRAIN_DATASET_ID!r} and the raw-JSON fallback failed.\n"
                f"  Hub error : {hub_exc.__class__.__name__}: {hub_exc}\n"
                f"  raw error : {raw_exc}"
            ) from raw_exc
        _validate_train_records(records, f"raw JSON at {RAW_TRAIN_JSON_URL}")
        return records


def build_train_subset(
    records: Sequence[Mapping[str, Any]],
    n: int = DEFAULT_TRAIN_SUBSET_SIZE,
    seed: int = DEFAULT_TRAIN_SUBSET_SEED,
) -> list[dict[str, Any]]:
    """Deterministically subsample ``n`` records, order preserved.

    Order is preserved (rather than returned in shuffled order) so that
    re-running this with the same seed and re-inspecting "the first 3
    examples" during Phase 2.3 shows the same 3 examples every time.
    """
    if n > len(records):
        raise DataPipelineError(f"Requested subset of {n} but only {len(records)} records are available.")
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    chosen = sorted(indices[:n])
    return [dict(records[i]) for i in chosen]


def truncation_stats(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    max_len: int = DEFAULT_MAX_LEN,
) -> dict[str, float]:
    """Report what fraction of formatted examples exceed ``max_len`` tokens.

    Run this once after loading data and before launching Phase 4 training.
    A high truncation rate silently drops the model's target completion (the
    part *after* the prompt) for the worst-affected examples, which looks
    exactly like flat/noisy accuracy later and is easy to misdiagnose as a
    method problem instead of a data problem.
    """
    lengths = [len(tokenizer(format_example(r), add_special_tokens=False)["input_ids"]) for r in records]
    over = sum(1 for L in lengths if L > max_len)
    return {
        "n": len(lengths),
        "max_len": max_len,
        "mean_tokens": sum(lengths) / len(lengths),
        "max_tokens": max(lengths),
        "n_truncated": over,
        "fraction_truncated": over / len(lengths),
    }


def tokenize_for_training(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    max_len: int = DEFAULT_MAX_LEN,
):
    """Tokenize with the prompt portion of each example masked out of the loss.

    Optional: Phase 4's ``SFTTrainer`` can tokenize text-field examples on its
    own via ``formatting_func=format_example``, so this function is not on the
    critical path. It exists for offline inspection, for unit tests that don't
    want to spin up a trainer, and for anyone who ends up hand-rolling the
    training loop instead of using ``SFTTrainer``.

    Returns a ``datasets.Dataset`` with ``input_ids``, ``attention_mask``, and
    ``labels`` (prompt tokens set to -100, so the loss only sees the answer).
    """
    from datasets import Dataset

    eos = tokenizer.eos_token or ""
    input_ids_col: list[list[int]] = []
    attn_col: list[list[int]] = []
    labels_col: list[list[int]] = []

    for record in records:
        prompt = generate_prompt(record["instruction"], record.get("input"))
        full_text = f"{prompt}{record['output']}{eos}"

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:max_len]

        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        input_ids_col.append(full_ids)
        attn_col.append([1] * len(full_ids))
        labels_col.append(labels)

    return Dataset.from_dict({"input_ids": input_ids_col, "attention_mask": attn_col, "labels": labels_col})


def save_jsonl(records: Sequence[Mapping[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------
# Real held-out evaluation sets (never a train-split proxy)
# --------------------------------------------------------------------------


def download_eval_split(name: str, dest_dir: str = "dataset") -> str:
    """Fetch ``dataset/<name>/test.json`` from LLM-Adapters if not already local.

    Mirrors the exact path ``commonsense_evaluate.py`` expects
    (``dataset/<name>/test.json``), so Phase 4/7 code that shells out to that
    script, or a hand-rolled evaluator using the same layout, finds the file
    in the place it looks.
    """
    if name not in ALL_EVAL_DATASETS:
        raise DataPipelineError(f"Unknown eval dataset {name!r}; known: {list(ALL_EVAL_DATASETS)}")
    out_path = os.path.join(dest_dir, name, "test.json")
    if os.path.isfile(out_path):
        return out_path
    url = RAW_EVAL_JSON_URL_TEMPLATE.format(name=name)
    data = _fetch_json_url(url)
    if not data:
        raise DataPipelineError(f"{url} returned an empty test set.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return out_path


def load_eval_split(name: str, dest_dir: str = "dataset") -> list[dict[str, Any]]:
    """Load a real held-out test set, downloading it first if necessary."""
    path = download_eval_split(name, dest_dir)
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    if not records:
        raise DataPipelineError(f"{path} is empty.")
    sample = records[0]
    for field in ("instruction", "answer"):
        if field not in sample:
            raise DataPipelineError(
                f"{path} record is missing expected field {field!r}; got keys {sorted(sample)}."
            )
    return records


# --------------------------------------------------------------------------
# CLI: build the training subset, fetch eval sets, eyeball either (Phase 2)
# --------------------------------------------------------------------------


def _print_examples(records: Sequence[Mapping[str, Any]], n: int = 3) -> None:
    for i, record in enumerate(records[:n]):
        print(f"--- example {i} ---")
        print("instruction:", record.get("instruction"))
        if record.get("input"):
            print("input      :", record.get("input"))
        print("output     :", record.get("output"))
        if "answer" in record:
            print("answer     :", record.get("answer"))
        print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m adagralora.data")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("build-train", help="download + subsample the training set")
    p_train.add_argument("--n", type=int, default=DEFAULT_TRAIN_SUBSET_SIZE)
    p_train.add_argument("--seed", type=int, default=DEFAULT_TRAIN_SUBSET_SEED)
    p_train.add_argument("--out", default="data/train_subset.jsonl")
    p_train.add_argument("--eyeball", type=int, default=3, help="print this many examples; 0 to skip")

    p_eval = sub.add_parser("fetch-eval", help="download real held-out test sets")
    p_eval.add_argument("--datasets", nargs="+", default=list(DEFAULT_EVAL_DATASETS), choices=ALL_EVAL_DATASETS)
    p_eval.add_argument("--dest-dir", default="dataset")
    p_eval.add_argument("--eyeball", type=int, default=3, help="print this many examples per set; 0 to skip")

    args = parser.parse_args(argv)

    if args.command == "build-train":
        print(f"Loading {HF_TRAIN_DATASET_ID} (falls back to raw JSON if unavailable)...")
        records = load_commonsense_170k()
        print(f"Loaded {len(records)} records.")
        subset = build_train_subset(records, n=args.n, seed=args.seed)
        save_jsonl(subset, args.out)
        print(f"Saved {len(subset)} records (seed={args.seed}) to {args.out}")
        if args.eyeball:
            _print_examples(subset, args.eyeball)
        return 0

    if args.command == "fetch-eval":
        for name in args.datasets:
            path = download_eval_split(name, args.dest_dir)
            records = load_eval_split(name, args.dest_dir)
            print(f"{name}: {len(records)} examples -> {path}")
            if args.eyeball:
                _print_examples(records, args.eyeball)
        return 0

    parser.error(f"unknown command {args.command!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
