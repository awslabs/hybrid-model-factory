"""Pre-process The Cauldron (HuggingFaceM4/the_cauldron) subsets into proper
sharegpt parquet for hmf training.

Cauldron rows look like:
    {
        "texts": [{"user": "Q?", "assistant": "A.", "source": "TallyQA"}, ...],
        "images": [<PIL.Image>, ...],
    }

hmf's sharegpt converter expects:
    {
        "conversations": [{"from": "human", "value": "Q?"},
                          {"from": "gpt",   "value": "A."}, ...],
        "images": [<PIL.Image>, ...],
    }

Cauldron also stores images at the row level with no inline placeholder, but
mm_plugin requires one ``<image>`` token per image in the conversation. We
prepend ``<image>`` to the first user message of each row.

After running this, point each ``cauldron_*_local`` entry in
``training/data/dataset_info.json`` at the per-subset ``data.parquet`` it
writes (e.g. via ``file_name`` field).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Optional

# Subsets that have proven safe and useful for VL SFT.
DEFAULT_SUBSETS = (
    "chartqa", "docvqa", "ai2d", "ocrvqa", "tallyqa",
    "textvqa", "aokvqa", "scienceqa",
)

# Per-subset hard caps. Tuned to match natural row counts when the corpus is
# already small (no random truncation), and to ~30k for the larger ones.
DEFAULT_CAPS: Mapping[str, int] = {
    "ai2d":      2434,    # full
    "scienceqa": 4976,    # full
    "docvqa":    10189,   # full
    "chartqa":   18265,   # full
    "aokvqa":    16500,   # full
    "textvqa":   22000,   # full
    "tallyqa":   30000,   # capped from 98k
    "ocrvqa":    30000,   # capped from 165k
    # Skipped intentionally:
    #   "plotqa"     — 30+ turns/row, 15k chars median, exceeds cutoff_len=1024
    #   "clevr_math" — references local /fsx/m4 image paths from M4 team's filesystem
}


def _expand_row(row):
    """Convert one Cauldron row to sharegpt format with ``<image>`` prepended."""
    convo = []
    for i, turn in enumerate(row["texts"]):
        user_text = turn.get("user", "")
        if i == 0:
            # Cauldron rows have exactly 1 image; prepend <image> to the first
            # user message so mm_plugin's image-token count matches.
            user_text = "<image>" + user_text
        convo.append({"from": "human", "value": user_text})
        convo.append({"from": "gpt", "value": turn.get("assistant", "")})
    return {"conversations": convo, "images": row.get("images", [])}


def _assistant_response_health_check(ds, subset_name: str) -> None:
    """Print top assistant responses to spot boilerplate / refusal patterns.

    A 2% data artifact — e.g. textvqa's "Answering does not require reading
    text in the image." — can hijack downstream MCQ benchmarks at eval time.
    Always sanity-check the top responses before training.
    """
    responses = []
    for convo in ds["conversations"]:
        for turn in convo:
            if turn["from"] == "gpt":
                responses.append(turn["value"])
    if not responses:
        return

    ctr = Counter(responses)
    total = len(responses)
    print(f"  [health] {subset_name}: {len(ctr)} unique assistant responses, top 3:")
    for resp, n in ctr.most_common(3):
        pct = n / total * 100
        flag = "⚠️" if pct > 1.0 and len(resp) > 30 else ""
        print(f"    {flag} {n:>5d}x ({pct:5.1f}%)  {resp[:100]!r}")


def prepare_subset(subset: str, out_root: Path, cap: Optional[int] = None) -> Optional[Path]:
    """Download a Cauldron subset, expand it, write one parquet.

    Idempotent: skips if ``_DONE`` flag exists in the output dir. Returns the
    output path on success, ``None`` if skipped.
    """
    from datasets import load_dataset  # local import — only needed when prepping

    out_dir = out_root / subset
    flag = out_dir / "_DONE"
    if flag.exists():
        print(f"[skip] {subset} already processed at {out_dir}")
        return None

    print(f"[load] HuggingFaceM4/the_cauldron / {subset}")
    ds = load_dataset("HuggingFaceM4/the_cauldron", subset, split="train", num_proc=4)

    if cap and len(ds) > cap:
        ds = ds.shuffle(seed=42).select(range(cap))
        print(f"  capped to {cap} rows")
    else:
        print(f"  {len(ds)} rows (no cap)")

    print(f"  expanding to sharegpt...")
    ds = ds.map(_expand_row, remove_columns=ds.column_names, num_proc=4)

    _assistant_response_health_check(ds, subset)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data.parquet"
    print(f"  writing {out_path}")
    # Use Dataset.to_parquet (NOT pandas.to_parquet) — preserves nested image
    # schema correctly so HF datasets can roundtrip without
    # ArrowNotImplementedError on chunked-array output.
    ds.to_parquet(str(out_path))
    flag.touch()
    print(f"  done -> {out_path}")
    return out_path


def prepare_all(
    out_dir: str,
    subsets: Iterable[str] = DEFAULT_SUBSETS,
    default_cap: int = 30000,
    overrides: Optional[Mapping[str, int]] = None,
) -> None:
    """Run ``prepare_subset`` for every requested subset.

    Args:
        out_dir: Root directory; one subdir per subset is created.
        subsets: Iterable of Cauldron subset names. Default is an 8-subset mix
            of concise short-answer VL data.
        default_cap: Per-subset cap when not in DEFAULT_CAPS or overrides.
        overrides: Optional per-subset cap overrides.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    caps = dict(DEFAULT_CAPS)
    if overrides:
        caps.update(overrides)

    for sub in subsets:
        prepare_subset(sub, out_root, caps.get(sub, default_cap))

    print("All requested subsets processed.")


# ---------------------------------------------------------------------------
# Boilerplate filter for Cauldron textvqa
# ---------------------------------------------------------------------------

# Cauldron's textvqa contains ~2.1% rows where the assistant response is the
# literal string "Answering does not require reading text in the image."
# This is a refusal placeholder for non-OCR questions in an OCR-targeted
# dataset. SFT on these rows can teach the model to apply the refusal as a
# generic VL response, hurting MCQ benchmarks like AI2D.
TEXTVQA_BOILERPLATE = "does not require reading"


def filter_textvqa_boilerplate(src_parquet: str, out_parquet: str) -> None:
    """Write a copy of a Cauldron textvqa parquet with boilerplate rows removed.

    ``src_parquet`` should be the output of ``prepare_subset("textvqa", ...)``.
    Loads via HF Datasets (NOT pandas) so the nested image schema roundtrips
    correctly when written back out.
    """
    from datasets import Dataset
    from pathlib import Path

    src = Path(src_parquet)
    if not src.exists():
        raise FileNotFoundError(
            f"Source parquet missing at {src}. Run prepare_subset('textvqa', ...) first."
        )

    print(f"[load] {src}")
    ds = Dataset.from_parquet(str(src))
    print(f"  src rows: {len(ds)}")

    def _is_clean(example):
        for turn in example["conversations"]:
            if turn["from"] == "gpt" and TEXTVQA_BOILERPLATE in turn["value"]:
                return False
        return True

    print(f"[filter] dropping rows whose assistant turn contains "
          f"{TEXTVQA_BOILERPLATE!r}")
    ds_clean = ds.filter(_is_clean, num_proc=8)
    dropped = len(ds) - len(ds_clean)
    print(f"  clean rows: {len(ds_clean)} (dropped {dropped})")

    out = Path(out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[write] {out}")
    ds_clean.to_parquet(str(out))
    print("Done.")


__all__ = [
    "DEFAULT_SUBSETS",
    "DEFAULT_CAPS",
    "TEXTVQA_BOILERPLATE",
    "prepare_subset",
    "prepare_all",
    "filter_textvqa_boilerplate",
]
