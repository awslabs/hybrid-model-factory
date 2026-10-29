"""
Times a decision model's forward and backward passes on real tokenized batches, on one GPU, to find
where training time goes.

    python scripts/decision_benchmark.py \
        --model_name_or_path path/to/decision-model \
        --tokenized_path path/to/tokenized-decision-data

It reports:
    1. the time of each batch, with each batch padded to its longest row (as in training) and
       padded to a multiple of pad_to_multiple_of tokens. A new sequence length costs much more
       than a repeated one when kernels are compiled or tuned per shape;
    2. the same batch run twice, where a large difference again points at per-shape compilation;
    3. a torch.profiler table of the top CUDA kernels and CPU operations over a few training steps.
"""

import random
import statistics
import time

import fire
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer

import hmf.model.hybrid_zoo.models.model_register  # noqa: F401  # Registers hybrid and decision models
from hmf.data import DecisionDataCollator
from hmf.train.decision.trainer import MODEL_INPUTS


LORA_TARGETS = (
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj"
)


def _environment() -> None:
    import transformers

    print(f"GPU: {torch.cuda.get_device_name()} | torch {torch.__version__} | transformers {transformers.__version__}")
    try:
        import fla

        print(f"flash-linear-attention {fla.__version__}")
    except ImportError:
        print("flash-linear-attention: not installed")
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5

        print(f"Qwen3.5 Gated DeltaNet fast path: {modeling_qwen3_5.is_fast_path_available}")
    except ImportError:
        pass


def _step(model, batch, train: bool) -> None:
    inputs = {name: batch[name] for name in MODEL_INPUTS}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if train:
            out = model(**inputs)
            F.nll_loss(out.log_probs, batch["label"]).backward()
        else:
            with torch.no_grad():
                model(**inputs)


def _time(model, batch, train: bool) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    _step(model, batch, train)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def _report(name: str, times: list, batches: list) -> None:
    lengths = [batch["input_ids"].shape[1] for batch in batches]
    print(f"\n{name}")
    print(f"  distinct padded lengths: {len(set(lengths))} of {len(lengths)} batches")
    print(f"  first batch: {times[0]:.3f} s | median of the rest: {statistics.median(times[1:]):.3f} s")
    print("  per batch (padded length: seconds): " + ", ".join(f"{n}: {t:.2f}" for n, t in zip(lengths, times)))


def benchmark(
    model_name_or_path: str,
    tokenized_path: str,
    batch_size: int = 4,
    num_batches: int = 20,
    pad_to_multiple_of: int = 256,
    lora: bool = True,
    gradient_checkpointing: bool = False,
    profile_steps: int = 3,
    seed: int = 0,
):
    _environment()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path, dtype=torch.bfloat16).cuda()
    if lora:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=16, lora_alpha=32, target_modules=LORA_TARGETS.split(","), modules_to_save=["decision_head"]
        )
        model = get_peft_model(model, config)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    # Random batches of the first augmentation of the training split
    dataset = load_from_disk(tokenized_path)["train"]
    rows = [i for i, augmentation in enumerate(dataset["augmentation"]) if augmentation == 0]
    rng = random.Random(seed)
    indices = [rng.sample(rows, batch_size) for _ in range(num_batches)]
    features = [[dataset[i] for i in batch] for batch in indices]
    lengths = [len(f["input_ids"]) for batch in features for f in batch]
    print(f"\nrows: {len(lengths)} | prompt length mean {statistics.mean(lengths):.0f}, max {max(lengths)}")

    def collate(multiple):
        collator = DecisionDataCollator(tokenizer=tokenizer, pad_to_multiple_of=multiple)
        return [{k: v.cuda() for k, v in collator(batch).items()} for batch in features]

    natural, padded = collate(None), collate(pad_to_multiple_of)

    # 1. Each batch's time, natural padding then padding to a multiple of pad_to_multiple_of
    for name, batches in [("natural padding", natural), (f"padded to a multiple of {pad_to_multiple_of}", padded)]:
        for train in (False, True):
            times = [_time(model, batch, train) for batch in batches]
            model.zero_grad(set_to_none=True)
            _report(f"{'forward + backward' if train else 'forward'}, {name}", times, batches)

    # 2. The same, never-seen shape twice, where per-shape compilation shows as a slow first run
    batch = {k: v.clone() for k, v in natural[0].items()}
    extra = 37  # a padded length no batch above had
    batch["input_ids"] = F.pad(batch["input_ids"], (0, extra), value=tokenizer.pad_token_id)
    batch["attention_mask"] = F.pad(batch["attention_mask"], (0, extra), value=0)
    first, second = _time(model, batch, True), _time(model, batch, True)
    model.zero_grad(set_to_none=True)
    print(f"\nnew shape, forward + backward: first run {first:.3f} s, second run {second:.3f} s")

    # 3. Where the time goes, over a few training steps on batches already seen
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for batch in natural[:profile_steps]:
            _step(model, batch, train=True)
        torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    print("\nTop CUDA kernels:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    print("\nTop CPU operations:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))


if __name__ == "__main__":
    fire.Fire(benchmark)
