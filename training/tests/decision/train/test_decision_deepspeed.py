"""
Tests for training decision models with DeepSpeed, and for gradient checkpointing of long batches
only (decision_checkpoint_min_tokens).

On the CPU:
    - the pointer head's layers compute in float32 in their own forward, which is what DeepSpeed
      ZeRO-3 needs to gather their weights;
    - the optimizer's parameter groups and counts under ZeRO-3, whose ranks hold empty placeholders;
    - checkpointing turned on and off per batch.

On two or more GPUs, `hmf train` on two GPUs with DeepSpeed ZeRO-2 and ZeRO-3 (HMF's configs in
examples/deepspeed/), for three backbones, which are stock Qwen3, stock Qwen3.5 (Gated DeltaNet layers), and
an HMF hybrid with a GKA layer. Each run trains with a label set (both KL terms), evaluation, and
checkpointing of long batches only, and is checked against the same run without DeepSpeed. The
first step's losses agree, the head's parameters are counted and grouped correctly, and the saved
adapter and head reload without DeepSpeed.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import yaml
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast, TrainingArguments

import hmf.model.hybrid_zoo.models.model_register  # noqa: F401  # Registers hybrid and decision models
from hmf.model.hybrid_zoo.models.decision.modeling_decision import PointerHead
from hmf.train.decision.trainer import DecisionTrainer, checkpointing_layers


EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
HEAD_DIM = 16


# The head under ZeRO-3 (CPU)


def test_head_layers_run_their_own_forward_in_float32():
    head = PointerHead(32, dim=HEAD_DIM).to(torch.bfloat16).eval()  # weights in bf16, as under DeepSpeed
    called = []
    for name in ("norm", "q", "k"):
        getattr(head, name).register_forward_pre_hook(lambda module, args, name=name: called.append(name))

    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = head(
            torch.randn(2, 32, dtype=torch.bfloat16),
            torch.randn(2, 3, 32, dtype=torch.bfloat16),
            torch.ones(2, 3, dtype=torch.bool),
        )

    # ZeRO-3 gathers a layer's weights when its forward runs, so every use goes through one
    assert sorted(set(called)) == ["k", "norm", "q"]
    assert logits.dtype == torch.float32


class _TinyDecisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decision_head = PointerHead(8, dim=4)
        self.adapter = nn.Linear(8, 8)


def test_optimizer_groups_under_zero3(tmp_path):
    model = _TinyDecisionModel()
    shapes = {}
    for name, param in model.named_parameters():
        # A ZeRO-3 rank holds an empty placeholder, with the real shape and size as attributes
        shapes[name] = tuple(param.shape)
        param.ds_shape, param.ds_numel = param.shape, param.numel()
        param.data = torch.empty(0)
    trainer = SimpleNamespace(
        optimizer=None,
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path), learning_rate=1e-4, weight_decay=0.01, report_to="none"),
        finetuning_args=SimpleNamespace(decision_head_lr=1e-3),
    )
    logs = []
    from hmf.train.decision import trainer as trainer_module

    original = trainer_module.logger.info_rank0
    trainer_module.logger.info_rank0 = logs.append
    try:
        optimizer = DecisionTrainer.create_optimizer(trainer)
    finally:
        trainer_module.logger.info_rank0 = original

    head = model.decision_head
    decayed = next(g for g in optimizer.param_groups if g["weight_decay"] > 0)
    assert {id(p) for p in decayed["params"]} == {id(head.q.weight), id(head.k.weight)}
    # Counts use the real sizes: 8 x 4 matrices and 4 biases for q and k, and a LayerNorm of 8
    assert logs == [
        "Optimizer: 88 head parameters at lr 0.001 (64 with weight decay 0.01), 72 other parameters at lr 0.0001."
    ]


class _TinyPeftTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.decision_head = PointerHead(8, dim=4)
        self.adapter = nn.Linear(8, 8)

    def forward(self, x):
        return self.adapter(x)


@pytest.mark.parametrize("zero3", [True, False])
def test_zero3_marks_the_trained_head_as_one_unit(monkeypatch, zero3):
    pytest.importorskip("deepspeed")
    import transformers.integrations.deepspeed as hf_deepspeed
    from deepspeed.utils import z3_leaf_module
    from peft import LoraConfig, get_peft_model
    from peft.utils import ModulesToSaveWrapper

    from hmf.train.decision.workflow import mark_zero3_leaf_modules

    model = get_peft_model(
        _TinyPeftTarget(), LoraConfig(target_modules=["adapter"], modules_to_save=["decision_head"])
    )
    monkeypatch.setattr(hf_deepspeed, "is_deepspeed_zero3_enabled", lambda: zero3)

    mark_zero3_leaf_modules(model)

    wrappers = [module for module in model.modules() if isinstance(module, ModulesToSaveWrapper)]
    assert len(wrappers) == 1
    assert z3_leaf_module(wrappers[0]) is zero3  # only under ZeRO-3
    # Nothing else is a leaf, including the LoRA layer and the two copies of the head inside the wrapper
    others = [module for module in model.modules() if not isinstance(module, ModulesToSaveWrapper)]
    assert not any(z3_leaf_module(module) for module in others)


# Checkpointing of long batches (CPU)


class _Layer(nn.Module):
    def __init__(self, checkpointing: bool):
        super().__init__()
        self.gradient_checkpointing = checkpointing


def test_checkpointing_layers():
    model = nn.Sequential(_Layer(True), _Layer(False), nn.Sequential(_Layer(True)))
    assert checkpointing_layers(model) == [model[0], model[2][0]]


@pytest.mark.parametrize(
    "min_tokens, shape, training, expected",
    [
        (100, (2, 64), True, True),     # 128 padded tokens, checkpointed
        (100, (2, 32), True, False),    # 64, not checkpointed
        (100, (2, 50), True, False),    # exactly 100, not checkpointed (it must be more than min_tokens)
        (0, (2, 32), True, True),       # 0, every batch, as without the setting
        (100, (2, 32), False, True),    # evaluation, unchanged
    ],
)
def test_checkpointing_of_long_batches_only(min_tokens, shape, training, expected):
    model = nn.Sequential(_Layer(True), _Layer(True)).train(training)
    trainer = SimpleNamespace(
        finetuning_args=SimpleNamespace(decision_checkpoint_min_tokens=min_tokens), _checkpointing_layers=None
    )

    DecisionTrainer._set_checkpointing(trainer, model, {"input_ids": torch.zeros(shape, dtype=torch.long)})

    assert [layer.gradient_checkpointing for layer in model] == [expected, expected]


def test_checkpointing_turns_back_on():
    model = nn.Sequential(_Layer(True)).train()
    trainer = SimpleNamespace(
        finetuning_args=SimpleNamespace(decision_checkpoint_min_tokens=100), _checkpointing_layers=None
    )
    for shape, expected in [((2, 32), False), ((2, 64), True), ((1, 10), False)]:
        DecisionTrainer._set_checkpointing(trainer, model, {"input_ids": torch.zeros(shape, dtype=torch.long)})
        assert model[0].gradient_checkpointing is expected


def test_every_rank_makes_the_same_choice(monkeypatch):
    # Another rank has a batch of 1,000 padded tokens, so this rank's short batch is checkpointed too
    import torch.distributed as dist

    def all_reduce(tensor, op=None):
        tensor.fill_(max(int(tensor), 1000))

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    model = nn.Sequential(_Layer(True)).train()
    trainer = SimpleNamespace(
        finetuning_args=SimpleNamespace(decision_checkpoint_min_tokens=100),
        _checkpointing_layers=None,
        args=SimpleNamespace(world_size=2),
    )

    DecisionTrainer._set_checkpointing(trainer, model, {"input_ids": torch.zeros((2, 32), dtype=torch.long)})

    assert model[0].gradient_checkpointing is True


# DeepSpeed on two GPUs


needs_two_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2 or shutil.which("hmf") is None, reason="needs two GPUs and the hmf command"
)
WORDS = ["state", "words", "question", "other", "refund", "return", "it", "ignore", "escalate", "to", "a",
         "human", "low", "mid", "high", "top", "true", "false", "holds", "answer", "options"]


def _save_tokenizer(path: Path) -> None:
    """A word-level tokenizer for the toy rows, in which the digits (the option numbers) are single tokens."""
    vocab = {"[UNK]": 0, "[PAD]": 1}
    for word in [str(n) for n in range(10)] + WORDS:
        vocab[word] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(path)


def _causal_lm(backbone: str):
    """A tiny causal LM: stock Qwen3, stock Qwen3.5 (Gated DeltaNet and attention), or an HMF GKA hybrid."""
    import transformers

    if backbone == "qwen3":
        return transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
            vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, tie_word_embeddings=True,
        ))
    if backbone == "qwen3_5":
        if getattr(transformers, "Qwen3_5TextConfig", None) is None:
            pytest.skip(f"transformers {transformers.__version__} has no Qwen3.5")
        return transformers.Qwen3_5ForCausalLM(transformers.Qwen3_5TextConfig(
            vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=1, layer_types=["linear_attention", "full_attention"], tie_word_embeddings=True,
        ))
    pytest.importorskip("fla")
    from hmf.model.hybrid_zoo.models.hybrid_qwen3.configuration_hybrid_qwen3 import HybridQwen3Config
    from hmf.model.hybrid_zoo.models.hybrid_qwen3.modeling_hybrid_qwen3 import HybridQwen3ForCausalLM

    return HybridQwen3ForCausalLM(HybridQwen3Config(
        vocab_size=64, hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=64, hybrid_override_pattern="GKA-*", tie_word_embeddings=True,
    ))


def _decision_model(backbone: str, root: Path) -> tuple:
    """A tiny causal LM of the backbone, and the decision model converted from it with hmf decision-init."""
    from hmf.priming.decision_init import DecisionInitConfig, init_decision_model

    source_dir, decision_dir = root / "source", root / "decision"
    torch.manual_seed(0)
    _causal_lm(backbone).to(torch.bfloat16).save_pretrained(source_dir)
    _save_tokenizer(source_dir)
    init_decision_model(
        DecisionInitConfig(
            base_model_name_or_path=str(source_dir), output_dir=str(decision_dir), dtype=torch.bfloat16,
            decision_head_dim=HEAD_DIM, decision_head_dropout=0.0, seed=0,
        )
    )
    return str(source_dir), str(decision_dir)


@pytest.fixture
def labeled(request, tmp_path, decision_dataset_dir) -> dict:
    """A tiny decision model of the requested backbone, and its toy data tokenized and labeled."""
    return _prepare(tmp_path, request.param, decision_dataset_dir)


def _prepare(tmp_path: Path, backbone: str, decision_dataset_dir: str) -> dict:
    """A tiny decision model of a backbone, and the data of decision_dataset_dir tokenized and labeled."""
    from hmf.data import get_dataset, get_template_and_fix_tokenizer
    from hmf.data.decision_label import run_decision_label
    from hmf.hparams.parser import _parse_train_args
    from hmf.model import load_tokenizer

    source_dir, model_dir = _decision_model(backbone, tmp_path)  # the causal LM is the anchor
    with open(os.path.join(model_dir, "config.json")) as f:
        hidden_size = json.load(f)["text_config"]["hidden_size"]

    data_config = {
        "model_name_or_path": model_dir, "stage": "decision", "do_train": True, "finetuning_type": "lora",
        "dataset": "toy_decision", "dataset_dir": decision_dataset_dir, "cutoff_len": 512, "val_size": 0.2,
        "tokenized_path": str(tmp_path / "tokenized"), "overwrite_cache": True, "seed": 0,
        "output_dir": str(tmp_path / "unused"), "report_to": "none",
    }
    model_args, data_args, training_args, _, _ = _parse_train_args(dict(data_config))
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    get_dataset(template, model_args, data_args, training_args, stage="decision", **tokenizer_module)

    # A teacher for every row, and the anchor from the causal LM
    rows = [json.loads(line) for line in Path(decision_dataset_dir, "toy.jsonl").read_text().splitlines()]
    with open(tmp_path / "teacher.jsonl", "w") as f:
        for i, row in enumerate(rows):
            n = 2 if row["kind"] == "noul" else len(row["options"])
            f.write(json.dumps({"source": "toy.jsonl", "row_index": i, "probs": [1.0 / n] * n}) + "\n")
    label_config = tmp_path / "label.yaml"
    label_config.write_text(yaml.safe_dump({
        **data_config, "decision_label_folder": "labels", "decision_anchor_model": source_dir,
        "decision_teacher_file": str(tmp_path / "teacher.jsonl"),
    }))
    run_decision_label(str(label_config), num_gpus=1)
    return {"root": tmp_path, "backbone": backbone, "model_dir": model_dir, "hidden_size": hidden_size,
            "data_config": data_config}


def _train(labeled: dict, name: str, deepspeed: str = None, **overrides) -> dict:
    """hmf train on two GPUs (with a DeepSpeed config, or plain multi-GPU), and what it logged and saved."""
    output_dir = labeled["root"] / name
    config = {
        **labeled["data_config"],
        "do_eval": True,
        "lora_target": "all",
        "lora_rank": 4,
        "decision_label_folder": "labels",
        "decision_anchor_kl_weight": 0.3,
        "decision_teacher_kl_weight": 1.0,
        # Gradient checkpointing is on, for batches of more than 120 padded tokens only. With the toy
        # prompts padded to multiples of 16, some batches are checkpointed and some are not
        "decision_checkpoint_min_tokens": 120,
        "decision_pad_to_multiple_of": 16,
        "max_steps": 3,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "learning_rate": 1e-3,
        "decision_head_lr": 1e-2,
        "weight_decay": 0.01,
        "eval_strategy": "steps",
        "eval_steps": 3,
        "save_strategy": "no",
        "logging_steps": 1,
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "overwrite_cache": False,
        "bf16": True,
        "ddp_timeout": 600,
    }
    config.update(overrides)
    if deepspeed is not None:
        config["deepspeed"] = str(EXAMPLES / "deepspeed" / f"ds_{deepspeed}_config.json")
    config_path = labeled["root"] / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(config))

    env = dict(os.environ, FORCE_TORCHRUN="1", NPROC_PER_NODE="2", CUDA_VISIBLE_DEVICES="0,1")
    try:
        # A hang between the GPUs (e.g. ranks running different collectives) fails instead of blocking
        result = subprocess.run(
            ["hmf", "train", str(config_path)], env=env, capture_output=True, text=True, timeout=900
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"{name}: no result after {error.timeout} s; the GPUs may be waiting on each other")
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-5000:]
    with open(output_dir / "trainer_state.json") as f:
        logs = json.load(f)["log_history"]
    return {"output_dir": output_dir, "output": output, "logs": logs}


def _check_run(run: dict, labeled: dict) -> None:
    hidden_size = labeled["hidden_size"]
    # The head is counted and grouped by its real sizes, so its two matrices decay and its norm and biases do not
    head = 2 * hidden_size + 2 * (hidden_size * HEAD_DIM + HEAD_DIM)
    match = re.search(r"Optimizer: ([\d,]+) head parameters at lr [\d.e-]+ \(([\d,]+) with weight decay", run["output"])
    assert match, "no optimizer line in the output"
    assert int(match.group(1).replace(",", "")) == head
    assert int(match.group(2).replace(",", "")) == 2 * hidden_size * HEAD_DIM

    train_logs = [entry for entry in run["logs"] if "loss" in entry]
    assert len(train_logs) == 3
    for entry in train_logs:
        assert all(value == value for value in entry.values() if isinstance(value, float))  # no NaN
    assert any("anchor_kl" in entry for entry in train_logs) and any("teacher_kl" in entry for entry in train_logs)
    assert any("eval_loss" in entry for entry in run["logs"])

    # The adapter holds the trained head, whole (not a rank's empty shard)
    weights = load_file(run["output_dir"] / "adapter_model.safetensors")
    head_weights = {key: value for key, value in weights.items() if ".decision_head." in key}
    q_weight = next(value for key, value in head_weights.items() if key.endswith("q.weight"))
    assert tuple(q_weight.shape) == (HEAD_DIM, hidden_size) and q_weight.abs().sum() > 0
    assert any("lora_A" in key for key in weights)


def _reload_probabilities(run: dict, labeled: dict) -> list:
    """The probabilities of two prompts, with the saved adapter and head reloaded without DeepSpeed."""
    from peft import PeftModel

    from hmf.data.decision_prompt import StrandsPromptFormat, encode_decision_prompt

    model = AutoModel.from_pretrained(labeled["model_dir"], dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(run["output_dir"])).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(labeled["model_dir"])
    options = [("refund", "return it"), ("ignore", None), ("escalate", "to a human")]
    rendered = StrandsPromptFormat.render("state words", "question ?", "choice", options)
    encoded = encode_decision_prompt(tokenizer, rendered)
    with torch.no_grad():
        out = model(
            input_ids=torch.tensor([encoded.input_ids], device="cuda"),
            attention_mask=torch.ones(1, len(encoded.input_ids), dtype=torch.long, device="cuda"),
            opt_idx=torch.tensor([encoded.opt_idx], device="cuda"),
            answer_idx=torch.tensor([encoded.answer_idx], device="cuda"),
        )
    return out.log_probs[0].exp().tolist()


@needs_two_gpus
@pytest.mark.parametrize("labeled", ["qwen3", "qwen3_5", "gka"], indirect=True)
@pytest.mark.parametrize("stage", ["z2", "z3"])
def test_deepspeed_training(labeled, stage):
    pytest.importorskip("deepspeed")
    baseline = _train(labeled, "ddp")
    run = _train(labeled, stage, deepspeed=stage)

    _check_run(baseline, labeled)
    _check_run(run, labeled)

    # The first step's losses, computed before any update, agree with plain multi-GPU training (bf16)
    first, first_baseline = run["logs"][0], baseline["logs"][0]
    for key in ("loss", "ce_loss"):
        assert first[key] == pytest.approx(first_baseline[key], rel=2e-2, abs=1e-3), key

    probs = _reload_probabilities(run, labeled)
    assert len(probs) == 3 and sum(probs) == pytest.approx(1.0, abs=1e-3)


@needs_two_gpus
def test_zero3_checkpointing_of_long_batches_on_both_gpus(tmp_path):
    """
    Long and short rows mixed at random, so that in most steps one GPU's batch is long and the other's short.
    Without the same checkpointing choice on both, the GPUs wait on each other forever under ZeRO-3.
    """
    pytest.importorskip("deepspeed")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = []
    for i in range(40):
        kind = ("noul", "choice", "score")[i % 3]
        options = {"noul": [["false", None], ["true", None]], "choice": [["refund", None], ["ignore", None]],
                   "score": [["0", "low"], ["1", "mid"], ["2", "high"]]}[kind]
        state = "state " + "words " * (150 if i % 2 == 0 else 2)  # about 200 tokens, or about 50
        rows.append({"kind": kind, "state": state, "instructions": "question ?", "options": options,
                     "label": 1, "task": ("a", "b")[i % 2]})
    (data_dir / "toy.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (data_dir / "dataset_info.json").write_text(
        json.dumps({"toy_decision": {"file_name": "toy.jsonl", "formatting": "decision"}})
    )
    labeled = _prepare(tmp_path, "qwen3", str(data_dir))

    # A batch of 2 rows is about 2 x 64 padded tokens if both are short, and 2 x 208 or more if one is long
    run = _train(labeled, "z3_mixed", deepspeed="z3", decision_checkpoint_min_tokens=300, max_steps=8,
                 eval_steps=8)

    train_logs = [entry for entry in run["logs"] if "loss" in entry]
    assert len(train_logs) == 8
