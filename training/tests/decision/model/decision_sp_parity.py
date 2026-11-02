"""
Checks decision models under sequence parallelism (SP) against the same models without it. Run with
torchrun on 2 or more GPUs (see run_decision_sp_parity.sh):

    torchrun --nproc-per-node 2 tests/decision/model/decision_sp_parity.py
    torchrun --nproc-per-node 4 tests/decision/model/decision_sp_parity.py --sp_size 2   # 2 SP groups

The test models are small decision models with random weights, whose backbones have two to four
layers and a hidden size of 128 (see BACKBONES). There is one backbone per SP strategy: GKA's P2P
state passing, GDN's all-gather, Mamba2, SWA, attention-only, and the Qwen3.5 dense HMF model. For
each of them, the reference is computed first on every rank, on the whole batch with plain flash
attention, before ring attention is patched in. The same weights are then run with SP, and the
checks compare:

    eval_log_probs      log-probabilities of a forward pass in eval mode
    train_loss          the decision loss in train mode (head dropout 0)
    grads               gradients after DDP's averaging (relative error and per-parameter cosine)
    grad_checkpointing  gradients with gradient checkpointing on
    lora                gradients of LoRA adapters and the head (PEFT model)
    sgd_steps           parameters after a few SGD steps on different batches
    head_dropout        with head dropout on, every rank of the SP group computes the same loss
    bad_length          a padded length that does not split into 2 * sp_size chunks is rejected
    backbone_check      the backbone check (see below) passes (also with LoRA)

The backbone check is the check that the decision trainer runs at its first batch, to make sure that
every layer of the backbone supports SP (see check_sequence_parallel_backbone). In bf16,
backbone_rejected also checks that the backbone check rejects two stock Hugging Face backbones and
names the right layers. The first is Qwen3.5, whose Gated DeltaNet layers have no SP
version. The second is Qwen3 with sliding-window attention, which ring attention does not support
because it ignores the window. HMF's own sliding-window layers are not affected, as they have their
own SP version (the swa backbone above).

The batch has right-padded rows of different lengths, with answer markers and options spread over
the chunks of every rank, including the last real token of the longest row.

The fp32 run is the main check of correctness. In fp32, the results with and without SP agree to
about 1e-6, so any error in the SP code shows up. It covers the attention-free backbones only, as
ring attention runs in bf16 only:

    TRITON_F32_DEFAULT=ieee NVIDIA_TF32_OVERRIDE=0 torchrun --nproc-per-node 2 \\
        tests/decision/model/decision_sp_parity.py --dtype fp32

By default, everything runs in bf16, which covers every backbone. In bf16, rounding makes the results
with and without SP differ slightly, so the comparisons have tolerances (see --tol_*). GKA's solver
amplifies this rounding (its layer alone differs by up to ~1% in bf16, and not at all in fp32), so
GKA backbones get looser bf16 tolerances for gradients and for losses after SGD steps
(GKA_BF16_TOL_*).

A backbone whose layers cannot be imported (e.g., without mamba_ssm installed) is skipped.
"""

import argparse
import datetime
import os
import traceback
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

import hmf.model.hybrid_zoo.models.model_register  # noqa: F401  # Registers hybrid and decision models
from hmf.model.hybrid_zoo.models.decision.configuration_decision import DecisionConfig
from hmf.model.hybrid_zoo.models.decision.modeling_decision import DecisionModel
from hmf.model.model_utils.sequence_parallel import apply_sequence_parallel
from hmf.model.model_utils.sequence_parallel_check import describe_backbone_problems, sequence_parallel_backbone_problems
from hmf.train.decision.trainer import decision_loss
from hmf.train.sft.workflow import set_sequence_parallel_group_recursive


VOCAB_SIZE = 64
SGD_STEPS = 3
SGD_LR = 0.05


# Backbones


def _hybrid_qwen3(pattern: str, **kwargs):
    from hmf.model.hybrid_zoo.models.hybrid_qwen3.configuration_hybrid_qwen3 import HybridQwen3Config

    return HybridQwen3Config(
        vocab_size=VOCAB_SIZE, hidden_size=128, intermediate_size=256, num_hidden_layers=len(pattern.split("-")),
        num_attention_heads=2, num_key_value_heads=1, head_dim=64, hybrid_override_pattern=pattern,
        tie_word_embeddings=True, **kwargs,
    )


def _qwen3_attention():
    from transformers import Qwen3Config

    return Qwen3Config(
        vocab_size=VOCAB_SIZE, hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=64, tie_word_embeddings=True,
    )


def _qwen3_5_hmf():
    from hmf.model.hybrid_zoo.models.qwen3_5_hmf.configuration_qwen3_5_hmf import Qwen3_5HMFTextConfig

    return Qwen3_5HMFTextConfig(
        vocab_size=VOCAB_SIZE, hidden_size=128, intermediate_size=256, num_hidden_layers=4, num_attention_heads=2,
        num_key_value_heads=1, head_dim=64, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=64, linear_value_head_dim=64, tie_word_embeddings=True,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        hybrid_override_pattern="GKA-GDN-GKA-*",
    )


BACKBONES: Dict[str, Callable] = {
    "gka": lambda: _hybrid_qwen3("GKA-*-GKA-*"),  # P2P state passing
    "gdn": lambda: _hybrid_qwen3("GDN-*-GDN-*"),  # all-gather
    "mamba2": lambda: _hybrid_qwen3("M2-*"),  # P2P state passing
    "swa": lambda: _hybrid_qwen3("SWA-*", swa_config={"window_size": 64}),  # neighbor exchange
    "attention": _qwen3_attention,  # stock Hugging Face model, ring attention only
    "qwen3_5_hmf": _qwen3_5_hmf,  # GKA, GDN, and attention
    # These have no attention layers, which ring attention runs in bf16 only, so they also run in fp32
    "gka_only": lambda: _hybrid_qwen3("GKA-GKA"),
    "gdn_only": lambda: _hybrid_qwen3("GDN-GDN"),
}
ATTENTION_FREE = ["gka_only", "gdn_only"]

# Backbones with GKA layers, and their looser tolerances in bf16 (see run_checks). In bf16, the GKA layer
# alone differs by up to ~1% with and without SP, which compounds over layers, and in sgd_steps over steps.
GKA_BACKBONES = {"gka", "gka_only", "qwen3_5_hmf"}
GKA_BF16_TOL_GRAD_REL, GKA_BF16_TOL_GRAD_COS = 0.2, 0.9
GKA_BF16_TOL_SGD_LOSS = 5e-2


def _qwen3_5_stock():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        vocab_size=VOCAB_SIZE, hidden_size=128, intermediate_size=256, num_hidden_layers=4, num_attention_heads=2,
        num_key_value_heads=1, head_dim=64, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=64, linear_value_head_dim=64, tie_word_embeddings=True,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    )


def _qwen3_sliding():
    from transformers import Qwen3Config

    return Qwen3Config(
        vocab_size=VOCAB_SIZE, hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=64, tie_word_embeddings=True,
        use_sliding_window=True, sliding_window=64, max_window_layers=0,  # every layer uses the window
    )


# Stock backbones that the backbone check must reject, each with its config, the layers the check must name,
# and the reason
REJECTED: Dict[str, tuple] = {
    "qwen3_5_stock": (_qwen3_5_stock, [0, 1, 2], "no sequence parallelism"),  # Hugging Face's GDN layers
    # Hugging Face's sliding window, which ring attention ignores (HMF's SWA layers have their own SP version)
    "qwen3_sliding": (_qwen3_sliding, [0, 1], "sliding-window attention"),
}

# The dtype of the backbones (--dtype)
DTYPE = torch.bfloat16


def build_model(name: str, attn_implementation: str, device: torch.device) -> DecisionModel:
    """A small decision model (see BACKBONES), with the head in float32 and the rest in DTYPE."""
    text_config = BACKBONES[name]() if name in BACKBONES else REJECTED[name][0]()
    config = DecisionConfig(text_config=text_config.to_dict(), decision_head_dim=16, decision_head_dropout=0.0)
    config._attn_implementation = attn_implementation
    config.text_config._attn_implementation = attn_implementation
    torch.manual_seed(0)
    model = DecisionModel(config).to(device=device, dtype=DTYPE)
    model.decision_head.float()
    return model


def broadcast_weights(model: nn.Module) -> None:
    """Makes every rank hold rank 0's weights."""
    for tensor in list(model.parameters()) + list(model.buffers()):
        dist.broadcast(tensor.data, src=0)


# Batches


def make_batch(seq_len: int, sp_size: int, seed: int, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Right-padded rows of different lengths, with options spread over the rows, so that answer markers and
    options fall in the chunks of every rank, including the very last position of the longest row.
    """
    gen = torch.Generator().manual_seed(seed)
    chunk = seq_len // (2 * sp_size)
    lengths = [seq_len, seq_len - chunk // 2 - 3, seq_len // 2 + 5, 3 * seq_len // 4]
    num_options = [5, 2, 3, 4]
    batch_size, width = len(lengths), max(num_options)
    input_ids = torch.full((batch_size, seq_len), 1, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
    opt_idx = torch.full((batch_size, width), -1, dtype=torch.long)
    label_dist = torch.zeros((batch_size, width))
    answer_idx = torch.zeros(batch_size, dtype=torch.long)
    for i, (length, k) in enumerate(zip(lengths, num_options)):
        input_ids[i, :length] = torch.randint(2, VOCAB_SIZE, (length,), generator=gen)
        attention_mask[i, :length] = 1
        # Options evenly spread from the first chunk to just before the answer marker
        opt_idx[i, :k] = torch.linspace(1, length - 2, k).long()
        answer_idx[i] = length - 1
        label_dist[i, torch.randint(0, k, (1,), generator=gen)] = 1.0
    batch = {
        "input_ids": input_ids, "attention_mask": attention_mask, "opt_idx": opt_idx, "answer_idx": answer_idx,
        "label_dist": label_dist, "weight": torch.ones(batch_size),
    }
    return {key: value.to(device) for key, value in batch.items()}


MODEL_INPUTS = ("input_ids", "attention_mask", "opt_idx", "answer_idx")


def loss_of(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(**{name: batch[name] for name in MODEL_INPUTS})
    return decision_loss(outputs.log_probs, batch)[0]


# Gradients


def trainable_grads(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.grad.detach().float().cpu() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}


def ddp_average(model: nn.Module) -> None:
    """Averages the gradients over all ranks, as DDP and DeepSpeed do."""
    world = dist.get_world_size()
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world


def compare_grads(ref: Dict[str, torch.Tensor], sp: Dict[str, torch.Tensor], tol_rel: float, tol_cos: float) -> tuple:
    """Returns (passed, message), from the global relative error and the lowest cosine over the parameters
    with a real gradient."""
    if ref.keys() != sp.keys():
        missing, extra = sorted(ref.keys() - sp.keys()), sorted(sp.keys() - ref.keys())
        return False, f"parameters differ: missing {missing[:3]}, extra {extra[:3]}"
    diff = sum(float((sp[n] - ref[n]).pow(2).sum()) for n in ref) ** 0.5
    norm = sum(float(ref[n].pow(2).sum()) for n in ref) ** 0.5
    rel = diff / max(norm, 1e-12)
    largest = max(float(g.norm()) for g in ref.values())
    cosines = {
        n: float(torch.nn.functional.cosine_similarity(sp[n].flatten(), ref[n].flatten(), dim=0))
        for n in ref if float(ref[n].norm()) > 1e-3 * largest
    }
    worst = min(cosines, key=cosines.get)
    passed = rel <= tol_rel and cosines[worst] >= tol_cos
    return passed, f"rel err {rel:.2e}, min cos {cosines[worst]:.5f} ({worst})"


def lora_wrap(model: nn.Module) -> nn.Module:
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(1)
    config = LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", modules_to_save=["decision_head"])
    model = get_peft_model(model, config)
    # LoRA's B starts at 0, which would make the adapters' gradients trivially equal, so it gets random weights
    for name, param in model.named_parameters():
        if "lora_B" in name:
            param.data.normal_(0, 0.02)
    return model


def sgd_run(model: nn.Module, batches: List[Dict[str, torch.Tensor]], sp: bool) -> tuple:
    """Losses and final parameters after one SGD step per batch."""
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=SGD_LR)
    losses = []
    for batch in batches:
        optimizer.zero_grad()
        loss = loss_of(model, batch)
        loss.backward()
        if sp:
            ddp_average(model)
        optimizer.step()
        losses.append(float(loss))
    return losses, {n: p.detach().float().cpu() for n, p in model.named_parameters()}


# The reference, without SP (before ring attention is patched in)


def reference(name: str, seq_len: int, sp_size: int, device: torch.device, lora: bool) -> dict:
    model = build_model(name, "flash_attention_2", device)
    broadcast_weights(model)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    batch = make_batch(seq_len, sp_size, seed=0, device=device)

    model.eval()
    with torch.no_grad():
        log_probs = model(**{n: batch[n] for n in MODEL_INPUTS}).log_probs.float().cpu()
    model.train()
    model.zero_grad()
    loss = loss_of(model, batch)
    loss.backward()
    result = {"initial": initial, "log_probs": log_probs, "loss": float(loss), "grads": trainable_grads(model)}

    model.load_state_dict(initial)
    batches = [make_batch(seq_len, sp_size, seed=s, device=device) for s in range(1, SGD_STEPS + 1)]
    result["sgd"] = sgd_run(model, batches, sp=False)

    if lora:
        model.load_state_dict(initial)
        peft_model = lora_wrap(model)
        peft_model.train()
        loss_of(peft_model, batch).backward()
        result["lora_grads"] = trainable_grads(peft_model)
    del model
    torch.cuda.empty_cache()
    return result


# The checks, with SP


def run_checks(name: str, ref: dict, seq_len: int, group, sp_size: int, device: torch.device, args) -> List[tuple]:
    results = []
    tol_rel, tol_cos, tol_sgd_loss = args.tol_grad_rel, args.tol_grad_cos, args.tol_loss
    if DTYPE == torch.bfloat16 and name in GKA_BACKBONES:
        # GKA's solver amplifies bf16 rounding, so its gradients and its losses after SGD steps get looser
        # bounds here. The fp32 run (--dtype fp32) checks GKA exactly.
        tol_rel, tol_cos = max(tol_rel, GKA_BF16_TOL_GRAD_REL), min(tol_cos, GKA_BF16_TOL_GRAD_COS)
        tol_sgd_loss = max(tol_sgd_loss, GKA_BF16_TOL_SGD_LOSS)

    def check(label: str, fn: Callable[[], tuple]) -> None:
        try:
            passed, message = fn()
        except Exception as error:  # report and go on to the next check
            passed, message = False, f"{type(error).__name__}: {error}"
            if args.verbose:
                traceback.print_exc()
        results.append((name, seq_len, label, passed, message))

    def fresh(lora: bool = False) -> nn.Module:
        model = build_model(name, "sequence_parallel_attention", device)
        model.load_state_dict(ref["initial"])
        if lora:
            model = lora_wrap(model)
        set_sequence_parallel_group_recursive(model, group)
        return model

    batch = make_batch(seq_len, sp_size, seed=0, device=device)

    def eval_log_probs():
        model = fresh()
        model.eval()
        with torch.no_grad():
            log_probs = model(**{n: batch[n] for n in MODEL_INPUTS}).log_probs.float().cpu()
        finite = torch.isfinite(ref["log_probs"])
        err = float((log_probs - ref["log_probs"])[finite].abs().max())
        same_mask = torch.equal(torch.isfinite(log_probs), finite)
        return err <= args.tol_log_prob and same_mask, f"max abs err {err:.2e}"

    def train_loss_and_grads():
        model = fresh()
        model.train()
        loss = loss_of(model, batch)
        loss.backward()
        ddp_average(model)
        loss_err = abs(float(loss) - ref["loss"])
        results.append((name, seq_len, "train_loss", loss_err <= args.tol_loss, f"abs err {loss_err:.2e}"))
        return compare_grads(ref["grads"], trainable_grads(model), tol_rel, tol_cos)

    def grad_checkpointing():
        model = fresh()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        loss_of(model, batch).backward()
        ddp_average(model)
        return compare_grads(ref["grads"], trainable_grads(model), tol_rel, tol_cos)

    def lora_grads():
        model = fresh(lora=True)
        model.train()
        loss_of(model, batch).backward()
        ddp_average(model)
        return compare_grads(ref["lora_grads"], trainable_grads(model), tol_rel, tol_cos)

    def sgd_steps():
        model = fresh()
        model.train()
        batches = [make_batch(seq_len, sp_size, seed=s, device=device) for s in range(1, SGD_STEPS + 1)]
        losses, params = sgd_run(model, batches, sp=True)
        ref_losses, ref_params = ref["sgd"]
        loss_err = max(abs(a - b) for a, b in zip(losses, ref_losses))
        diff = sum(float((params[n] - ref_params[n]).pow(2).sum()) for n in ref_params) ** 0.5
        moved = sum(float((ref_params[n] - ref["initial"][n].float().cpu()).pow(2).sum()) for n in ref_params) ** 0.5
        rel = diff / max(moved, 1e-12)  # relative to how far the reference moved
        passed = loss_err <= tol_sgd_loss and rel <= tol_rel
        return passed, f"max loss err {loss_err:.2e}, param err {rel:.2e} of the update"

    def head_dropout():
        model = fresh()
        model.decision_head.dropout = nn.Dropout(0.5)
        model.train()
        loss = loss_of(model, batch).detach().reshape(1)
        losses = [torch.zeros_like(loss) for _ in range(sp_size)]
        dist.all_gather(losses, loss, group=group)
        same = all(torch.equal(losses[0], other) for other in losses)
        return same, f"losses on the group: {[round(float(x), 5) for x in losses]}"

    def bad_length():
        model = fresh()
        bad = {n: batch[n][:, : seq_len - 1] if batch[n].dim() == 2 and batch[n].size(1) == seq_len else batch[n]
               for n in MODEL_INPUTS}
        try:
            model(**bad)
        except ValueError as error:
            return "multiple of 2 * sequence_parallel_size" in str(error), "raised ValueError"
        return False, "no error"

    def backbone_check():
        messages = []
        for lora in ([False, True] if "lora_grads" in ref else [False]):
            model = fresh(lora=lora)
            problems = sequence_parallel_backbone_problems(model, lambda: model(**{n: batch[n] for n in MODEL_INPUTS}))
            if problems:
                return False, f"{'with LoRA: ' if lora else ''}{describe_backbone_problems(problems)}"
            messages.append("no problems" + (" (also with LoRA)" if lora else ""))
        return True, messages[-1]

    check("backbone_check", backbone_check)
    check("eval_log_probs", eval_log_probs)
    check("grads", train_loss_and_grads)
    check("grad_checkpointing", grad_checkpointing)
    if "lora_grads" in ref:
        check("lora", lora_grads)
    check("sgd_steps", sgd_steps)
    check("head_dropout", head_dropout)
    check("bad_length", bad_length)
    torch.cuda.empty_cache()
    return results


def rejected_checks(seq_len: int, group, sp_size: int, device: torch.device, args) -> List[tuple]:
    """The backbone check must reject each backbone of REJECTED, naming exactly its unsupported layers."""
    results = []
    for name, (_, expected_layers, reason) in REJECTED.items():
        try:
            model = build_model(name, "sequence_parallel_attention", device)
            broadcast_weights(model)
            set_sequence_parallel_group_recursive(model, group)
            batch = make_batch(seq_len, sp_size, seed=0, device=device)
            problems = sequence_parallel_backbone_problems(model, lambda: model(**{n: batch[n] for n in MODEL_INPUTS}))
            named = sorted(index for index, _, why in problems if why == reason)
            passed = named == expected_layers and len(problems) == len(expected_layers)
            message = f"named layers {named} (expected {expected_layers}): {describe_backbone_problems(problems)}"
        except ImportError as error:
            passed, message = True, f"skipped ({error})"
        except Exception as error:
            passed, message = False, f"{type(error).__name__}: {error}"
            if args.verbose:
                traceback.print_exc()
        results.append((name, seq_len, "backbone_rejected", passed, message))
        torch.cuda.empty_cache()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sp_size", type=int, default=None, help="SP group size (default: all ranks)")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                        help="fp32 runs the attention-free backbones only, and is exact with TRITON_F32_DEFAULT=ieee "
                             "NVIDIA_TF32_OVERRIDE=0 for exact fp32")
    parser.add_argument("--models", default=None,
                        help=f"comma-separated, from {list(BACKBONES)} (default: all for bf16, {ATTENTION_FREE} for fp32)")
    parser.add_argument("--seq_lens", default="512,1024", help="padded prompt lengths, multiples of 2 * sp_size")
    parser.add_argument("--no_lora", action="store_true", help="skip the LoRA check")
    parser.add_argument("--tol_log_prob", type=float, default=None, help="default 5e-2 (bf16), 1e-4 (fp32)")
    parser.add_argument("--tol_loss", type=float, default=None, help="default 2e-2 (bf16), 1e-4 (fp32)")
    parser.add_argument("--tol_grad_rel", type=float, default=None, help="default 5e-2 (bf16), 1e-3 (fp32)")
    parser.add_argument("--tol_grad_cos", type=float, default=None, help="default 0.995 (bf16), 0.99999 (fp32)")
    parser.add_argument("--verbose", action="store_true", help="print tracebacks of failed checks")
    args = parser.parse_args()

    global DTYPE
    DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    defaults = {"bf16": (5e-2, 2e-2, 5e-2, 0.995), "fp32": (1e-4, 1e-4, 1e-3, 0.99999)}[args.dtype]
    for name, default in zip(("tol_log_prob", "tol_loss", "tol_grad_rel", "tol_grad_cos"), defaults):
        if getattr(args, name) is None:
            setattr(args, name, default)
    if args.models is None:
        args.models = ",".join(ATTENTION_FREE if args.dtype == "fp32" else BACKBONES)
    if args.dtype == "fp32" and set(args.models.split(",")) - set(ATTENTION_FREE):
        raise SystemExit(f"--dtype fp32 supports the attention-free backbones only: {ATTENTION_FREE}")

    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=10))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    sp_size = args.sp_size or world
    seq_lens = [int(n) for n in args.seq_lens.split(",")]
    if world < 2 or world % sp_size or any(n % (2 * sp_size) for n in seq_lens):
        raise SystemExit(f"Needs >= 2 ranks, world ({world}) divisible by sp_size ({sp_size}), and seq_lens "
                         f"divisible by {2 * sp_size}.")

    # 1. The references, before ring attention replaces flash attention
    references, skipped = {}, []
    for name in args.models.split(","):
        try:
            for seq_len in seq_lens:
                references[(name, seq_len)] = reference(name, seq_len, sp_size, device, lora=not args.no_lora)
        except ImportError as error:
            skipped.append(f"{name} ({error})")
            references = {key: value for key, value in references.items() if key[0] != name}

    # 2. The same models with SP
    group = apply_sequence_parallel(SimpleNamespace(sequence_parallel_size=sp_size, sequence_parallel_mode="zigzag-ring"))
    results = []
    for (name, seq_len), ref in references.items():
        results += run_checks(name, ref, seq_len, group, sp_size, device, args)
    if args.dtype == "bf16":  # the rejected backbones have attention layers, which ring attention runs in bf16 only
        results += rejected_checks(seq_lens[0], group, sp_size, device, args)

    # Every rank must pass, and rank 0 reports the results
    passed = torch.tensor([all(r[3] for r in results)], device=device, dtype=torch.int)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"\nDecision SP parity: world {world}, sp_size {sp_size}, seq_lens {seq_lens}, {args.dtype}")
        for name, seq_len, label, ok, message in results:
            print(f"  {'PASS' if ok else 'FAIL'}  {name:<12} L={seq_len:<5} {label:<19} {message}")
        for entry in skipped:
            print(f"  SKIP  {entry}")
        print("ALL PASSED" if passed.item() else "SOME CHECKS FAILED (on rank 0 or another rank)")
    dist.barrier()
    dist.destroy_process_group()
    return 0 if passed.item() else 1


if __name__ == "__main__":
    raise SystemExit(main())
