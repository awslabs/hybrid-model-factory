"""
Converts an HMF decision model, trained with LoRA, to a Strands Decider checkpoint, so that Strands
Decider's own tools can serve, calibrate, and evaluate it (strands-decider serve, training.evaluate
calibrate and eval, evaluation/, and evaluation/jevbench/jevbench.sh for JevBench).

    python scripts/hmf_decision_to_strands.py \
        --model_name_or_path path/to/decision-model \
        --adapter_name_or_path path/to/lora-output \
        --output_dir path/to/strands-checkpoint

    # Also prepare a check that Strands Decider's loader gives the same probabilities, on the first
    # rows of a decision data file. It writes <output_dir>.check/; then, with Strands Decider's
    # environment, from its repository:
    python scripts/hmf_decision_to_strands.py ... --check_data path/to/strands-decider/data/multistep_v14_eval.jsonl
    python path/to/<output_dir>.check/strands_parity_check.py path/to/<output_dir>.check

A Strands Decider checkpoint is a folder with:

    strands_decider_config.json   the model's settings
    head.safetensors              the pointer head's weights
    lora/                         the LoRA adapter, on the base model's decoder
    tokenizer files

Strands Decider builds the decoder from base_model and applies lora/ to it. That is the same model as
ours as long as the decision model's backbone is base_model unchanged, which holds for a model converted with
`hmf decision-init` and trained with LoRA, where only the adapter and the head train. A model trained
with finetuning_type full cannot be converted this way.

The conversion only reads and renames files. The LoRA weights move from the decision model's backbone
(base_model.model.model.<layer>) to the bare decoder (base_model.model.<layer>), and the head's weights,
saved with the adapter, become head.safetensors. The checkpoint is uncalibrated (temperature 1), so run
Strands Decider's `python -m training.evaluate calibrate` before serving or evaluating it.

scripts/strands_to_hmf_decision.py converts in the other direction.
"""

import json
import os
import shutil
from typing import List, Optional

import fire
import torch
from safetensors.torch import load_file, save_file


# The pointer head's weights, as named in both implementations
HEAD_KEYS = {"norm.weight", "norm.bias", "q.weight", "q.bias", "k.weight", "k.bias"}
# Where PEFT saves the decision model's parameters, and where they go on Strands Decider's decoder
BACKBONE_PREFIX, DECODER_PREFIX = "base_model.model.model.", "base_model.model."

def _split_adapter(adapter_name_or_path: str) -> tuple:
    """The adapter's LoRA weights renamed for the bare decoder, and the head's weights."""
    weights = load_file(os.path.join(adapter_name_or_path, "adapter_model.safetensors"))
    lora, head = {}, {}
    for key, value in weights.items():
        if ".decision_head." in key:
            # PEFT saves the trained copy of a module in modules_to_save without the wrapper's names
            name = key.split(".decision_head.", 1)[1].replace("modules_to_save.", "").replace("default.", "")
            head[name] = value.float()
        elif key.startswith(BACKBONE_PREFIX):
            lora[DECODER_PREFIX + key[len(BACKBONE_PREFIX):]] = value
        else:
            raise ValueError(f"{adapter_name_or_path}: unexpected weight {key} in the adapter.")

    if set(head) != HEAD_KEYS:
        raise ValueError(
            f"{adapter_name_or_path}: the adapter's head weights are {sorted(head)}, expected {sorted(HEAD_KEYS)}. "
            "Was the head trained with the adapter (decision_head in modules_to_save) and with its LayerNorm "
            "(decision_head_norm: true)?"
        )
    if not lora:
        raise ValueError(f"{adapter_name_or_path}: the adapter has no LoRA weights.")
    return lora, head


def _strands_config(decision_config: dict, adapter_config: dict, base_model: str, max_length: int,
                    force_bos: bool, ordinal_smoothing: float) -> dict:
    """The settings of the Strands Decider checkpoint. Training-only settings keep their defaults."""
    targets = adapter_config["target_modules"]
    return {
        "base_model": base_model,
        "head_type": "pointer",
        "pointer_dim": decision_config["decision_head_dim"],
        "head_dropout": decision_config.get("decision_head_dropout", 0.0),
        "max_length": max_length,
        "torch_dtype": "bfloat16",
        "temperature": 1.0,
        "temperature_by_kind": {},
        "ordinal_smoothing": ordinal_smoothing,
        "use_lora": True,
        "lora_r": adapter_config["r"],
        "lora_alpha": adapter_config["lora_alpha"],
        "lora_dropout": adapter_config.get("lora_dropout", 0.0),
        "lora_targets": sorted(targets) if isinstance(targets, list) else [targets],
        "force_bos": force_bos,
    }


def _hmf_probs(model_name_or_path: str, adapter_name_or_path: Optional[str], rows: List[dict], max_length: int,
               force_bos: bool) -> List[Optional[List[float]]]:
    """Our probabilities over the options of each row (None for prompts longer than max_length)."""
    import hmf.model.hybrid_zoo.models.model_register  # noqa: F401  # Registers decision models
    from hmf.data.decision_teachers import DecisionModelTeacher

    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = DecisionModelTeacher(
        model_name_or_path, device, adapter_name_or_path=adapter_name_or_path, force_bos=force_bos,
        max_len=max_length,
    )
    probs = teacher(rows)
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs


def write_check(
    check_dir: str,
    checkpoint: str,
    model_name_or_path: str,
    adapter_name_or_path: Optional[str],
    check_data: str,
    num_check_rows: int,
    max_length: int,
    force_bos: bool,
) -> str:
    """
    Writes a check folder with the first rows of check_data, the decision model's probabilities for
    them, and strands_parity_check.py, which Strands Decider's environment runs to compare the probabilities
    of the Strands Decider checkpoint with them. Used by this script and by strands_to_hmf_decision.py.

    Args:
        check_dir: The folder to write.
        checkpoint: The Strands Decider checkpoint to compare with.
        model_name_or_path: The decision model.
        adapter_name_or_path: Its LoRA adapter, or None for a model without one.
        check_data: A decision data file.
        num_check_rows: How many of its first rows to check.
        max_length: The longest prompt checked, in tokens.
        force_bos: Whether prompts start with the BOS token.

    Returns:
        The check folder.
    """
    with open(check_data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()][:num_check_rows]
    ours = _hmf_probs(model_name_or_path, adapter_name_or_path, rows, max_length, force_bos)

    check_dir = os.path.abspath(check_dir)
    os.makedirs(check_dir, exist_ok=True)
    with open(os.path.join(check_dir, "rows.jsonl"), "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    with open(os.path.join(check_dir, "hmf_probs.json"), "w") as f:
        json.dump(ours, f)
    with open(os.path.join(check_dir, "meta.json"), "w") as f:
        json.dump({"checkpoint": os.path.abspath(checkpoint), "check_data": os.path.abspath(check_data)}, f, indent=2)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "strands_parity_check.py"), check_dir)
    return check_dir


def convert(
    model_name_or_path: str,
    adapter_name_or_path: str,
    output_dir: str,
    base_model: Optional[str] = None,
    max_length: int = 4096,
    force_bos: bool = False,
    ordinal_smoothing: float = 0.1,
    check_data: Optional[str] = None,
    num_check_rows: int = 64,
):
    """
    Args:
        model_name_or_path: The decision model the adapter was trained on (from hmf decision-init).
        adapter_name_or_path: The LoRA adapter trained with stage `decision` (the training output_dir).
        output_dir: Where to write the Strands Decider checkpoint. Must not exist.
        base_model: The causal LM the decision model was converted from, which Strands Decider loads
            the decoder from, as a Hugging Face id or a local folder. Defaults to the one recorded by
            hmf decision-init (source_model_name_or_path).
        max_length: The longest prompt Strands Decider serves, in tokens. Defaults to 4096.
        force_bos: Whether prompts start with the BOS token (decision_force_bos in training).
            Defaults to False.
        ordinal_smoothing: The ordinal smoothing of score targets in training
            (decision_ordinal_smoothing), which Strands Decider's server corrects score confidence
            for. Defaults to 0.1.
        check_data: A decision data file to check the conversion on: our probabilities for its first
            rows are written to <output_dir>.check, for strands_parity_check.py to compare with Strands
            Decider's. Defaults to None (no check).
        num_check_rows: How many of its first rows to check. Defaults to 64.
    """
    if os.path.exists(output_dir):
        raise ValueError(f"{output_dir} already exists.")
    if check_data is not None and not os.path.isfile(check_data):
        raise ValueError(f"check_data {check_data} does not exist.")
    with open(os.path.join(model_name_or_path, "config.json")) as f:
        decision_config = json.load(f)
    if decision_config.get("model_type") != "hmf_decision":
        raise ValueError(f"{model_name_or_path} is not a decision model (hmf decision-init writes one).")
    if not decision_config.get("decision_head_norm", True):
        raise ValueError("Strands Decider's pointer head always has a LayerNorm; this model's head has none.")
    with open(os.path.join(adapter_name_or_path, "adapter_config.json")) as f:
        adapter_config = json.load(f)

    base_model = base_model or decision_config.get("source_model_name_or_path")
    if base_model is None:
        raise ValueError("Set base_model: the decision model does not record the model it was converted from.")
    lora, head = _split_adapter(adapter_name_or_path)

    os.makedirs(os.path.join(output_dir, "lora"))
    save_file(head, os.path.join(output_dir, "head.safetensors"), metadata={"format": "pt"})
    save_file(lora, os.path.join(output_dir, "lora", "adapter_model.safetensors"))
    # The adapter now applies to the bare decoder, without the head
    adapter_config.update(base_model_name_or_path=base_model, modules_to_save=None, task_type=None)
    with open(os.path.join(output_dir, "lora", "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)
    with open(os.path.join(output_dir, "strands_decider_config.json"), "w") as f:
        config = _strands_config(decision_config, adapter_config, base_model, max_length, force_bos, ordinal_smoothing)
        json.dump(config, f, indent=2)

    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(model_name_or_path).save_pretrained(output_dir)
    print(f"Strands Decider checkpoint written to {output_dir} (base model {base_model}, uncalibrated).")

    if check_data is not None:
        check_dir = write_check(
            output_dir.rstrip("/") + ".check", output_dir, model_name_or_path, adapter_name_or_path, check_data,
            num_check_rows, max_length, force_bos,
        )
        print(
            f"Check written to {check_dir}. To run it, with Strands Decider's environment, from its repository:\n"
            f"    python {check_dir}/strands_parity_check.py {check_dir}"
        )


if __name__ == "__main__":
    fire.Fire(convert)
