"""
Converts a Strands Decider checkpoint to an HMF decision model, so that HMF can train, calibrate,
evaluate and serve it. It is the reverse of scripts/hmf_decision_to_strands.py.

    # A decision model and a LoRA adapter, as LoRA training in HMF writes them
    python scripts/strands_to_hmf_decision.py \
        --checkpoint StrandsAgents/strands-decider-2B-hobson-v19 \
        --output_dir path/to/strands-decider-2b

    # One decision model, with the adapter merged into its weights
    python scripts/strands_to_hmf_decision.py \
        --checkpoint StrandsAgents/strands-decider-2B-hobson-v19 \
        --output_dir path/to/strands-decider-2b-merged --merge

The checkpoint is a local folder or a model on the Hugging Face Hub. A Strands Decider checkpoint is a
folder with:

    strands_decider_config.json   the model's settings, among them base_model, the model it was trained on
    head.safetensors              the pointer head's weights
    lora/                         the LoRA adapter, on the base model's decoder
    tokenizer files

The base model's own weights are not in the checkpoint. The conversion builds the decision model from
the base model, at the revision the checkpoint records, as hmf decision-init does. Then:

    - Without --merge, it writes the decision model to <output_dir>/model and the LoRA adapter to
      <output_dir>/adapter. The adapter holds the LoRA weights and the head, as the full copy of
      decision_head that LoRA training in HMF saves. With --decision_model, an existing decision model
      of the same base model is used instead, and only <output_dir>/adapter is written.
    - With --merge, it merges the adapter into the decision model, as hmf export does, and writes one
      decision model to output_dir.

The converted model is the same model as the checkpoint. Its LoRA weights are renamed from the bare
decoder (base_model.model.<layer>) to the decision model's backbone (base_model.model.model.<layer>),
and the head's weights are copied unchanged. The checkpoint's calibration, if it has one, is written as
decision_calibration.json, next to the adapter or in the merged model's folder (see
src/hmf/eval/decision/README.md).

To keep training the converted adapter, set adapter_name_or_path to <output_dir>/adapter in a training
config with stage decision. The adapter keeps its own rank, alpha, and target modules, whatever
lora_rank, lora_alpha, and lora_target say. To train a new adapter instead, either add
create_new_adapter: true to that config, or convert with --merge and train on the merged model. Either
way, the training config must match the checkpoint's settings, which the conversion prints at the end.

Only checkpoints whose base model hmf decision-init converts can be converted. This includes the
Qwen3.5 releases, but not the Gemma 4 ones, whose multimodal checkpoints need loading code that HMF
does not have.

    # Also prepare a check that Strands Decider's loader gives the same probabilities, on the first
    # rows of a decision data file. It writes <output_dir>.check/; then, with Strands Decider's
    # environment, from its repository:
    python scripts/strands_to_hmf_decision.py ... --check_data path/to/strands-decider/data/multistep_v14_eval.jsonl
    python path/to/<output_dir>.check/strands_parity_check.py path/to/<output_dir>.check
"""

import importlib.util
import json
import os
import shutil
import tempfile
from typing import Dict, Optional

import fire
import torch
from safetensors.torch import load_file


# The settings file of a checkpoint, under its current name and under the name of checkpoints saved
# before Strands Decider was renamed
CONFIG_NAMES = ("strands_decider_config.json", "hobson_config.json")
# The pointer head's weights, as named in both implementations
HEAD_KEYS = {"norm.weight", "norm.bias", "q.weight", "q.bias", "k.weight", "k.bias"}
# Where PEFT saves the LoRA weights of Strands Decider's decoder, and where they go on the decision model's backbone
DECODER_PREFIX, BACKBONE_PREFIX = "base_model.model.", "base_model.model.model."
# Base models that hmf decision-init cannot convert yet, which are Gemma 4's multimodal checkpoints, whose
# text weights the causal LM class does not load
UNSUPPORTED_MODEL_TYPES = {"gemma4", "gemma4_unified"}


def _checkpoint_dir(checkpoint: str, revision: Optional[str]) -> str:
    """The checkpoint's folder, which is checkpoint itself if it is a local folder, else its download from the Hub."""
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint, revision=revision)


def _read_config(checkpoint_dir: str) -> dict:
    """The checkpoint's settings, after checking that the checkpoint is one this script converts exactly."""
    for name in CONFIG_NAMES:
        path = os.path.join(checkpoint_dir, name)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                config = json.load(f)
            break
    else:
        raise ValueError(f"{checkpoint_dir} is not a Strands Decider checkpoint: it has no {CONFIG_NAMES[0]}.")

    if config.get("head_type", "pointer") != "pointer":
        raise ValueError(f"{checkpoint_dir} has a {config['head_type']} head; only pointer heads can be converted.")
    if not config.get("use_lora", True) or config.get("full_weight_targets"):
        raise ValueError(f"{checkpoint_dir} was not trained with LoRA only; only LoRA checkpoints can be converted.")
    if not os.path.isdir(os.path.join(checkpoint_dir, "lora")):
        raise ValueError(f"{checkpoint_dir} has no lora/ folder.")
    if "base_model" not in config:
        raise ValueError(f"{checkpoint_dir}: its settings do not name the base model (base_model).")
    return config


def _base_revision(checkpoint_dir: str, config: dict) -> Optional[str]:
    """
    The revision of the base model the checkpoint was trained on, which is the one in its settings,
    else the one its provenance.json records for the same base model (as Strands Decider's loader reads it).
    """
    if config.get("base_revision"):
        return config["base_revision"]
    path = os.path.join(checkpoint_dir, "provenance.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        provenance = json.load(f)
    if provenance.get("base_model") != config["base_model"]:
        return None
    return provenance.get("base_model_revision")


def _check_base_model(base_model: str, revision: Optional[str]) -> None:
    """Refuses base models that hmf decision-init cannot convert, reading only their config.json."""
    if os.path.isdir(base_model):
        path = os.path.join(base_model, "config.json")
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(base_model, "config.json", revision=revision)
    with open(path, encoding="utf-8") as f:
        model_type = json.load(f).get("model_type")
    if model_type in UNSUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"The base model {base_model} is a {model_type} model, which hmf decision-init cannot convert yet."
        )


def _read_head(checkpoint_dir: str, pointer_dim: int) -> Dict[str, torch.Tensor]:
    """
    The pointer head's weights, in float32, read from head.safetensors, or from slot_head.pt in checkpoints
    saved before Strands Decider switched to safetensors.
    """
    path = os.path.join(checkpoint_dir, "head.safetensors")
    if os.path.isfile(path):
        head = load_file(path)
    elif os.path.isfile(os.path.join(checkpoint_dir, "slot_head.pt")):
        head = torch.load(os.path.join(checkpoint_dir, "slot_head.pt"), map_location="cpu", weights_only=True)
    else:
        raise ValueError(f"{checkpoint_dir} has no head weights (head.safetensors).")

    if set(head) != HEAD_KEYS:
        raise ValueError(f"{checkpoint_dir}: the head's weights are {sorted(head)}, expected {sorted(HEAD_KEYS)}.")
    if head["q.weight"].shape[0] != pointer_dim:
        raise ValueError(
            f"{checkpoint_dir}: the head's width is {head['q.weight'].shape[0]}, but its settings say {pointer_dim}."
        )
    return {name: value.float() for name, value in head.items()}


def _check_decision_model(decision_model: str, base_model: str, pointer_dim: int) -> None:
    """Checks that an existing decision model can take the checkpoint's adapter."""
    with open(os.path.join(decision_model, "config.json"), encoding="utf-8") as f:
        decision_config = json.load(f)
    if decision_config.get("model_type") != "hmf_decision":
        raise ValueError(f"{decision_model} is not a decision model (hmf decision-init writes one).")
    if decision_config.get("decision_head_dim") != pointer_dim or not decision_config.get("decision_head_norm", True):
        raise ValueError(
            f"{decision_model}'s head (width {decision_config.get('decision_head_dim')}) differs from the "
            f"checkpoint's (width {pointer_dim}, with a LayerNorm)."
        )
    source = decision_config.get("source_model_name_or_path")
    if source != base_model:
        print(
            f"Note: {decision_model} was converted from {source}, and the checkpoint was trained on {base_model}. "
            "They must be the same model, at the same revision."
        )


def _init_decision_model(base_model: str, revision: Optional[str], output_dir: str, config: dict) -> None:
    """Converts the base model into a decision model whose head has the checkpoint's shape."""
    from hmf.priming.decision_init import DecisionInitConfig, init_decision_model

    init_decision_model(
        DecisionInitConfig(
            base_model_name_or_path=base_model,
            base_model_revision=revision,
            output_dir=output_dir,
            dtype=getattr(torch, config.get("torch_dtype", "bfloat16")),
            decision_head_dim=config.get("pointer_dim", 256),
            decision_head_dropout=config.get("head_dropout", 0.0),
            seed=0,  # The head is replaced by the checkpoint's
        )
    )


def _check_tokenizer(checkpoint_dir: str, decision_model: str) -> None:
    """Checks that the checkpoint's tokenizer and the decision model's have the same vocabulary."""
    from transformers import AutoTokenizer

    if AutoTokenizer.from_pretrained(checkpoint_dir).get_vocab() != AutoTokenizer.from_pretrained(
        decision_model
    ).get_vocab():
        raise ValueError(f"The tokenizers of {checkpoint_dir} and of the decision model {decision_model} differ.")


def _write_adapter(checkpoint_dir: str, decision_model: str, head: Dict[str, torch.Tensor], adapter_dir: str) -> None:
    """
    Writes the checkpoint's LoRA adapter and head as an adapter of the decision model. PEFT builds the
    adapter on the decision model and saves it, so the folder is exactly what LoRA training in HMF saves,
    and every LoRA weight is checked against a module of the decision model.
    """
    from peft import PeftConfig, get_peft_model, set_peft_model_state_dict
    from transformers import AutoModel

    import hmf.model.hybrid_zoo.models.model_register  # noqa: F401  # Registers decision models

    lora_dir = os.path.join(checkpoint_dir, "lora")
    lora_config = PeftConfig.from_pretrained(lora_dir)
    if lora_config.peft_type != "LORA":
        raise ValueError(f"{lora_dir} is a {lora_config.peft_type} adapter; only LoRA adapters can be converted.")
    # The adapter now applies to the decision model, and saves the head with it
    lora_config.base_model_name_or_path = os.path.abspath(decision_model)
    lora_config.modules_to_save = ["decision_head"]
    lora_config.task_type = None
    lora_config.inference_mode = False

    weights = {}
    for key, value in load_file(os.path.join(lora_dir, "adapter_model.safetensors")).items():
        if not key.startswith(DECODER_PREFIX):
            raise ValueError(f"{lora_dir}: unexpected weight {key} in the adapter.")
        weights[BACKBONE_PREFIX + key[len(DECODER_PREFIX):]] = value
    if not weights:
        raise ValueError(f"{lora_dir}: the adapter has no LoRA weights.")
    # The head, under the name PEFT gives the saved copy of a module in modules_to_save
    weights.update({f"base_model.model.decision_head.{name}": value for name, value in head.items()})

    model = AutoModel.from_pretrained(decision_model, dtype="auto")
    peft_model = get_peft_model(model, lora_config)
    result = set_peft_model_state_dict(peft_model, weights)
    missing = [key for key in result.missing_keys if "lora_" in key]
    if result.unexpected_keys or missing:
        raise ValueError(
            f"The LoRA weights of {lora_dir} do not match the modules of the decision model {decision_model}: "
            f"{len(result.unexpected_keys)} unknown (e.g. {result.unexpected_keys[:2]}), {len(missing)} missing "
            f"(e.g. {missing[:2]})."
        )
    loaded_head = peft_model.base_model.model.decision_head.modules_to_save["default"].state_dict()
    if any(not torch.equal(loaded_head[name].cpu(), value) for name, value in head.items()):
        raise ValueError("PEFT did not load the checkpoint's head into the adapter's copy of decision_head.")
    peft_model.save_pretrained(adapter_dir)


def _write_calibration(folder: str, config: dict, checkpoint: str) -> Optional[str]:
    """
    Writes the checkpoint's temperatures as decision_calibration.json in folder, unless the checkpoint is
    uncalibrated (temperature 1 and no temperature by kind). Returns the file written, if any.
    """
    from hmf.eval.decision.predictor import CALIBRATION_FILE

    temperature = float(config.get("temperature", 1.0))
    by_kind = {kind: float(value) for kind, value in (config.get("temperature_by_kind") or {}).items()}
    if temperature == 1.0 and not by_kind:
        return None
    path = os.path.join(folder, CALIBRATION_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "temperature": temperature,
                "temperature_by_kind": by_kind,
                "ordinal_smoothing": config.get("ordinal_smoothing", 0.0),
                "source": f"the Strands Decider checkpoint {checkpoint}",
            },
            f,
            indent=2,
        )
    return path


def _merge(decision_model: str, adapter_dir: str, output_dir: str, dtype: str) -> None:
    """Merges the adapter into the decision model and saves the result to output_dir, as hmf export does."""
    from hmf.train.tuner import export_model

    export_model(
        {
            "model_name_or_path": decision_model,
            "adapter_name_or_path": adapter_dir,
            "finetuning_type": "lora",
            "stage": "decision",
            "export_dir": output_dir,
            "infer_dtype": dtype,
        }
    )


def _write_check(check_dir: str, checkpoint_dir: str, model: str, adapter: Optional[str], check_data: str,
                 num_check_rows: int, max_length: int, force_bos: bool) -> str:
    """Writes the check folder with hmf_decision_to_strands.py's code, which this script shares."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hmf_decision_to_strands.py")
    spec = importlib.util.spec_from_file_location("hmf_decision_to_strands", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.write_check(
        check_dir, checkpoint_dir, model, adapter, check_data, num_check_rows, max_length, force_bos
    )


def convert(
    checkpoint: str,
    output_dir: str,
    merge: bool = False,
    decision_model: Optional[str] = None,
    revision: Optional[str] = None,
    check_data: Optional[str] = None,
    num_check_rows: int = 64,
) -> None:
    """
    Args:
        checkpoint: The Strands Decider checkpoint: a local folder, or a model on the Hugging Face Hub.
        output_dir: Where to write the decision model. Must not exist.
        merge: Whether to merge the adapter into the decision model and write one model. Defaults to
            False, which writes the decision model and the adapter separately.
        decision_model: An existing decision model of the checkpoint's base model (from hmf
            decision-init), to use instead of converting the base model again. Defaults to None.
        revision: The revision of the checkpoint on the Hugging Face Hub. Defaults to None, which
            downloads its default branch.
        check_data: A decision data file to check the conversion on: the decision model's
            probabilities for its first rows are written to <output_dir>.check, for
            strands_parity_check.py to compare with Strands Decider's. Defaults to None (no check).
        num_check_rows: How many of its first rows to check. Defaults to 64.
    """
    if os.path.exists(output_dir):
        raise ValueError(f"{output_dir} already exists.")
    if check_data is not None and not os.path.isfile(check_data):
        raise ValueError(f"check_data {check_data} does not exist.")

    checkpoint_dir = _checkpoint_dir(checkpoint, revision)
    config = _read_config(checkpoint_dir)
    base_model, base_revision = config["base_model"], _base_revision(checkpoint_dir, config)
    pointer_dim = config.get("pointer_dim", 256)
    head = _read_head(checkpoint_dir, pointer_dim)
    if decision_model is not None:
        _check_decision_model(decision_model, base_model, pointer_dim)
    else:
        _check_base_model(base_model, base_revision)

    # With --merge, the decision model and the adapter are only steps, written next to output_dir
    # (on the same file system) and deleted afterwards
    output_dir = os.path.abspath(output_dir)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix=".strands_to_hmf_decision-", dir=os.path.dirname(output_dir)) if merge else None
    root = work_dir or output_dir
    try:
        if decision_model is None:
            decision_model = os.path.join(root, "model")
            _init_decision_model(base_model, base_revision, decision_model, config)
        _check_tokenizer(checkpoint_dir, decision_model)
        adapter_dir = os.path.join(root, "adapter")
        _write_adapter(checkpoint_dir, decision_model, head, adapter_dir)

        if merge:
            _merge(decision_model, adapter_dir, output_dir, config.get("torch_dtype", "bfloat16"))
            model, adapter = output_dir, None
        else:
            model, adapter = decision_model, adapter_dir
        calibration = _write_calibration(adapter or model, config, checkpoint)
    finally:
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)

    revision_note = f" at revision {base_revision}" if base_revision else ""
    print(f"Converted {checkpoint} (base model {base_model}{revision_note}).")
    print(f"  decision model: {model}")
    if adapter is not None:
        print(f"  adapter:        {adapter}")
    print(f"  calibration:    {calibration or 'none (the checkpoint is uncalibrated)'}")
    print(
        "Configs that train or evaluate it must match the checkpoint's settings:\n"
        "  decision_prompt_format: strands\n"
        f"  cutoff_len: {config.get('max_length', 4096)}\n"
        f"  decision_force_bos: {str(bool(config.get('force_bos', False))).lower()}\n"
        f"  decision_ordinal_smoothing: {config.get('ordinal_smoothing', 0.0)}"
    )
    if adapter is not None:
        print(
            f"Training with adapter_name_or_path: {adapter} continues its LoRA adapter (rank {config.get('lora_r')}, "
            f"alpha {config.get('lora_alpha')}), whatever lora_rank and lora_alpha say."
        )

    if check_data is not None:
        check_dir = _write_check(
            output_dir.rstrip("/") + ".check", checkpoint_dir, model, adapter, check_data, num_check_rows,
            config.get("max_length", 4096), bool(config.get("force_bos", False)),
        )
        print(
            f"Check written to {check_dir}. To run it, with Strands Decider's environment, from its repository:\n"
            f"    python {check_dir}/strands_parity_check.py {check_dir}"
        )


if __name__ == "__main__":
    fire.Fire(convert)
