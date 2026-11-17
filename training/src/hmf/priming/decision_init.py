"""
This file contains the code used to convert a causal LM into a decision model (see
model/hybrid_zoo/models/decision/). Instead of generating text, a decision model reads a prompt
that lists options and returns a probability for each option.

The conversion currently keeps the causal LM's backbone, drops its LM head, and adds a pointer
head, initialized as specified by decision_head_init. It can be used via the CLI:

    hmf decision-init path/to/decision_init_config.yaml

or in Python:

    config = load_config("path/to/config.yaml")
    init_decision_model(config)

The source model is any causal LM that loads with AutoModelForCausalLM, either a stock Transformer
or one of our Hybrid models. For multimodal checkpoints such as Qwen3.5 and Gemma 4, only the text
decoder is kept. Gemma 4's checkpoints are loaded with their multimodal class instead, since the causal
LM class does not load their text weights.
"""

import os
from dataclasses import dataclass, fields
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers
import yaml
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

import hmf.model.hybrid_zoo.models.model_register  # Registers hybrid models and suppresses warnings. Must be first
from hmf.model.hybrid_zoo.models.decision.configuration_decision import DecisionConfig
from hmf.model.hybrid_zoo.models.decision.modeling_decision import DecisionModel

from .hybridize_logger import get_logger
from .hybridize_model import DTYPE_MAP, load_pretrained_model
from .utils import copy_auxiliary_files


logger = get_logger(__name__)

# Gemma 4's model types, and the multimodal class that loads each one's text weights. The 12B is
# gemma4_unified, a class of its own with the same layout, and the 26B-A4B is gemma4 with a mixture of
# experts in every layer.
GEMMA4_CLASSES = {
    "gemma4": "Gemma4ForConditionalGeneration",
    "gemma4_unified": "Gemma4UnifiedForConditionalGeneration",
}


@dataclass
class DecisionInitConfig:
    """
    dataclass for converting a causal LM into a decision model.

    Attributes:
        base_model_name_or_path: The path or HF id of the source causal LM.
        output_dir: The folder in which the decision model will be saved.
        base_model_revision: The revision (branch, tag, or commit) of the source causal LM on the
            Hugging Face Hub, when base_model_name_or_path is an HF id. Defaults to None, which loads
            its default branch.
        dtype: The torch dtype of the backbone. The pointer head is always stored in float32.
            Defaults to bfloat16.
        decision_head_dim: Width of the pointer head's query and key projections. Defaults to 256.
        decision_head_norm: Whether the pointer head applies a LayerNorm before its projections.
            Defaults to True.
        decision_head_dropout: Dropout on the pointer head's inputs. 0 disables it. Defaults to 0.05.
        decision_head_init: How to initialize the pointer head. "random" uses PyTorch's default
            initialization. Defaults to "random".
        decision_attention: How the backbone attends over the prompt, "causal" or "bidirectional"
            (see DecisionConfig). Defaults to "causal".
        seed: Seed for the head's initialization. If None, the initialization is not seeded and
            differs between runs. Defaults to None.
        copy_auxiliary_files: Whether to copy auxiliary files (e.g., system prompts, tokenizer
            files) from the source model directory to the output directory. Defaults to False.
    """

    base_model_name_or_path: str
    output_dir: str
    base_model_revision: Optional[str] = None
    dtype: Optional[torch.dtype] = torch.bfloat16
    decision_head_dim: int = 256
    decision_head_norm: bool = True
    decision_head_dropout: float = 0.05
    decision_head_init: str = "random"
    decision_attention: str = "causal"
    seed: Optional[int] = None
    copy_auxiliary_files: bool = False


def load_config(path: str) -> DecisionInitConfig:
    """
    Given the path to a yaml file containing arguments like those found in DecisionInitConfig,
    loads the file and copies the arguments to an instance of DecisionInitConfig.

    Arguments:
        path: The path to the yaml file.

    Returns:
        An instance of DecisionInitConfig with the arguments specified in the @path file.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    unknown = set(data) - {f.name for f in fields(DecisionInitConfig)}
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")

    # Handle dtype field - convert string to torch dtype
    if data.get("dtype") is not None:
        dtype_str = data["dtype"].lower()
        if dtype_str not in DTYPE_MAP:
            raise ValueError(
                f"Unknown dtype '{data['dtype']}'. Supported values: {list(DTYPE_MAP.keys())}"
            )
        data["dtype"] = DTYPE_MAP[dtype_str]

    return DecisionInitConfig(**data)


def _init_head_random(decision_model: DecisionModel, causal_lm: PreTrainedModel) -> None:
    """Initializes the pointer head with PyTorch's default initialization."""
    decision_model.decision_head.reset_parameters()


# Head initialization methods, selected by decision_head_init. Each takes the decision model
# (backbone already in place) and the source causal LM, and initializes decision_model.decision_head.
HEAD_INIT_FNS: Dict[str, Callable[[DecisionModel, PreTrainedModel], None]] = {
    "random": _init_head_random,
}


def _head_kwargs(config: DecisionInitConfig) -> dict:
    """The DecisionConfig settings that decision-init takes from its config, which are the head's and the attention."""
    return {
        "decision_head_dim": config.decision_head_dim,
        "decision_head_norm": config.decision_head_norm,
        "decision_head_dropout": config.decision_head_dropout,
        "decision_head_init": config.decision_head_init,
        "decision_attention": config.decision_attention,
    }


def _load_source_model(
    source_path: str, dtype: torch.dtype
) -> Tuple[PreTrainedModel, nn.Module, PreTrainedTokenizer]:
    """
    Loads the source model, and returns it with its backbone and its tokenizer. The backbone is the
    causal LM without its LM head. Gemma 4's checkpoints are multimodal, and the causal LM class leaves
    their text weights randomly initialized, so they are loaded with their multimodal class, whose
    text decoder is the backbone. Their vision and audio encoders are not kept.

    Arguments:
        source_path: The source model's folder.
        dtype: The dtype to load it in.
    """
    model_type = AutoConfig.from_pretrained(source_path).model_type
    if model_type not in GEMMA4_CLASSES:
        causal_lm, tokenizer, _ = load_pretrained_model(source_path, dtype=dtype)
        return causal_lm, causal_lm.base_model, tokenizer

    class_name = GEMMA4_CLASSES[model_type]
    if not hasattr(transformers, class_name):
        raise ValueError(
            f"{source_path} is a {model_type} model, which needs transformers' {class_name}, and transformers "
            f"{transformers.__version__} does not have it."
        )
    model = getattr(transformers, class_name).from_pretrained(source_path, dtype=dtype)
    return model, model.model.language_model, AutoTokenizer.from_pretrained(source_path)


def init_decision_model(config: DecisionInitConfig) -> None:
    """
    Converts the causal LM specified in config into a decision model and saves it in the directory
    specified by config. Currently, this keeps the backbone, drops the LM head, and adds a pointer
    head initialized with HEAD_INIT_FNS[config.decision_head_init].

    Arguments:
        config: The config containing the arguments specified above. Typically the output of
            load_config.
    """
    logger.section("Decision Model Initialization")

    # Validate the head options before loading the source model
    if config.decision_head_init not in HEAD_INIT_FNS:
        raise ValueError(
            f"Unknown decision_head_init '{config.decision_head_init}'. "
            f"Supported values: {list(HEAD_INIT_FNS)}"
        )
    DecisionConfig(**_head_kwargs(config))

    # Load the source causal LM, at the requested revision if one is given
    model_name_or_path = config.base_model_name_or_path
    source_path = model_name_or_path
    if config.base_model_revision is not None:
        if os.path.isdir(model_name_or_path):
            raise ValueError(
                f"base_model_revision applies to a model on the Hugging Face Hub, but {model_name_or_path} is a "
                "local folder."
            )
        from huggingface_hub import snapshot_download

        source_path = snapshot_download(model_name_or_path, revision=config.base_model_revision)
        logger.info(f"Revision: {config.base_model_revision}")
    dtype = config.dtype or torch.bfloat16
    logger.info(f"Loading source model from: {model_name_or_path}")
    logger.info(f"Target dtype: {dtype}")
    causal_lm, backbone, tokenizer = _load_source_model(source_path, dtype)

    source_num_params = sum(p.numel() for p in causal_lm.parameters())
    logger.info(f"Source model loaded: {source_num_params:,} parameters")
    logger.info(f"Model type: {causal_lm.config.model_type}")
    text_config = backbone.config

    # Weights are dequantized when loading, so the saved backbone is no longer quantized
    if getattr(text_config, "quantization_config", None) is not None:
        logger.info("Removing quantization_config from backbone config (weights are no longer quantized)")
        del text_config.quantization_config

    decision_config = DecisionConfig(
        text_config=text_config,
        source_model_name_or_path=model_name_or_path,
        **_head_kwargs(config),
    )

    # Build the decision model on the meta device and plug the loaded backbone in, so the
    # backbone is neither re-initialized nor held in memory twice
    with torch.device("meta"):
        decision_model = DecisionModel(decision_config)

    if type(decision_model.model) is not type(backbone):
        raise ValueError(
            f"AutoModel builds {type(decision_model.model).__name__} from the backbone config, but the source "
            f"model's backbone is {type(backbone).__name__}. The saved checkpoint would not reload."
        )
    decision_model.model = backbone

    # Initialize the pointer head, in float32, on the backbone's device
    decision_model.decision_head.to_empty(device=backbone.device)
    if config.seed is not None:
        logger.info(f"Seeding head initialization with seed {config.seed}")
        torch.manual_seed(config.seed)
    HEAD_INIT_FNS[config.decision_head_init](decision_model, causal_lm)
    logger.info(f"Pointer head initialized: {config.decision_head_init}")
    del causal_lm

    decision_num_params = sum(p.numel() for p in decision_model.parameters())
    head_num_params = sum(p.numel() for p in decision_model.decision_head.parameters())
    param_diff = decision_num_params - source_num_params
    param_diff_pct = (param_diff / source_num_params) * 100

    # Save
    output_dir = config.output_dir
    logger.info(f"Saving decision model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    decision_model.save_pretrained(output_dir, max_shard_size="5GB")
    tokenizer.save_pretrained(output_dir)

    # Copy auxiliary files from source model (system prompts, tokenizer files, etc.)
    if config.copy_auxiliary_files:
        copy_auxiliary_files(source_path, output_dir)

    # Final summary
    logger.summary(
        f"Decision model initialization complete:\n"
        f"  Source model params:   {source_num_params:,}\n"
        f"  Decision model params: {decision_num_params:,}\n"
        f"  Pointer head params:   {head_num_params:,}\n"
        f"  Parameter change:      {param_diff:+,} ({param_diff_pct:+.2f}%)\n"
        f"  Saved to:              {output_dir}"
    )
