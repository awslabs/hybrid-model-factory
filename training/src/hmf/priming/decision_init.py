"""
This file contains the code used to convert a causal LM into a decision model (see
model/hybrid_zoo/models/decision/). Instead of generating text, a decision model reads a prompt
that lists options and returns a probability for each option.

The conversion keeps the causal LM's backbone, drops its LM head, and adds a decision head
(decision_head_type), initialized as specified by decision_head_init. The head is either a pointer
head, initialized at random, or a label head, which keeps the LM head's rows of the letter codes'
tokens (see model/hybrid_zoo/models/decision/). It can be used via the CLI:

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
from hmf.model.hybrid_zoo.models.decision.configuration_decision import LABEL_HEAD_TYPES, DecisionConfig
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
        dtype: The torch dtype of the backbone. The decision head is always stored in float32.
            Defaults to bfloat16.
        decision_head_type: The decision head, "pointer", "label", "label_trained" or "label_pointer"
            (see DecisionConfig). Defaults to "pointer".
        decision_head_dim: Width of the pointer head's query and key projections. Defaults to 256.
        decision_head_norm: Whether the pointer head applies a LayerNorm before its projections.
            Defaults to True.
        decision_head_dropout: Dropout on the pointer head's inputs. 0 disables it. Defaults to 0.05.
        decision_head_init: How to initialize the head. "random" uses PyTorch's default initialization,
            for the pointer head. "lm_head" copies the source LM's LM-head rows of the letter codes' tokens
            (see data/decision_prompt.py), for the label heads, and starts the pointer of a label_pointer
            head at random with its query projection zero. "auto" is "random" for the pointer head and
            "lm_head" for the label heads. Defaults to "auto".
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
    decision_head_type: str = "pointer"
    decision_head_dim: int = 256
    decision_head_norm: bool = True
    decision_head_dropout: float = 0.05
    decision_head_init: str = "auto"
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


def _init_head_random(
    decision_model: DecisionModel, causal_lm: PreTrainedModel, tokenizer: PreTrainedTokenizer
) -> None:
    """Initializes the pointer head with PyTorch's default initialization."""
    decision_model.decision_head.reset_parameters()


def _init_head_lm_head(
    decision_model: DecisionModel, causal_lm: PreTrainedModel, tokenizer: PreTrainedTokenizer
) -> None:
    """
    Initializes a label head with the source LM's LM-head rows of the letter codes' tokens, in code order, and the
    pointer of a label_pointer head with PyTorch's default initialization and a zero query projection.
    """
    from hmf.data.decision_prompt import letter_codes, single_token_id

    head = decision_model.decision_head
    head.reset_parameters()
    token_ids = [single_token_id(tokenizer, code) for code in letter_codes(tokenizer)]
    weight = causal_lm.get_output_embeddings().weight  # [vocab_size, hidden_size]
    head.copy_rows(weight[torch.tensor(token_ids, device=weight.device)].detach().to(head_device(head)))


def head_device(head: nn.Module) -> torch.device:
    """The device of a head's parameters or buffers."""
    return next(iter(list(head.parameters()) + list(head.buffers()))).device


# Head initialization methods, selected by decision_head_init. Each takes the decision model
# (backbone already in place), the source causal LM and its tokenizer, and initializes
# decision_model.decision_head.
HEAD_INIT_FNS: Dict[str, Callable[[DecisionModel, PreTrainedModel, PreTrainedTokenizer], None]] = {
    "random": _init_head_random,
    "lm_head": _init_head_lm_head,
}
# The initializations each head type takes, the first being the one "auto" picks
HEAD_INITS = {
    "pointer": ("random",),
    "label": ("lm_head",),
    "label_trained": ("lm_head",),
    "label_pointer": ("lm_head",),
}


def resolve_head_init(config: DecisionInitConfig) -> str:
    """The head initialization of a config, with "auto" resolved, checked against the head type."""
    if config.decision_head_type not in HEAD_INITS:
        raise ValueError(
            f"Unknown decision_head_type '{config.decision_head_type}'. Supported values: {list(HEAD_INITS)}"
        )
    allowed = HEAD_INITS[config.decision_head_type]
    if config.decision_head_init == "auto":
        return allowed[0]
    if config.decision_head_init not in HEAD_INIT_FNS:
        raise ValueError(
            f"Unknown decision_head_init '{config.decision_head_init}'. "
            f"Supported values: {['auto', *HEAD_INIT_FNS]}"
        )
    if config.decision_head_init not in allowed:
        raise ValueError(
            f"A {config.decision_head_type} head is initialized with {' or '.join(allowed)}, not "
            f"{config.decision_head_init}."
        )
    return config.decision_head_init


def _head_kwargs(
    config: DecisionInitConfig, num_labels: Optional[int] = None, softcap: Optional[float] = None
) -> dict:
    """
    The DecisionConfig settings that decision-init takes from its config, which are the head's and the attention,
    with a label head's number of rows and logit cap, which come from the source model.
    """
    return {
        "decision_head_type": config.decision_head_type,
        "decision_num_labels": num_labels,
        "decision_label_softcap": softcap,
        "decision_head_dim": config.decision_head_dim,
        "decision_head_norm": config.decision_head_norm,
        "decision_head_dropout": config.decision_head_dropout,
        "decision_head_init": resolve_head_init(config),
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
    is_label_head = config.decision_head_type in LABEL_HEAD_TYPES
    DecisionConfig(**_head_kwargs(config, num_labels=2 if is_label_head else None))

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
    num_labels, softcap = None, None
    if is_label_head:
        from hmf.data.decision_prompt import letter_codes

        num_labels = len(letter_codes(tokenizer))
        softcap = getattr(text_config, "final_logit_softcapping", None)
        logger.info(f"Label head: {num_labels} letter codes" + (f", logits capped at {softcap}" if softcap else ""))

    # Weights are dequantized when loading, so the saved backbone is no longer quantized
    if getattr(text_config, "quantization_config", None) is not None:
        logger.info("Removing quantization_config from backbone config (weights are no longer quantized)")
        del text_config.quantization_config

    decision_config = DecisionConfig(
        text_config=text_config,
        source_model_name_or_path=model_name_or_path,
        **_head_kwargs(config, num_labels, softcap),
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

    # Initialize the decision head, in float32, on the backbone's device
    decision_model.decision_head.to_empty(device=backbone.device)
    if config.seed is not None:
        logger.info(f"Seeding head initialization with seed {config.seed}")
        torch.manual_seed(config.seed)
    head_init = decision_config.decision_head_init
    HEAD_INIT_FNS[head_init](decision_model, causal_lm, tokenizer)
    logger.info(f"Decision head ({config.decision_head_type}) initialized: {head_init}")
    del causal_lm

    decision_num_params = sum(p.numel() for p in decision_model.parameters())
    head_num_params = sum(p.numel() for p in decision_model.decision_head.parameters())
    head_num_params += sum(b.numel() for b in decision_model.decision_head.buffers())
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
        f"  Decision head params:  {head_num_params:,}\n"
        f"  Parameter change:      {param_diff:+,} ({param_diff_pct:+.2f}%)\n"
        f"  Saved to:              {output_dir}"
    )
