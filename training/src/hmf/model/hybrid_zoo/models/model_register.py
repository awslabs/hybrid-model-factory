"""Model registration for Hybrid architectures.

This module registers custom hybrid model classes with the transformers library's
AutoModel classes, enabling them to be loaded via AutoModelForCausalLM.from_pretrained()
and similar methods.

Registration happens automatically when this module is imported. Models are registered
only once to avoid duplicate registration warnings.

Without the CUDA-only libraries that the hybrid layers use (e.g., on a Mac), a model family whose
layers need them is registered with stand-in classes, which raise a clear error when such a model
is loaded. Every other model, decision models on transformers backbones included, still loads.
"""

import importlib
from typing import List, Optional, Type

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

# Suppress "[ERROR] Config not found for hybrid-*" warnings from transformers auto_docstring
try:
    from transformers.utils.auto_docstring import HARDCODED_CONFIG_FOR_MODELS
    HARDCODED_CONFIG_FOR_MODELS.update({
        "hybrid-llama": "HybridLlamaConfig",
        "hybrid-ministral3": "HybridMinistral3Config",
        "hybrid-qwen2": "HybridQwen2Config",
        "hybrid-qwen2-5-vl": "HybridQwen2_5_VLConfig",
        "hybrid-qwen3": "HybridQwen3Config",
        "hybrid-qwen3-moe": "HybridQwen3MoeConfig",
        "qwen3-next-hmf": "Qwen3NextHMFConfig",
        "qwen3-5-moe-hmf-text": "Qwen3_5MoeHMFTextConfig",
        "qwen3-5-moe-hmf": "Qwen3_5MoeHMFConfig",
        "qwen3-5-hmf-text": "Qwen3_5HMFTextConfig",
        "qwen3-5-hmf": "Qwen3_5HMFConfig",
    })
except ImportError:
    pass  # Older transformers versions may not have this module

_models_registered = False

# The model types registered with stand-ins, and the CUDA-only library each was first missing
_unavailable_model_types = {}

# Mapping from model class name suffix to AutoModel class
AUTO_MODEL_REGISTRY = {
    "ForCausalLM": AutoModelForCausalLM,
    "ForSequenceClassification": AutoModelForSequenceClassification,
    "ForQuestionAnswering": AutoModelForQuestionAnswering,
    "ForTokenClassification": AutoModelForTokenClassification,
    "Model": AutoModel,
}


def register_model_family(
    config_class: Type[PretrainedConfig],
    model_classes: List[type],
) -> None:
    """
    Register a hybrid model family with transformers AutoModel classes.
    
    Args:
        config_class: The configuration class (must have model_type attribute)
        model_classes: List of model classes to register
    """
    model_type = config_class.model_type
    AutoConfig.register(model_type, config_class)
    
    for model_class in model_classes:
        class_name = model_class.__name__
        for suffix, auto_class in AUTO_MODEL_REGISTRY.items():
            if class_name.endswith(suffix):
                auto_class.register(config_class, model_class)
                break


# The libraries that HMF's hybrid layers need and that only exist on CUDA machines
CUDA_ONLY_LIBRARIES = ("flash_attn", "fla", "mamba_ssm", "causal_conv1d", "triton", "deepspeed")


def _missing_cuda_library(error: ImportError) -> Optional[str]:
    """The CUDA-only library whose absence caused error, or None if it was caused by something else."""
    name = (error.name or "").split(".")[0]
    return name if name in CUDA_ONLY_LIBRARIES else None


def _unavailable_model(name: str, reason: str) -> type:
    """
    A stand-in for a model class whose layers cannot be imported: loading or building it raises
    reason. It has no config_class, so it can be registered with any of its family's configs.
    """

    def unavailable(*args, **kwargs):
        raise ImportError(f"{name} cannot be used here: {reason}")

    return type(
        name,
        (),
        {
            "from_pretrained": classmethod(unavailable),
            "_from_config": classmethod(unavailable),
            "__init__": unavailable,
        },
    )


def _import_models(module: str, names: List[str], config_class: Type[PretrainedConfig]) -> List[type]:
    """
    The model classes names from module (relative to this package). If module cannot be imported
    because a CUDA-only library is missing (e.g., on a Mac), stand-ins that raise a clear error when
    used are returned instead, so that every other model still registers and loads.

    Args:
        module: The modeling module, relative to this package (e.g., ".hybrid_qwen3.modeling_hybrid_qwen3").
        names: The names of the model classes to import from it.
        config_class: The configuration class of the models.

    Returns:
        The model classes, or their stand-ins, in the order of names.
    """
    try:
        imported = importlib.import_module(module, __package__)
    except ImportError as error:
        library = _missing_cuda_library(error)
        if library is None:
            raise
        _unavailable_model_types[config_class.model_type] = library
        reason = f"its layers need CUDA-only libraries, and {library} is not installed"
        return [_unavailable_model(name, reason) for name in names]
    return [getattr(imported, name) for name in names]


def register_hybrid_models():
    """
    Register hybrid model classes with transformers AutoModel classes.
    
    Note: It is not necessary to call this function directly - simply importing
    this module will trigger registration.
    """
    global _models_registered
    if _models_registered:
        return
    
    # Hybrid Qwen2
    from .hybrid_qwen2.configuration_hybrid_qwen2 import HybridQwen2Config
    register_model_family(
        HybridQwen2Config,
        _import_models(
            ".hybrid_qwen2.modeling_hybrid_qwen2",
            [
                "HybridQwen2ForCausalLM",
                "HybridQwen2ForSequenceClassification",
                "HybridQwen2ForTokenClassification",
                "HybridQwen2Model",
            ],
            HybridQwen2Config,
        ),
    )
    
    # Hybrid Qwen3
    from .hybrid_qwen3.configuration_hybrid_qwen3 import HybridQwen3Config
    register_model_family(
        HybridQwen3Config,
        _import_models(
            ".hybrid_qwen3.modeling_hybrid_qwen3",
            [
                "HybridQwen3ForCausalLM",
                "HybridQwen3ForSequenceClassification",
                "HybridQwen3ForTokenClassification",
                "HybridQwen3Model",
            ],
            HybridQwen3Config,
        ),
    )
    
    # Hybrid Qwen3 MoE
    from .hybrid_qwen3_moe.configuration_hybrid_qwen3_moe import HybridQwen3MoeConfig
    register_model_family(
        HybridQwen3MoeConfig,
        _import_models(
            ".hybrid_qwen3_moe.modeling_hybrid_qwen3_moe",
            [
                "HybridQwen3MoeForCausalLM",
                "HybridQwen3MoeForSequenceClassification",
                "HybridQwen3MoeForTokenClassification",
                "HybridQwen3MoeModel",
            ],
            HybridQwen3MoeConfig,
        ),
    )
    
    # Hybrid Llama
    from .hybrid_llama.configuration_hybrid_llama import HybridLlamaConfig
    register_model_family(
        HybridLlamaConfig,
        _import_models(
            ".hybrid_llama.modeling_hybrid_llama",
            [
                "HybridLlamaForCausalLM",
                "HybridLlamaForQuestionAnswering",
                "HybridLlamaForSequenceClassification",
                "HybridLlamaForTokenClassification",
                "HybridLlamaModel",
            ],
            HybridLlamaConfig,
        ),
    )
    
    # Hybrid Ministral3
    from .hybrid_ministral3.configuration_hybrid_ministral3 import HybridMinistral3Config
    register_model_family(
        HybridMinistral3Config,
        _import_models(
            ".hybrid_ministral3.modeling_hybrid_ministral3",
            [
                "HybridMinistral3ForCausalLM",
                "HybridMinistral3ForQuestionAnswering",
                "HybridMinistral3ForSequenceClassification",
                "HybridMinistral3ForTokenClassification",
                "HybridMinistral3Model",
            ],
            HybridMinistral3Config,
        ),
    )

    # Qwen3.5-MoE (HMF) — text-only (ForCausalLM uses TextConfig)
    from .qwen3_5_moe_hmf.configuration_qwen3_5_moe_hmf import (
        Qwen3_5MoeHMFConfig,
        Qwen3_5MoeHMFTextConfig,
    )
    (
        Qwen3_5MoeHMFForCausalLM,
        Qwen3_5MoeHMFForConditionalGeneration,
        Qwen3_5MoeHMFModel,
        Qwen3_5MoeHMFTextModel,
    ) = _import_models(
        ".qwen3_5_moe_hmf.modeling_qwen3_5_moe_hmf",
        [
            "Qwen3_5MoeHMFForCausalLM",
            "Qwen3_5MoeHMFForConditionalGeneration",
            "Qwen3_5MoeHMFModel",
            "Qwen3_5MoeHMFTextModel",
        ],
        Qwen3_5MoeHMFConfig,
    )
    AutoConfig.register(Qwen3_5MoeHMFTextConfig.model_type, Qwen3_5MoeHMFTextConfig)
    AutoModelForCausalLM.register(Qwen3_5MoeHMFTextConfig, Qwen3_5MoeHMFForCausalLM)
    # The text backbone alone, which a decision model rebuilds from its text_config. Registered in the
    # mapping directly, as the text model inherits config_class (the composite config) from its base class
    AutoModel._model_mapping.register(Qwen3_5MoeHMFTextConfig, Qwen3_5MoeHMFTextModel, exist_ok=True)
    # Qwen3.5-MoE (HMF) composite (ForConditionalGeneration uses full Config)
    register_model_family(
        Qwen3_5MoeHMFConfig,
        [Qwen3_5MoeHMFForConditionalGeneration, Qwen3_5MoeHMFModel],
    )
    # Also allow AutoModelForCausalLM to load from composite config
    # (bypasses config_class annotation check that auto_class.register() enforces)
    AutoModelForCausalLM._model_mapping.register(
        Qwen3_5MoeHMFConfig, Qwen3_5MoeHMFForCausalLM, exist_ok=True
    )

    # Qwen3.5 Dense (HMF) — text-only (ForCausalLM uses TextConfig)
    from .qwen3_5_hmf.configuration_qwen3_5_hmf import (
        Qwen3_5HMFConfig,
        Qwen3_5HMFTextConfig,
    )
    (
        Qwen3_5HMFForCausalLM,
        Qwen3_5HMFForConditionalGeneration,
        Qwen3_5HMFModel,
        Qwen3_5HMFTextModel,
    ) = _import_models(
        ".qwen3_5_hmf.modeling_qwen3_5_hmf",
        ["Qwen3_5HMFForCausalLM", "Qwen3_5HMFForConditionalGeneration", "Qwen3_5HMFModel", "Qwen3_5HMFTextModel"],
        Qwen3_5HMFConfig,
    )
    AutoConfig.register(Qwen3_5HMFTextConfig.model_type, Qwen3_5HMFTextConfig)
    AutoModelForCausalLM.register(Qwen3_5HMFTextConfig, Qwen3_5HMFForCausalLM)
    # The text backbone alone, which a decision model rebuilds from its text_config. Registered in the
    # mapping directly, as the text model inherits config_class (the composite config) from its base class
    AutoModel._model_mapping.register(Qwen3_5HMFTextConfig, Qwen3_5HMFTextModel, exist_ok=True)
    # Qwen3.5 Dense (HMF) composite (ForConditionalGeneration uses full Config)
    register_model_family(
        Qwen3_5HMFConfig,
        [Qwen3_5HMFForConditionalGeneration, Qwen3_5HMFModel],
    )
    AutoModelForCausalLM._model_mapping.register(
        Qwen3_5HMFConfig, Qwen3_5HMFForCausalLM, exist_ok=True
    )

    # Qwen3-Next (HMF)
    from .qwen3_next_hmf.configuration_qwen3_next_hmf import Qwen3NextHMFConfig
    register_model_family(
        Qwen3NextHMFConfig,
        _import_models(
            ".qwen3_next_hmf.modeling_qwen3_next_hmf",
            ["Qwen3NextHMFForCausalLM",
             "Qwen3NextHMFForQuestionAnswering",
             "Qwen3NextHMFModel",
             "Qwen3NextHMFPreTrainedModel",
             "Qwen3NextHMFForSequenceClassification",
             "Qwen3NextHMFForTokenClassification"],
            Qwen3NextHMFConfig,
        ),
    )
    

    # Hybrid Qwen2.5-VL — multimodal wrapper around HybridQwen2 text backbone
    from .hybrid_qwen2_5_vl.configuration_hybrid_qwen2_5_vl import (
        HybridQwen2_5_VLConfig,
    )
    HybridQwen2_5_VLForConditionalGeneration, HybridQwen2_5_VLModel = _import_models(
        ".hybrid_qwen2_5_vl.modeling_hybrid_qwen2_5_vl",
        ["HybridQwen2_5_VLForConditionalGeneration", "HybridQwen2_5_VLModel"],
        HybridQwen2_5_VLConfig,
    )
    AutoConfig.register(HybridQwen2_5_VLConfig.model_type, HybridQwen2_5_VLConfig)
    try:
        from transformers import AutoModelForImageTextToText
        AutoModelForImageTextToText.register(
            HybridQwen2_5_VLConfig, HybridQwen2_5_VLForConditionalGeneration
        )
    except ImportError:
        pass
    AutoModel.register(HybridQwen2_5_VLConfig, HybridQwen2_5_VLModel)

    # Decision models (any AutoModel backbone with a pointer head instead of an LM head)
    from .decision.configuration_decision import DecisionConfig
    from .decision.modeling_decision import DecisionModel
    register_model_family(DecisionConfig, [DecisionModel])

    if _unavailable_model_types:
        missing = sorted(set(_unavailable_model_types.values()))
        logger.warning(
            f"HMF's hybrid models are unavailable, as their layers need CUDA-only libraries that are not installed "
            f"({', '.join(missing)}): {', '.join(_unavailable_model_types)}. Other models, including decision "
            "models on transformers backbones, are unaffected."
        )
    _models_registered = True


register_hybrid_models()
