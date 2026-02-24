"""Model registration for Hybrid architectures.

This module registers custom hybrid model classes with the transformers library's
AutoModel classes, enabling them to be loaded via AutoModelForCausalLM.from_pretrained()
and similar methods.

Registration happens automatically when this module is imported. Models are registered
only once to avoid duplicate registration warnings.
"""

from typing import List, Type

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
from transformers.configuration_utils import PretrainedConfig

# Suppress "[ERROR] Config not found for hybrid-*" warnings from transformers auto_docstring
try:
    from transformers.utils.auto_docstring import HARDCODED_CONFIG_FOR_MODELS
    HARDCODED_CONFIG_FOR_MODELS.update({
        "hybrid-llama": "HybridLlamaConfig",
        "hybrid-ministral3": "HybridMinistral3Config",
        "hybrid-qwen2": "HybridQwen2Config",
        "hybrid-qwen3": "HybridQwen3Config",
        "hybrid-qwen3-moe": "HybridQwen3MoeConfig",
        "qwen3-next-hmf": "Qwen3NextHMFConfig",
        "qwen3-5-moe-hmf-text": "Qwen3_5MoeHMFTextConfig",
        "qwen3-5-moe-hmf": "Qwen3_5MoeHMFConfig",
    })
except ImportError:
    pass  # Older transformers versions may not have this module

_models_registered = False

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
    from .hybrid_qwen2.modeling_hybrid_qwen2 import (
        HybridQwen2ForCausalLM,
        HybridQwen2ForSequenceClassification,
        HybridQwen2ForTokenClassification,
        HybridQwen2Model,
    )
    register_model_family(
        HybridQwen2Config,
        [HybridQwen2ForCausalLM, HybridQwen2ForSequenceClassification, 
         HybridQwen2ForTokenClassification, HybridQwen2Model],
    )
    
    # Hybrid Qwen3
    from .hybrid_qwen3.configuration_hybrid_qwen3 import HybridQwen3Config
    from .hybrid_qwen3.modeling_hybrid_qwen3 import (
        HybridQwen3ForCausalLM,
        HybridQwen3ForSequenceClassification,
        HybridQwen3ForTokenClassification,
        HybridQwen3Model,
    )
    register_model_family(
        HybridQwen3Config,
        [HybridQwen3ForCausalLM, HybridQwen3ForSequenceClassification,
         HybridQwen3ForTokenClassification, HybridQwen3Model],
    )
    
    # Hybrid Qwen3 MoE
    from .hybrid_qwen3_moe.configuration_hybrid_qwen3_moe import HybridQwen3MoeConfig
    from .hybrid_qwen3_moe.modeling_hybrid_qwen3_moe import (
        HybridQwen3MoeForCausalLM,
        HybridQwen3MoeForSequenceClassification,
        HybridQwen3MoeForTokenClassification,
        HybridQwen3MoeModel,
    )
    register_model_family(
        HybridQwen3MoeConfig,
        [HybridQwen3MoeForCausalLM, HybridQwen3MoeForSequenceClassification,
         HybridQwen3MoeForTokenClassification, HybridQwen3MoeModel],
    )
    
    # Hybrid Llama
    from .hybrid_llama.configuration_hybrid_llama import HybridLlamaConfig
    from .hybrid_llama.modeling_hybrid_llama import (
        HybridLlamaForCausalLM,
        HybridLlamaForQuestionAnswering,
        HybridLlamaForSequenceClassification,
        HybridLlamaForTokenClassification,
        HybridLlamaModel,
    )
    register_model_family(
        HybridLlamaConfig,
        [HybridLlamaForCausalLM, HybridLlamaForQuestionAnswering,
         HybridLlamaForSequenceClassification, HybridLlamaForTokenClassification,
         HybridLlamaModel],
    )
    
    # Hybrid Ministral3
    from .hybrid_ministral3.configuration_hybrid_ministral3 import HybridMinistral3Config
    from .hybrid_ministral3.modeling_hybrid_ministral3 import (
        HybridMinistral3ForCausalLM,
        HybridMinistral3ForQuestionAnswering,
        HybridMinistral3ForSequenceClassification,
        HybridMinistral3ForTokenClassification,
        HybridMinistral3Model,
    )
    register_model_family(
        HybridMinistral3Config,
        [HybridMinistral3ForCausalLM, HybridMinistral3ForQuestionAnswering,
         HybridMinistral3ForSequenceClassification, HybridMinistral3ForTokenClassification,
         HybridMinistral3Model],
    )

    # Qwen3.5-MoE (HMF) — text-only (ForCausalLM uses TextConfig)
    from .qwen3_5_moe_hmf.configuration_qwen3_5_moe_hmf import (
        Qwen3_5MoeHMFConfig,
        Qwen3_5MoeHMFTextConfig,
    )
    from .qwen3_5_moe_hmf.modeling_qwen3_5_moe_hmf import (
        Qwen3_5MoeHMFForCausalLM,
        Qwen3_5MoeHMFForConditionalGeneration,
        Qwen3_5MoeHMFModel,
    )
    AutoConfig.register(Qwen3_5MoeHMFTextConfig.model_type, Qwen3_5MoeHMFTextConfig)
    AutoModelForCausalLM.register(Qwen3_5MoeHMFTextConfig, Qwen3_5MoeHMFForCausalLM)
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

    # Qwen3-Next (HMF)
    from .qwen3_next_hmf.configuration_qwen3_next_hmf import Qwen3NextHMFConfig
    from .qwen3_next_hmf.modeling_qwen3_next_hmf import (
        Qwen3NextHMFForCausalLM,
        Qwen3NextHMFForQuestionAnswering,
        Qwen3NextHMFModel,
        Qwen3NextHMFPreTrainedModel,
        Qwen3NextHMFForSequenceClassification,
        Qwen3NextHMFForTokenClassification,
    )
    register_model_family(
        Qwen3NextHMFConfig,
        [Qwen3NextHMFForCausalLM,
        Qwen3NextHMFForQuestionAnswering,
        Qwen3NextHMFModel,
        Qwen3NextHMFPreTrainedModel,
        Qwen3NextHMFForSequenceClassification,
        Qwen3NextHMFForTokenClassification],
    )
    
    _models_registered = True


register_hybrid_models()
