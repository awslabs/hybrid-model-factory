"""Model registration for L2A architectures.

This module registers L2A model classes with the transformers library's
AutoModel classes, enabling them to be loaded via AutoModelForCausalLM.from_pretrained()
and similar methods.

Registration happens automatically when this module is imported.
"""

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
)

_models_registered = False


def register_module(module, module_name, config_class):
    obj_class = getattr(module, module_name)

    if module_name.endswith("ForCausalLM"):
        AutoModelForCausalLM.register(config_class, obj_class)
    elif module_name.endswith("ForSequenceClassification"):
        AutoModelForSequenceClassification.register(config_class, obj_class)
    elif module_name.endswith("ForQuestionAnswering"):
        AutoModelForQuestionAnswering.register(config_class, obj_class)
    elif module_name.endswith("ForTokenClassification"):
        AutoModelForTokenClassification.register(config_class, obj_class)
    elif module_name.endswith("Model"):
        AutoModel.register(config_class, obj_class)


def register_l2a_qwen3():
    from .l2a_qwen3.modeling_qwen3 import __all__ as all_qwen3_models
    from .l2a_qwen3.configuration_qwen3 import __all__ as all_qwen3_configs

    from .l2a_qwen3 import modeling_qwen3
    from .l2a_qwen3 import configuration_qwen3

    model_type = "l2a_qwen3"

    for qwen3_config in all_qwen3_configs:
        l2a_qwen3_config = getattr(configuration_qwen3, qwen3_config)
        AutoConfig.register(model_type, l2a_qwen3_config)

    for qwen3_model in all_qwen3_models:
        register_module(modeling_qwen3, qwen3_model, l2a_qwen3_config)


def register_l2a_models():
    """
    Register L2A model classes with transformers AutoModel classes.

    Simply importing this module is sufficient to trigger registration.
    """
    global _models_registered
    if not _models_registered:
        register_l2a_qwen3()
        _models_registered = True


register_l2a_models()