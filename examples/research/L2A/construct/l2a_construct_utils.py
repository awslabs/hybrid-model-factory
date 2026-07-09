"""Utility functions for copying weights from a base model to an L2A model during construction."""


def copy_shared_weights(base_model, hybrid_model, l2a_config):
    """
    Copies over the shared parameters from the base_model to the hybrid model.
    By shared parameters we mean all parameters except the hybrid layer parameters,
    which are handled separately.

    Args:
        base_model: The base model from which to copy the parameters.
        hybrid_model: The hybrid model to which the parameters will be copied.
        l2a_config: The L2A construction config.
    Returns:
        hybridization_candidates: List of parameter names that are L2A-specific.
    """
    base_model_state_dict = base_model.state_dict()
    hybrid_model_state_dict = hybrid_model.state_dict()

    hybridization_candidates = []

    for param_name in hybrid_model_state_dict.keys():
        if param_name in base_model_state_dict:
            print(f"Copying {param_name} to {param_name}")
            hybrid_model_state_dict[param_name].copy_(base_model_state_dict[param_name])
        else:
            print(f"{param_name} initialized randomly")
            hybridization_candidates.append(param_name)

    hybrid_model.load_state_dict(hybrid_model_state_dict)

    return hybridization_candidates


def copy_shared_weights_with_baseinit(base_model, hybrid_model, l2a_config):
    """
    Copies shared parameters from the base model to the L2A model, and initializes
    L2A-specific parameters (Global Attention projections) from their corresponding
    base model layers.

    For parameters that exist in both models (shared embeddings, MLP, norms, SWA projections),
    weights are copied directly. For L2A-specific parameters (suffixed with '_global'),
    the suffix is stripped to find the corresponding base layer and initialize from it.
    Router (sigmoid_linear) parameters are initialized randomly (or to zero if configured).

    Args:
        base_model: The base model from which to copy the parameters.
        hybrid_model: The L2A model to which the parameters will be copied.
        l2a_config: The L2A construction config.
    Returns:
        hybridization_candidates: List of parameter names that are L2A-specific.
    """
    base_model_state_dict = base_model.state_dict()
    hybrid_model_state_dict = hybrid_model.state_dict()

    hybridization_candidates = []

    for param_name in hybrid_model_state_dict.keys():
        if param_name in base_model_state_dict:
            print(f"Copying {param_name} to {param_name}")
            hybrid_model_state_dict[param_name].copy_(base_model_state_dict[param_name])
        else:
            if ("sigmoid_linear" in param_name or "alpha" in param_name):
                print(f"{param_name} initialized randomly")
                if l2a_config.sigmoid_linear_zero_init and "sigmoid_linear" in param_name:
                    print(f"{param_name} is initialized to zero")
                    hybrid_model_state_dict[param_name].zero_()
            else:
                # Map L2A-specific params back to base: strip _l2a or _global suffix
                base_layer = param_name.replace("_l2a", "").replace("_global", "")
                print(f"{param_name} initialized from {base_layer}")
                hybrid_model_state_dict[param_name].copy_(base_model_state_dict[base_layer])
            hybridization_candidates.append(param_name)

    hybrid_model.load_state_dict(hybrid_model_state_dict)

    return hybridization_candidates