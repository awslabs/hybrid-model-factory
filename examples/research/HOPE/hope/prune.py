"""
Model pruning module.

Applies a pruning set (JSON mapping layer indices to expert indices) to a
HuggingFace MoE model checkpoint, producing a smaller model with fewer
experts per layer.
"""

import os
import json
import torch
import transformers


# Files that save_pretrained may not write but downstream tools may need
EXTRA_FILES_TO_LINK = [
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
]


def load_pruneset(path, num_moe_layers):
    """
    Load and validate a pruning set JSON file.

    Args:
        path: Path to JSON file mapping layer indices to lists of expert
            indices to prune.
        num_moe_layers: Expected number of MoE layers.

    Returns:
        Dict mapping int layer indices to sorted lists of int expert indices.
    """
    with open(path, "r") as f:
        pruneset = {
            int(key): sorted(vals) for key, vals in json.load(f).items()
        }

    if sorted(pruneset.keys()) != list(range(num_moe_layers)):
        raise ValueError(
            "Pruning set must cover all %d MoE layers, got: %s"
            % (num_moe_layers, sorted(pruneset.keys()))
        )
    counts = [len(v) for v in pruneset.values()]
    if len(set(counts)) != 1:
        raise ValueError(
            "Non-uniform pruning is not supported in this version. "
            "Got per-layer counts: %s" % dict(zip(pruneset.keys(), counts))
        )
    return pruneset


def _load_model(model_path):
    """
    Load the model, handling multimodal models (e.g. Qwen3.5) that need
    the full model class to preserve the vision encoder.

    Args:
        model_path: Path to HuggingFace model directory.

    Returns:
        The loaded model.
    """
    config = transformers.AutoConfig.from_pretrained(
        model_path, trust_remote_code=True
    )
    if hasattr(config, "text_config"):
        # Multimodal model: load the full model including vision encoder
        return transformers.Qwen3_5MoeForConditionalGeneration.from_pretrained(
            model_path, device_map="cpu", torch_dtype="auto",
            trust_remote_code=True,
        )
    return transformers.AutoModelForCausalLM.from_pretrained(
        model_path, device_map="cpu", torch_dtype="auto",
        trust_remote_code=True,
    )


def _get_layers_and_config(model):
    """
    Extract the transformer layers and MoE config from a model, handling
    different model architectures.

    Args:
        model: A loaded HuggingFace model.

    Returns:
        Tuple of (layers, moe_config).
    """
    arch = model.__class__.__name__
    if arch == "Qwen3_5MoeForConditionalGeneration":
        return model.model.language_model.layers, model.config.text_config
    if arch in ("Qwen3MoeForCausalLM", "Qwen3NextForCausalLM",
                "Qwen3_5MoeForCausalLM"):
        return model.model.layers, model.config
    if arch == "Glm4MoeForCausalLM":
        return model.model.layers, model.config
    raise ValueError("Unsupported architecture: %s" % arch)


def prune_moe_layer(moe_block, retain_inds):
    """
    Prune a single MoE layer in place, keeping only the specified experts.

    Args:
        moe_block: A SparseMoeBlock (``layer.mlp``) with ``.experts`` and
            ``.gate`` attributes.
        retain_inds: List of expert indices to retain.
    """
    experts = moe_block.experts
    router = moe_block.gate
    inds_ten = torch.tensor(retain_inds, dtype=torch.long)

    # Prune expert weights
    experts.gate_up_proj = torch.nn.Parameter(
        experts.gate_up_proj.data[inds_ten]
    )
    experts.down_proj = torch.nn.Parameter(
        experts.down_proj.data[inds_ten]
    )
    experts.num_experts = len(retain_inds)

    # Prune router weights
    router.weight = torch.nn.Parameter(router.weight.data[inds_ten])

    # Update Qwen3/Qwen3.5 attributes
    if hasattr(router, "num_experts"):
        router.num_experts = len(retain_inds)

    # Update GLM-4/GLM-4.5 attributes
    if hasattr(router, "n_routed_experts"):
        router.n_routed_experts = len(retain_inds)
    if hasattr(router, "e_score_correction_bias"):
        router.e_score_correction_bias = (
            router.e_score_correction_bias[inds_ten]
        )
    if hasattr(moe_block, "n_routed_experts"):
        moe_block.n_routed_experts = len(retain_inds)


def _link_extra_files(src_path, dst_path):
    """Symlink auxiliary files from the source model directory."""
    for fname in EXTRA_FILES_TO_LINK:
        src = os.path.realpath(os.path.join(src_path, fname))
        dst = os.path.join(dst_path, fname)
        if not os.path.exists(src):
            continue
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)


def prune_model(src_model_path, pruneset_path, dst_model_path):
    """
    Load a MoE model, apply a pruning set, and save the pruned checkpoint.
    For multimodal models (e.g. Qwen3.5), loads the full model including
    the vision encoder so the saved checkpoint is complete.

    Args:
        src_model_path: Path to the original HuggingFace model.
        pruneset_path: Path to the JSON pruning set.
        dst_model_path: Path to save the pruned model.
    """
    print("Loading model from %s ..." % src_model_path)
    model = _load_model(src_model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        src_model_path, trust_remote_code=True,
    )
    print("Found architecture: %s" % model.__class__.__name__)

    layers, moe_config = _get_layers_and_config(model)

    if hasattr(moe_config, "num_experts"):
        num_experts = moe_config.num_experts
    else:
        num_experts = moe_config.num_local_experts

    # Find MoE layers
    moe_layer_inds = [
        i for i, layer in enumerate(layers)
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts")
    ]
    num_moe_layers = len(moe_layer_inds)

    # Load pruning set
    pruneset = load_pruneset(pruneset_path, num_moe_layers)
    num_prune = len(next(iter(pruneset.values())))
    retained_count = num_experts - num_prune

    top_k = moe_config.num_experts_per_tok
    if retained_count < top_k:
        raise ValueError(
            "Pruning %d experts would leave %d per layer, which is fewer than "
            "top_k=%d" % (num_prune, retained_count, top_k)
        )

    print("Pruning %d/%d experts in each of %d layers" % (
        num_prune, num_experts, num_moe_layers
    ))

    # Apply pruning
    for moe_layer_i in sorted(pruneset.keys()):
        layer_i = moe_layer_inds[moe_layer_i]
        prune_inds = set(pruneset[moe_layer_i])
        retain_inds = [i for i in range(num_experts) if i not in prune_inds]
        prune_moe_layer(layers[layer_i].mlp, retain_inds)

    # Update config with retained count
    if hasattr(moe_config, "num_experts"):
        moe_config.num_experts = retained_count
    if hasattr(moe_config, "num_local_experts"):
        moe_config.num_local_experts = retained_count
    if hasattr(moe_config, "n_routed_experts"):
        moe_config.n_routed_experts = retained_count

    # Save pruned model
    os.makedirs(dst_model_path, exist_ok=True)
    print("Saving pruned model to %s ..." % dst_model_path)
    model.save_pretrained(dst_model_path, save_original_format=True)
    tokenizer.save_pretrained(dst_model_path)

    # Overwrite config.json from source with updated expert counts
    # This preserves the original config structure, avoiding issues with
    # different transformers versions rewriting fields (e.g. rope_parameters)
    src_config_path = os.path.join(src_model_path, "config.json")
    dst_config_path = os.path.join(dst_model_path, "config.json")
    with open(src_config_path, "r") as f:
        src_config = json.load(f)

    def _set(cfg, key, val):
        if key in cfg and cfg[key] is not None:
            cfg[key] = val
        if "text_config" in cfg:
            tc = cfg["text_config"]
            if key in tc and tc[key] is not None:
                tc[key] = val

    _set(src_config, "num_experts", retained_count)
    _set(src_config, "n_routed_experts", retained_count)
    _set(src_config, "num_local_experts", retained_count)

    with open(dst_config_path, "w") as f:
        json.dump(src_config, f, indent=2)

    _link_extra_files(src_model_path, dst_model_path)
    print("Saved pruned model with %d experts per layer." % retained_count)
