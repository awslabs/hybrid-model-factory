"""Create symlinked SWA variants of a Qwen model.

For each layer index, creates a directory that symlinks all files from the
source model except config.json, which is modified to use HybridQwen*ForCausalLM
with a single SWA layer at that index.
"""
import argparse
import json
import os
import sys

ARCH_MAP = {
    "Qwen2ForCausalLM": ("HybridQwen2ForCausalLM", "hybrid_qwen2"),
    "Qwen3ForCausalLM": ("HybridQwen3ForCausalLM", "hybrid_qwen3"),
    "Qwen3MoeForCausalLM": ("HybridQwen3MoeForCausalLM", "hybrid_qwen3_moe"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Path to original Qwen model directory")
    parser.add_argument("--dst", required=True, help="Base directory for SWA variant outputs")
    parser.add_argument("--window-size", type=int, default=2048, help="SWA window size (default: 2048)")
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)

    with open(os.path.join(src, "config.json")) as f:
        base_config = json.load(f)

    orig_arch = base_config["architectures"][0]
    if orig_arch not in ARCH_MAP:
        print(f"Error: unsupported architecture '{orig_arch}'. Must be one of: {list(ARCH_MAP.keys())}", file=sys.stderr)
        sys.exit(1)

    hybrid_arch, hybrid_model_type = ARCH_MAP[orig_arch]
    num_layers = base_config["num_hidden_layers"]
    src_files = os.listdir(src)

    os.makedirs(dst, exist_ok=True)

    for swa_idx in range(num_layers):
        variant_dir = os.path.join(dst, f"swa_layer_{swa_idx}")
        os.makedirs(variant_dir, exist_ok=True)

        for fname in src_files:
            if fname == "config.json":
                continue
            dst_path = os.path.join(variant_dir, fname)
            if not os.path.exists(dst_path):
                os.symlink(os.path.join(src, fname), dst_path)

        pattern = "-".join("SWA" if i == swa_idx else "*" for i in range(num_layers))

        config = base_config.copy()
        config["architectures"] = [hybrid_arch]
        config["model_type"] = hybrid_model_type
        config["hybrid_override_pattern"] = pattern
        config["swa_config"] = {"window_size": args.window_size}

        with open(os.path.join(variant_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    print(f"Created {num_layers} SWA variants in {dst}")


if __name__ == "__main__":
    main()
