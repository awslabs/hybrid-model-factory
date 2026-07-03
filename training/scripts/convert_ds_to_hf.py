import os
import argparse
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

import hmf.model.hybrid_zoo.models.model_register
from hmf.priming.fused_to_standard import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Convert DeepSpeed Zero shards back to HF formatted checkpoints.")
    parser.add_argument("--base_model_name_or_path", type=str, required=True,
                        help="Path or Hub ID of the original model (e.g., meta-llama/Meta-Llama-3-8B)")
    parser.add_argument("--ds_checkpoint_dir", type=str, required=True,
                        help="Path to the directory containing the 'global_stepXXX' folder from DeepSpeed (e.g., output/checkpoint-500)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Target directory to save the clean HF model files.")
    parser.add_argument("--max_shard_size", type=str, default="5GB",
                        help="Maximum size of each individual weight shard (e.g., '5GB' or '10GB')")
    parser.add_argument("--safe_serialization", action="store_true", default=True,
                        help="Save model weights using safetensors format.")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Converting DeepSpeed shards from {args.ds_checkpoint_dir} directly into HF format at {args.output_dir}...")
    state_dict = get_fp32_state_dict_from_zero_checkpoint(args.ds_checkpoint_dir)

    model, config, tokenizer = load_model(args.base_model_name_or_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.save_pretrained(args.output_dir, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nSuccessfully converted DeepSpeed checkpoint to HF format at: {args.output_dir}")

if __name__ == "__main__":
    main()

