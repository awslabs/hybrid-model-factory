"""
Needle-in-a-Haystack (NIAH) evaluation example for cache composition.

This script demonstrates how to use cache composition for long-context retrieval tasks.
It processes a context containing many key-value pairs and tests the model's ability
to retrieve specific values when queried.

Usage:
    python niah_example.py --model /path/to/model --num_chunks 2 --compose_type concat_kv_soup_ssm

The context is read from 'niah_context.txt' in the same directory.

Cache Composition Strategies:
    The compose_type argument selects a combination of KV cache strategy and SSM
    state combination strategy, formatted as ``{kv}_{ssm}``.

    KV cache strategies:
        full_kv   - Retain the full KV cache from a single concatenated forward pass.
                    Most accurate but highest memory usage.
        concat_kv - Process chunks independently in a batch, then concatenate KV
                    entries (excluding padding/prefix tokens) into a single cache.
        sw        - Sliding window: keep only the shared prefix and the last chunk's
                    KV cache, zeroing out intermediate tokens. Lowest memory.

    SSM state combination strategies:
        fuse      - Weighted combination of per-chunk SSM states using PICASO
                    coefficients (Mamba2/DeltaNet) or summation (GKA).
                    See https://arxiv.org/abs/2502.17605
        soup      - Simple averaging (model soup) of SSM states across chunks.
        kv_only   - Zero out all SSM states; rely solely on the KV cache.

    Available compose_type values:
        full_kv_fuse_ssm, full_kv_soup_ssm, full_kv_only,
        concat_kv_fuse_ssm, concat_kv_soup_ssm, concat_kv_only,
        sw_fuse_ssm, sw_soup_ssm, sw_kv_only
"""

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import hmf.model.hybrid_zoo.models.model_register
from hmf.model.hybrid_zoo.models.cache_compose import wrap_for_composition
import torch

# Replace with your model path
MODEL_PATH = "/path/to/your/hybrid/model"

# Task prompt for NIAH evaluation
TASK_PROMPT = (
    "A special magic number is hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the number afterwards."
)

# Test questions and expected answers
QUESTIONS = [
    "\nWhat is the special magic number for cynical-spotlight mentioned in the provided text? "
    "The special magic number for cynical-spotlight mentioned in the provided text is",
    "\nWhat is the special magic number for plucky-violin mentioned in the provided text? "
    "The special magic number for plucky-violin mentioned in the provided text is",
    "\nWhat is the special magic number for murky-hops mentioned in the provided text? "
    "The special magic number for murky-hops mentioned in the provided text is",
    "\nWhat is the special magic number for melted-elite mentioned in the provided text? "
    "The special magic number for melted-elite mentioned in the provided text is",
    "\nWhat is the special magic number for decorous-hippopotamus mentioned in the provided text? "
    "The special magic number for decorous-hippopotamus mentioned in the provided text is",
]
ANSWERS = ["5713926", "9894713", "1741958", "2176077", "9899342"]


def load_context() -> str:
    """Load context from niah_context.txt in the same directory."""
    context_path = Path(__file__).parent / "niah_context.txt"
    if not context_path.exists():
        raise FileNotFoundError(
            f"Context file not found: {context_path}\n"
            "Please create niah_context.txt with your NIAH context."
        )
    return context_path.read_text()


def main(model_path: str, num_chunks: int, compose_type: str):
    """
    Run NIAH evaluation with cache composition.
    
    Args:
        model_path: Path to the hybrid model checkpoint
        num_chunks: Number of chunks to split the context into for composition
        compose_type: Cache composition strategy (e.g. concat_kv_soup_ssm, sw_soup_ssm, full_kv_only)
    """
    context = load_context()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        dtype="auto", 
        device_map="auto", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Wrap model for cache composition
    composer_model = wrap_for_composition(model, tokenizer)
    
    # Tokenize task prompt and context
    task_prompt_ids = tokenizer(
        TASK_PROMPT, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(model.device)
    context_ids = tokenizer(
        context, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(model.device)

    # Warmup generation
    print("Warmup generation...")
    warmup_input = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
    _ = composer_model.generate(**warmup_input, max_new_tokens=10, use_cache=True)
    print("Warmup complete.\n")

    correct = 0
    for question, answer in zip(QUESTIONS, ANSWERS):
        question_ids = tokenizer(
            question, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)
        input_ids = torch.cat([context_ids, question_ids], dim=1)
        
        # Configure cache composition
        cache_compose_args = {
            "compose_type": compose_type,
            "num_chunks": num_chunks,
            "sequential_positions": False,
            "prefix_input_ids": task_prompt_ids,
            "suffix_input_ids": question_ids,
        }
        
        input_len = task_prompt_ids.shape[1] + context_ids.shape[1]
        
        outputs = composer_model.generate(
            input_ids=input_ids,
            max_new_tokens=20,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            use_cache=True,
            **cache_compose_args,
        )
        
        output_ids = outputs["sequences"] if isinstance(outputs, dict) else outputs
        generated_tokens = output_ids[0][input_len:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        is_correct = answer in output_text
        correct += int(is_correct)
        
        print(f"Q: {question.strip()}")
        print(f"Output: {output_text}")
        print(f"Expected: {answer} | {'✓' if is_correct else '✗'}\n")
    
    print(f"Accuracy: {100 * correct / len(ANSWERS):.1f}% ({correct}/{len(ANSWERS)})")


if __name__ == "__main__":
    import argparse
    
    COMPOSE_CHOICES = [
        "full_kv_fuse_ssm", "full_kv_soup_ssm", "full_kv_only",
        "concat_kv_fuse_ssm", "concat_kv_soup_ssm", "concat_kv_only",
        "sw_fuse_ssm", "sw_soup_ssm", "sw_kv_only",
    ]

    parser = argparse.ArgumentParser(
        description="NIAH evaluation with cache composition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=MODEL_PATH,
        help="Path to hybrid model checkpoint"
    )
    parser.add_argument(
        "--num_chunks", 
        type=int, 
        default=2, 
        help="Number of chunks for cache composition"
    )
    parser.add_argument(
        "--compose_type",
        type=str,
        default="concat_kv_soup_ssm",
        choices=COMPOSE_CHOICES,
        help="Cache composition strategy (default: concat_kv_soup_ssm)"
    )
    args = parser.parse_args()
    
    main(args.model, args.num_chunks, args.compose_type)
