#!/usr/bin/env python3
"""
Self-contained sharded preprocessing for LlamaFactory.

Usage:
    hybridfactory preprocess config.yaml [--shard_size N] [--parallel_jobs J] [--cleanup]
    hmf preprocess config.yaml [--shard_size N] [--parallel_jobs J] [--cleanup]

    # Or invoked directly as a module:
    python -m hmf.data.sharded_preprocess_and_tokenize config.yaml [--shard_size N] [--parallel_jobs J] [--cleanup]
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Must be before any torch/CUDA imports

import argparse
import json
import multiprocessing
import shutil
import signal
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from omegaconf import OmegaConf
from typing import Literal

import yaml
from datasets import DatasetDict, concatenate_datasets, load_from_disk

from .loader import _load_single_dataset, _get_preprocessed_dataset, _get_sequence_parallel_dataset
from .parser import get_dataset_list
from .data_utils import merge_dataset
from .template import get_template_and_fix_tokenizer
from ..model import load_tokenizer
from ..hparams.parser import _parse_train_args
from ..extras.logging import get_logger

# Use spawn to avoid CUDA fork issues
mp_context = multiprocessing.get_context('spawn')


logger = get_logger(__name__)


def get_args_from_config(config_path: str):
    """Parse llamafactory args from config file, stripping training-specific settings."""
    dict_config = OmegaConf.load(Path(config_path).absolute())
    cfg = OmegaConf.to_container(dict_config)
    
    # Remove distributed training settings since we are just preprocessing
    for key in [
        "deepspeed",
        "fsdp", 
        "fsdp_config",
        "ddp_timeout",
        "ddp_find_unused_parameters",
        "local_rank",
        "bf16",
        "fp16",
        "torch_compile",
    ]:
        cfg.pop(key, None)
    
    # Use _parse_train_args directly to skip distributed training validation
    return _parse_train_args(cfg)


# === PHASE 0: PARALLEL DATASET ALIGNMENT ===

def align_single_dataset_worker(args: tuple) -> tuple[str, bool, str, int]:
    """Worker: align a single dataset."""
    dataset_name, config_path, output_path = args
    
    if Path(f"{output_path}/state.json").exists():
        try:
            ds = load_from_disk(output_path)
            return (dataset_name, True, "already complete", len(ds))
        except:
            pass
    
    try:
        model_args, data_args, training_args, _, _ = get_args_from_config(config_path)
        dataset_attr = get_dataset_list([dataset_name], data_args.dataset_dir)[0]
        dataset = _load_single_dataset(dataset_attr, model_args, data_args, training_args)
        
        dataset.reset_format()
        os.makedirs(output_path, exist_ok=True)
        dataset.save_to_disk(output_path)
        
        return (dataset_name, True, "success", len(dataset))
    except Exception as e:
        return (dataset_name, False, str(e)[:200], 0)


def phase0_parallel_align(config_path: str, datasets: list[str], aligned_dir: str, parallel_jobs: int) -> bool:
    """Phase 0: Align each dataset in parallel."""
    
    print("\n" + "=" * 60)
    print(f"PHASE 0: Aligning {len(datasets)} datasets")
    print("=" * 60)
    
    os.makedirs(aligned_dir, exist_ok=True)
    
    tasks = []
    for name in datasets:
        output_path = os.path.join(aligned_dir, name)
        if Path(f"{output_path}/state.json").exists():
            print(f"  {name}: already aligned, skipping", flush=True)
        else:
            tasks.append((name, config_path, output_path))
    
    if not tasks:
        print("All datasets already aligned", flush=True)
        return True
    
    # Hard limit to 4 workers - Phase 0 is I/O bound and RAM heavy
    num_workers = min(4, len(tasks))
    print(f"Processing {len(tasks)} datasets with {num_workers} workers...", flush=True)
    
    failed = []
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context) as executor:
        futures = {executor.submit(align_single_dataset_worker, t): t[0] for t in tasks}
        for future in as_completed(futures):
            name, success, msg, count = future.result()
            status = "✓" if success else "✗"
            extra = f"({count:,} examples)" if success else ""
            print(f"  {name}: {status} {msg} {extra}", flush=True)
            if not success:
                failed.append(name)
    
    if failed:
        print(f"\nFailed: {failed}", flush=True)
        return False
    return True


# === PHASE 1: MERGE + SHARD ===

def phase1_merge_and_shard(
    config_path: str,
    datasets: list[str],
    aligned_dir: str,
    shard_size: int,
    seed: int,
) -> tuple[bool, int]:
    """Phase 1: Merge aligned datasets and shard."""
    
    print("\n" + "=" * 60)
    print("PHASE 1: Merging and sharding")
    print("=" * 60)
    
    sharding_info_path = os.path.join(aligned_dir, "sharding_info.json")
    if Path(sharding_info_path).exists():
        with open(sharding_info_path) as f:
            info = json.load(f)
        print(f"Already complete: {info['num_shards']} shards ({info['total_examples']:,} examples)")
        return True, info["num_shards"]
    
    # Load aligned datasets
    print(f"Loading {len(datasets)} datasets...")
    aligned = []
    dataset_sizes = {}
    for name in datasets:
        path = os.path.join(aligned_dir, name)
        ds = load_from_disk(path, keep_in_memory=True)
        aligned.append(ds)
        dataset_sizes[name] = len(ds)
        print(f"  {name}: {len(ds):,} examples")
    
    total_available = sum(dataset_sizes.values())
    print(f"Total available: {total_available:,} examples")
    
    _, data_args, _, _, _ = get_args_from_config(config_path)
    
    max_merged_samples = getattr(data_args, 'max_merged_samples', None)

    # interleave_fast requires max_merged_samples - default to total if not specified
    if max_merged_samples is None and data_args.mix_strategy == "interleave_fast":
        max_merged_samples = total_available
        print(f"max_merged_samples: {max_merged_samples:,} (auto-set for interleave_fast)")
    elif max_merged_samples is not None:
        print(f"max_merged_samples: {max_merged_samples:,} (from config)")
    else:
        print("max_merged_samples: None (using full dataset)")
    
    # Merge datasets
    print(f"Merging with strategy '{data_args.mix_strategy}'...")
    import time
    start = time.time()
    
    merged = merge_dataset(
        all_datasets=aligned,
        data_args=data_args,
        seed=seed,
        max_merged_samples=max_merged_samples,
    )
    
    elapsed = time.time() - start
    total = len(merged)
    print(f"Merge complete: {total:,} samples in {elapsed:.1f}s")
    
    # Shuffle if not already shuffled by interleave strategies
    if data_args.mix_strategy == "concat":
        print("Shuffling merged dataset...")
        merged = merged.shuffle(seed=seed)
    
    # Calculate shards (ceil division)
    num_shards = (total + shard_size - 1) // shard_size
    
    print(f"Saving {num_shards} shards (shard_size={shard_size:,})...")
    for i in tqdm(range(num_shards)):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, total)
        
        shard = merged.select(range(start_idx, end_idx))
        shard.reset_format()
        shard.save_to_disk(os.path.join(aligned_dir, f"shard_{i:04d}"))
    
    # Save metadata
    info = {
        "num_shards": num_shards,
        "shard_size": shard_size,
        "total_examples": total,
        "total_available": total_available,
        "max_merged_samples": max_merged_samples,
        "mix_strategy": data_args.mix_strategy,
        "dataset_sizes": dataset_sizes,
    }
    with open(sharding_info_path, "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"Saved sharding_info.json")
    
    return True, num_shards


# === PHASE 2: PARALLEL TOKENIZATION ===

def tokenize_shard_worker(args: tuple) -> tuple[int, bool, str]:
    """Worker: tokenize a single shard using existing helpers."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    shard_index, config_path, aligned_dir, encoded_dir, stage = args
    
    output_path = os.path.join(encoded_dir, f"shard_{shard_index}")
    if Path(f"{output_path}/dataset_dict.json").exists():
        return (shard_index, True, "already complete")
    
    try:
        model_args, data_args, training_args, _, _ = get_args_from_config(config_path)
        data_args.preprocessing_num_workers = 1
        
        tokenizer_module = load_tokenizer(model_args)
        tokenizer = tokenizer_module["tokenizer"]
        template = get_template_and_fix_tokenizer(tokenizer, data_args)
        
        shard_path = os.path.join(aligned_dir, f"shard_{shard_index:04d}")
        dataset = load_from_disk(shard_path)
        
        dataset = _get_preprocessed_dataset(
            dataset=dataset,
            data_args=data_args,
            training_args=training_args,
            stage=stage,
            template=template,
            tokenizer=tokenizer,
            processor=tokenizer_module.get("processor"),
            is_eval=False,
        )
        
        if model_args.sequence_parallel_size > 1:
            dataset = _get_sequence_parallel_dataset(
                dataset=dataset,
                data_args=data_args,
                model_args=model_args,
                training_args=training_args,
                tokenizer=tokenizer,
                is_eval=False,
            )
        
        os.makedirs(output_path, exist_ok=True)
        DatasetDict({"train": dataset}).save_to_disk(output_path)
        return (shard_index, True, "success")
    
    except Exception as e:
        return (shard_index, False, str(e)[:300])


def phase2_parallel_tokenize(
    config_path: str,
    aligned_dir: str,
    encoded_dir: str,
    num_shards: int,
    parallel_jobs: int,
    stage: str,
) -> bool:
    """Phase 2: Tokenize shards in parallel."""
    
    print("\n" + "=" * 60)
    print(f"PHASE 2: Tokenizing {num_shards} shards")
    print("=" * 60)
    
    os.makedirs(encoded_dir, exist_ok=True)
    
    tasks = []
    for i in range(num_shards):
        if Path(f"{encoded_dir}/shard_{i}/dataset_dict.json").exists():
            print(f"  Shard {i}: already complete, skipping")
        else:
            tasks.append((i, config_path, aligned_dir, encoded_dir, stage))
    
    if not tasks:
        print("All shards already tokenized")
        return True
    
    print(f"Processing {len(tasks)} shards with {parallel_jobs} workers...")
    
    failed = []
    with ProcessPoolExecutor(max_workers=parallel_jobs, mp_context=mp_context) as executor:
        futures = {executor.submit(tokenize_shard_worker, t): t[0] for t in tasks}
        for future in as_completed(futures):
            idx, success, msg = future.result()
            print(f"  Shard {idx:4d}: {'✓' if success else '✗'} {msg}")
            if not success:
                failed.append(idx)
    
    if failed:
        print(f"\nFailed: {failed}")
        return False
    return True


# === PHASE 3: CONCATENATE ===

def phase3_concatenate(encoded_dir: str, output_path: str, num_shards: int) -> bool:
    """Phase 3: Concatenate tokenized shards."""
    
    print("\n" + "=" * 60)
    print("PHASE 3: Concatenating")
    print("=" * 60)
    
    missing = [i for i in range(num_shards) if not Path(f"{encoded_dir}/shard_{i}/dataset_dict.json").exists()]
    if missing:
        print(f"ERROR: Missing shards: {missing}")
        return False
    
    print(f"Loading {num_shards} shards into memory...")
    shards = []
    for i in tqdm(range(num_shards)):
        ds = load_from_disk(f"{encoded_dir}/shard_{i}", keep_in_memory=True)
        if isinstance(ds, DatasetDict):
            ds = ds["train"]
        shards.append(ds)
    
    print("Concatenating...")
    full = concatenate_datasets(shards)
    print(f"Total: {len(full):,} examples")
    
    print(f"Saving to {output_path}...")
    DatasetDict({"train": full}).save_to_disk(output_path)
    
    print("Done!")
    return True


def phase3_concatenate_fast(encoded_path: str, output_path: str, num_shards: int) -> bool:
    """Concatenate by hardlinking arrow files - instant, no data copy."""
    
    print("\n" + "=" * 60)
    print("PHASE 3: Concatenating (hardlinks)")
    print("=" * 60)

    train_out = Path(output_path) / "train"
    train_out.mkdir(parents=True, exist_ok=True)

    all_arrow_files = []
    for i in range(num_shards):
        shard_train = Path(f"{encoded_path}/shard_{i}/train")
        arrow_files = sorted(shard_train.glob("data-*.arrow"))
        all_arrow_files.extend(arrow_files)

    num_files = len(all_arrow_files)
    print(f"Hardlinking {num_files} arrow files...")

    data_files = []
    for idx, src in enumerate(tqdm(all_arrow_files)):
        new_name = f"data-{idx:05d}-of-{num_files:05d}.arrow"
        dst = train_out / new_name
        data_files.append({"filename": new_name})
        os.link(src, dst)

    state = {
        "_data_files": data_files,
        "_fingerprint": "merged",
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": "train",
    }
    with open(train_out / "state.json", "w") as f:
        json.dump(state, f)

    shutil.copy2(
        f"{encoded_path}/shard_0/train/dataset_info.json",
        train_out / "dataset_info.json"
    )

    with open(Path(output_path) / "dataset_dict.json", "w") as f:
        json.dump({"splits": ["train"]}, f)

    print(f"Done: {num_files} files hardlinked")
    return True


# === MAIN ===

def run_preprocess(
    config_path: str,
    shard_size: int = 250000,
    parallel_jobs: int = 20,
    cleanup: bool = False,
) -> int:
    """Main preprocessing entry point."""
    
    config_path = os.path.abspath(config_path)
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    stage = cfg.get("stage", "sft")
    assert stage in ("sft", "pt"), f"Only 'sft' and 'pt' stages are supported, got '{stage}'"
    assert not cfg.get("streaming", False), "Streaming mode is not supported for sharded preprocessing"
    
    tokenized_path = cfg["tokenized_path"]
    datasets = [d.strip() for d in cfg["dataset"].split(",")]
    batch_size = cfg.get("preprocessing_batch_size", 1000)
    seed = cfg.get("seed", 42)
    
    base_dir = Path(tokenized_path).parent
    run_name = Path(tokenized_path).name
    aligned_dir = str(base_dir / f".tmp_{run_name}_aligned")
    encoded_dir = str(base_dir / f".tmp_{run_name}_encoded")
    
    def signal_handler(signum, frame):
        print(f"\n\n Data preprocessing interrupted. Temporary directories that may need cleanup:")
        print(f"  - {aligned_dir}")
        print(f"  - {encoded_dir}")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    if shard_size % batch_size != 0:
        print(f"ERROR: shard_size ({shard_size}) must be multiple of batch_size ({batch_size})")
        return 1
    
    print("=" * 60)
    print("SHARDED PREPROCESSING")
    print("=" * 60)
    print(f"Config:     {config_path}")
    print(f"Stage:      {stage}")
    print(f"Datasets:   {len(datasets)}")
    print(f"Output:     {tokenized_path}")
    print(f"Aligned dir (temporary): {aligned_dir}")
    print(f"Encoded dir (temporary): {encoded_dir}")
    print(f"Shard Size: {shard_size:,}")
    print(f"Parallel:   {parallel_jobs}")
    print("=" * 60)
    
    if Path(f"{tokenized_path}/dataset_dict.json").exists():
        print(f"\nAlready complete: {tokenized_path}")
        return 0
    
    if not phase0_parallel_align(config_path, datasets, aligned_dir, parallel_jobs):
        return 1
    
    success, num_shards = phase1_merge_and_shard(
        config_path, datasets, aligned_dir, shard_size, seed
    )
    if not success:
        return 1
    
    if not phase2_parallel_tokenize(config_path, aligned_dir, encoded_dir, num_shards, parallel_jobs, stage):
        return 1
    
    if not phase3_concatenate_fast(encoded_dir, tokenized_path, num_shards):
        return 1
    
    if cleanup:
        print("\nCleaning up...")
        shutil.rmtree(aligned_dir, ignore_errors=True)
        shutil.rmtree(encoded_dir, ignore_errors=True)
    
    print("\n" + "=" * 60)
    print(f"COMPLETE: {tokenized_path}")
    print("=" * 60)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML config file")
    parser.add_argument("--shard_size", type=int, default=75000)
    parser.add_argument("--parallel_jobs", type=int, default=128)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    
    return run_preprocess(
        args.config, 
        args.shard_size, 
        args.parallel_jobs, 
        args.cleanup, 
    )

if __name__ == "__main__":
    sys.exit(main())