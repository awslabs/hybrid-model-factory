"""
SWA sequence parallelism test.

Compares the output and gradients of SWA with SP (zig-zag context parallelism)
against a single-GPU reference run on the full sequence. The gathered SP output
should match the single-GPU output up to floating-point precision.

Usage:
    torchrun --nproc-per-node <SP_SIZE> test_swa.py \
        --seqlen 4096 --dtype bf16 --model_config /path/to/config.json
"""

import torch
from transformers import AutoConfig
from utils import (
    print_stats,
    init_sp_distr,
    get_argparser,
    logs_to_json,
    get_local_result,
    get_global_result,
)


if __name__ == "__main__":
    rank, world_size, device, cp_size, sp_group = init_sp_distr()
    distr_configs = (rank, world_size, device, cp_size, sp_group)

    args, dtype = get_argparser()
    args.sp = cp_size

    config = AutoConfig.from_pretrained(args.model_config)

    # Ensure swa_config is set
    if not hasattr(config, "swa_config") or config.swa_config is None:
        raise ValueError(
            "Model config must have swa_config with window_size. "
            "Add {\"swa_config\": {\"window_size\": 4096}} to config.json."
        )

    # SWA SP requires flash_attention_2 — eager attention doesn't support
    # sliding_window and can't produce correct results with partial sequences.
    config._attn_implementation = "flash_attention_2"

    if rank == 0:
        print("##### Args #####")
        print("Args: ", args)
        print("dtype: ", dtype)
        print(f"swa_config: {config.swa_config}")
        print(f"_attn_implementation: {config._attn_implementation}")
        print("###############")

    input_tensor = torch.randn(
        args.batch_size,
        args.seqlen,
        args.hidden_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    dout_tensor = torch.randn(
        args.batch_size, args.seqlen, args.hidden_dim, device=device, dtype=dtype
    )

    state_dict_path = args.init_state_dict_path
    if state_dict_path is not None:
        args.exp_name = (
            args.exp_name
            + "_"
            + args.init_state_dict_path.split("/")[-1].replace(".pth", "")
        )
        print(args.exp_name)

    # Run with SP
    state_dict, cp_out, cp_param_grads = get_local_result(
        dtype,
        input_tensor,
        dout_tensor,
        distr_configs,
        config,
        state_dict_path,
    )

    # Re-initialize distributed for the global (no-SP) run
    rank, world_size, device, cp_size, sp_group = init_sp_distr()
    distr_configs = (rank, world_size, device, cp_size, sp_group)

    # Run without SP (reference)
    no_cp_out, no_cp_param_grads = get_global_result(
        dtype,
        input_tensor,
        dout_tensor,
        distr_configs,
        config,
        state_dict,
    )

    if rank == 0:
        logs_tensors_diffs = {}
        print_stats(cp_out, no_cp_out, "global output", logs_tensors_diffs)

        for name in no_cp_param_grads:
            print_stats(
                cp_param_grads[name],
                no_cp_param_grads[name],
                f"global {name} grad",
                logs_tensors_diffs,
            )

        logs_to_json(args, logs_tensors_diffs, exp_name=args.exp_name)
