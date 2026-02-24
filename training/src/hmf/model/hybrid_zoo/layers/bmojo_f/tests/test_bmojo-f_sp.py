import torch
from utils import (
    log,
    set_seed,
    extract_local,
    reconstruct_from_local,
    gather_from_local,
    print_stats,
    init_sp_group,
)
from utils import (
    init_sp_distr,
    print_logs,
    get_argparser,
    logs_to_json,
    get_local_result,
    get_global_result,
)

from hmf.model.hybrid_zoo.layers.bmojo_f.modules import BMojoLayer


if __name__ == "__main__":
    rank, world_size, device, cp_size, sp_group = init_sp_distr()
    distr_configs = (rank, world_size, device, cp_size, sp_group)

    args, dtype = get_argparser()
    args.sp = cp_size
    config_path = args.model_config

    batch_size = 1
    hidden_dim = 768
    # seqlen = 24
    seqlen = 4096 * 1

    if rank == 0:
        print("##### Args #####")
        print("Args: ", args)
        print("dtype: ", dtype)
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

    state_dict, cp_out, cp_param_grads = get_local_result(
        BMojoLayer,
        dtype,
        input_tensor,
        dout_tensor,
        args.window_size,
        distr_configs,
        config_path,
        state_dict_path,
    )

    rank, world_size, device, cp_size, sp_group = init_sp_distr()
    distr_configs = (rank, world_size, device, cp_size, sp_group)

    no_cp_out, no_cp_param_grads = get_global_result(
        BMojoLayer,
        dtype,
        input_tensor,
        dout_tensor,
        args.window_size,
        distr_configs,
        config_path,
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
