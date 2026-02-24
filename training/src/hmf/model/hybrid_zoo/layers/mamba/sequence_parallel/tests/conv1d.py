import torch
from hmf.model.hybrid_zoo.layers.mamba.mamba2 import (
    Mamba2,
    causal_conv1d_fn,
    rearrange,
)
from hmf.model.hybrid_zoo.layers.sp_p2p_utils import reorder_for_ssm_p2p

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


class RefConv1d(Mamba2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None
    ):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)

        # [z, x, B, C, dt]
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_inner - 2 * self.d_xb - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.d_xb, self.nheads],
            dim=-1,
        )

        xBC = causal_conv1d_fn(
            xBC.transpose(1, 2).contiguous(),
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            bias=self.conv1d.bias,
            activation=None,
            seq_idx=seq_idx,
        ).transpose(1, 2)

        return xBC


class ZigZagConv1d(Mamba2):
    def __init__(self, cp_rank, cp_size, cp_group, cp_stream, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cp_rank = cp_rank
        self.cp_size = cp_size
        self.cp_group = cp_group
        self.cp_stream = cp_stream

    def forward(
        self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None
    ):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)

        """
            This function, reorders the chunk, and adds the last three timesteps from the previous chunk to each chunk.
            This is done to support the conv operation, which aggregates the inputs at the last d_conv timesteps.
        """
        zxbcdt = reorder_for_ssm_p2p(
            zxbcdt,
            self.cp_group,
            self.cp_stream,
            self.cp_size,
            self.cp_rank,
            self.d_conv,
        )

        # [z, x, B, C, dt]
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_inner - 2 * self.d_xb - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.d_xb, self.nheads],
            dim=-1,
        )

        xBC = causal_conv1d_fn(
            xBC.transpose(1, 2).contiguous(),
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            bias=self.conv1d.bias,
            activation=None,
            seq_idx=seq_idx,
        ).transpose(1, 2)

        xBC = xBC[:, self.d_conv - 1 :, :].contiguous()
        y = xBC
        b, l, d = y.shape
        y = y.view(b // 2, l * 2, d)
        return y


if __name__ == "__main__":
    rank, world_size, device, cp_size, sp_group = init_sp_distr()
    distr_configs = (rank, world_size, device, cp_size, sp_group)

    args, dtype = get_argparser()
    args.sp = cp_size
    config_path = args.model_config

    out_dim = (
        args.hidden_dim * 3
    )  # As the conv1d in the mamba layers has 3x more channels

    input_tensor = torch.randn(
        args.batch_size,
        args.seqlen,
        args.hidden_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    dout_tensor = torch.randn(
        args.batch_size, args.seqlen, out_dim, device=device, dtype=dtype
    )

    state_dict_path = None

    state_dict, cp_out, cp_param_grads = get_local_result(
        ZigZagConv1d,
        dtype,
        input_tensor,
        dout_tensor,
        args.head_dim,
        distr_configs,
        config_path,
        state_dict_path,
    )

    rank, world_size, device, cp_size, sp_group = init_sp_distr()
    distr_configs = (rank, world_size, device, cp_size, sp_group)

    no_cp_out, no_cp_param_grads = get_global_result(
        RefConv1d,
        dtype,
        input_tensor,
        dout_tensor,
        args.head_dim,
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
