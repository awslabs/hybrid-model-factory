"""
Calibration module for HOPE expert pruning.

Collects pairwise expert interaction statistics (the F-matrix) by running
calibration prompts through a MoE model and recording router decisions and
expert output norms.
"""

import os
import numpy as np
import torch
import h5py
import tqdm
import transformers


class HOExpertObserver:
    """
    Attaches forward hooks to MoE expert modules to collect second-order
    expert interaction statistics during inference.

    Supported architectures: any HuggingFace MoE model where MoE layers
    have ``layer.mlp.experts`` with ``gate_up_proj``, ``down_proj``, and
    ``act_fn`` attributes (e.g. Qwen3 MoE, Qwen3.5 MoE, GLM-4.5).

    Args:
        model: A loaded HuggingFace CausalLM model.
        stats_on_gpu: If True, place accumulator tensors on the least-used
            GPU for faster computation. If False, use CPU.
    """

    def __init__(self, model, stats_on_gpu=True):
        self.hooks = []
        self.stats = {}

        if stats_on_gpu and torch.cuda.is_available():
            best_dev = max(
                range(torch.cuda.device_count()),
                key=lambda i: torch.cuda.mem_get_info(i)[0],
            )
            self.sdev = torch.device("cuda:%d" % best_dev)
        else:
            self.sdev = torch.device("cpu")

        self._install_hooks(model)

    def _install_hooks(self, model):
        """Finds MoE layers and registers stat-collecting hooks."""
        if hasattr(model, "model"):
            layers = model.model.layers
        else:
            layers = model.layers

        moe_blocks = [
            layer.mlp
            for layer in layers
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts")
        ]
        for layer_i, moe_block in enumerate(moe_blocks):
            experts = moe_block.experts
            if not hasattr(experts, "gate_up_proj") or not hasattr(experts, "down_proj"):
                raise ValueError(
                    "Expected experts module with `gate_up_proj` and `down_proj`"
                )
            if experts.gate_up_proj.dim() != 3 or experts.down_proj.dim() != 3:
                raise ValueError(
                    "Expected 3D stacked expert weights"
                )

            hook = moe_block.experts.register_forward_hook(
                self._make_hook(layer_i)
            )
            self.hooks.append(hook)

    def _make_hook(self, layer_i):
        """Creates a hook function that collects expert-interaction stats."""

        @torch.no_grad()
        def hook_fn(module, args, output):
            hidden_states = args[0]  # num_tokens x hidden_dim
            top_k_index = args[1]    # num_tokens x top_k
            top_k_weights = args[2]  # num_tokens x top_k

            num_experts = module.num_experts
            num_tokens, top_k = top_k_index.shape
            hidden_dim = hidden_states.shape[1]

            # Initialize accumulators on first call
            if layer_i not in self.stats:
                self.stats[layer_i] = {
                    "norm_prod_sums": torch.zeros(
                        num_experts, num_experts,
                        dtype=torch.float64, device=self.sdev,
                    ),
                    "coselect_counts": torch.zeros(
                        num_experts, num_experts,
                        dtype=torch.long, device=self.sdev,
                    ),
                    "gate_sums": torch.zeros(
                        num_experts, dtype=torch.float64, device=self.sdev,
                    ),
                    "norm_sums": torch.zeros(
                        num_experts, dtype=torch.float64, device=self.sdev,
                    ),
                    "norm_square_sums": torch.zeros(
                        num_experts, dtype=torch.float64, device=self.sdev,
                    ),
                    "gate_norm_sums": torch.zeros(
                        num_experts, dtype=torch.float64, device=self.sdev,
                    ),
                    "gate_norm_square_sums": torch.zeros(
                        num_experts, dtype=torch.float64, device=self.sdev,
                    ),
                    "total_tokens": 0,
                }

            self.stats[layer_i]["total_tokens"] += num_tokens

            # Re-run active experts to get per-expert outputs
            expert_out = torch.zeros(
                num_tokens, top_k, hidden_dim,
                device=hidden_states.device, dtype=hidden_states.dtype,
            )
            expert_mask = torch.nn.functional.one_hot(
                top_k_index, num_classes=num_experts
            ).permute(2, 1, 0)  # num_experts x top_k x num_tokens

            expert_hit = torch.sum(expert_mask, dim=(1, 2)).nonzero()[:, 0]
            for expert_i in expert_hit:
                top_k_pos, token_inds = torch.where(expert_mask[expert_i])
                in_hidden = hidden_states[token_inds]

                gate, up = torch.nn.functional.linear(
                    in_hidden, module.gate_up_proj[expert_i]
                ).chunk(2, dim=-1)
                out_hidden = torch.nn.functional.linear(
                    module.act_fn(gate) * up, module.down_proj[expert_i]
                )
                expert_out[token_inds, top_k_pos] = out_hidden

            # Compute gate probs and norms
            gate_probs = top_k_weights.float()
            norms = expert_out.float().norm(dim=-1)
            stats = self.stats[layer_i]

            # First-order statistics
            for sel_i in range(top_k):
                expert_inds = top_k_index[:, sel_i].to(self.sdev)
                sel_gates = gate_probs[:, sel_i]
                sel_norms = norms[:, sel_i]

                stats["gate_sums"].scatter_add_(
                    0, expert_inds, sel_gates.to(torch.float64).to(self.sdev)
                )
                stats["norm_sums"].scatter_add_(
                    0, expert_inds, sel_norms.to(torch.float64).to(self.sdev)
                )
                stats["norm_square_sums"].scatter_add_(
                    0, expert_inds,
                    (sel_norms ** 2).to(torch.float64).to(self.sdev),
                )
                stats["gate_norm_sums"].scatter_add_(
                    0, expert_inds,
                    (sel_gates * sel_norms).to(torch.float64).to(self.sdev),
                )
                stats["gate_norm_square_sums"].scatter_add_(
                    0, expert_inds,
                    ((sel_gates ** 2) * (sel_norms ** 2))
                    .to(torch.float64).to(self.sdev),
                )

            # Pairwise statistics
            for sel_i in range(top_k):
                for sel_j in range(sel_i, top_k):
                    sel_i_expert_inds = top_k_index[:, sel_i]
                    sel_j_expert_inds = top_k_index[:, sel_j]

                    gate_prob_prod = gate_probs[:, sel_i] * gate_probs[:, sel_j]
                    norm_prods = norms[:, sel_i] * norms[:, sel_j]
                    weighted_norm_prods = gate_prob_prod * norm_prods

                    idx = (
                        (sel_i_expert_inds * num_experts) + sel_j_expert_inds
                    ).to(self.sdev)

                    stats["norm_prod_sums"].view(-1).scatter_add_(
                        0, idx,
                        weighted_norm_prods.to(torch.float64).to(self.sdev),
                    )
                    stats["coselect_counts"].view(-1).scatter_add_(
                        0, idx,
                        torch.ones(num_tokens, dtype=torch.long,
                                   device=self.sdev),
                    )

                    if sel_i != sel_j:
                        idx_t = (
                            (sel_j_expert_inds * num_experts)
                            + sel_i_expert_inds
                        ).to(self.sdev)
                        stats["norm_prod_sums"].view(-1).scatter_add_(
                            0, idx_t,
                            weighted_norm_prods.to(torch.float64).to(
                                self.sdev
                            ),
                        )
                        stats["coselect_counts"].view(-1).scatter_add_(
                            0, idx_t,
                            torch.ones(num_tokens, dtype=torch.long,
                                       device=self.sdev),
                        )

        return hook_fn

    def reset(self):
        """Clears accumulated statistics for a fresh calibration run."""
        self.stats.clear()

    def get_stats(self):
        """Returns collected stats as a dict of layer_i -> dict of numpy arrays."""
        return {
            layer_i: {
                key: val.cpu().numpy() if isinstance(val, torch.Tensor) else val
                for key, val in layer_stats.items()
            }
            for layer_i, layer_stats in self.stats.items()
        }

    def close(self):
        """Removes hooks and frees memory."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.stats.clear()


def calibrate(
    model_path,
    prompts,
    out_path,
    limit=None,
    seed=20260423,
    stats_on_cpu=False,
    max_prompt_length=None,
):
    """
    Run calibration on a MoE model to collect the F-matrix.

    Args:
        model_path: Path to a HuggingFace MoE model directory.
        prompts: A list of strings (will be tokenized) or a list of lists of
            ints (pre-tokenized prompt token IDs).
        out_path: Path to save the output HDF5 file.
        limit: If set, subsample this many prompts (or this fraction if < 1).
        seed: Random seed for subsampling.
        stats_on_cpu: If True, keep accumulators on CPU.
        max_prompt_length: Truncate prompts longer than this (in tokens).
    """
    print("Loading tokenizer and model from %s ..." % model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Tokenize if needed
    if prompts and isinstance(prompts[0], str):
        print("Tokenizing %d prompts..." % len(prompts))
        input_ids_list = []
        for p in prompts:
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            else:
                ids = tokenizer.encode(p)
            input_ids_list.append(ids)
    else:
        input_ids_list = list(prompts)

    # Subsample
    rng = np.random.default_rng(seed)
    if limit is not None:
        num = int(limit * len(input_ids_list)) if limit < 1 else int(limit)
        num = min(num, len(input_ids_list))
        indices = rng.choice(len(input_ids_list), num, replace=False)
        input_ids_list = [input_ids_list[i] for i in indices]

    print("Calibrating on %d prompts..." % len(input_ids_list))
    observer = HOExpertObserver(model, stats_on_gpu=(not stats_on_cpu))

    with torch.no_grad():
        for input_ids in tqdm.tqdm(input_ids_list, desc="Calibrating"):
            if max_prompt_length and len(input_ids) > max_prompt_length:
                input_ids = input_ids[:max_prompt_length]
            ids_tensor = torch.tensor([input_ids]).to(model.device)
            model(input_ids=ids_tensor)

    stats = observer.get_stats()
    observer.close()

    # Save to HDF5
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print("Saving observations to %s ..." % out_path)
    with h5py.File(out_path, "w") as f:
        task_group = f.create_group("default")
        for layer_i, layer_stats in stats.items():
            layer_group = task_group.create_group("layer_%d" % layer_i)
            for key, val in layer_stats.items():
                layer_group.create_dataset(key, data=val)

    print("Done. Collected stats for %d layers." % len(stats))
