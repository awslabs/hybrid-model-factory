"""
HOPE QP solver.

Builds the expert interaction matrix F from calibration observations and
solves a binary quadratic program per layer to find the optimal pruning set.
"""

import os
import json
import numpy as np
import scipy.optimize
import h5py


def build_f_matrix(obs_hdf5, task_id, layer_i):
    """
    Build the F-matrix for a given task and layer.

    Args:
        obs_hdf5: Open h5py File handle.
        task_id: Task group name in the HDF5.
        layer_i: Layer index (int).

    Returns:
        An E x E numpy array (the F-matrix for this layer).
    """
    layer_group = obs_hdf5[task_id]["layer_%d" % layer_i]
    unnorm_matrix = layer_group["norm_prod_sums"][:]

    counts = layer_group["coselect_counts"][:].astype(np.float64)
    matrix = np.where(counts > 0, unnorm_matrix / counts, 0)

    assert np.all(np.isfinite(matrix))
    return matrix


def solve_qp(matrix, num_ones):
    """
    Solve the binary quadratic program: minimize b^T M b such that b has
    ``num_ones`` entries equal to 1. Performs a continuous relaxation solved
    by SLSQP, then rounds to binary.

    Args:
        matrix: An E x E numpy array M to solve the QP over.
        num_ones: The number of 1s in the desired binary solution.

    Returns:
        Tuple of (b_binary, b_continuous, obj_binary, obj_continuous).
    """
    n = matrix.shape[0]
    assert matrix.shape == (n, n)

    def objective(b):
        return b @ matrix @ b

    def gradient(b):
        return 2 * matrix @ b

    constraints = {"type": "eq", "fun": lambda b: np.sum(b) - num_ones}
    bounds = [(0.0, 1.0)] * n
    b_init = np.full(n, num_ones / n)

    result = scipy.optimize.minimize(
        objective, b_init, jac=gradient, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    b_cont = result.x

    # Round: top num_ones entries become 1
    b_bin = np.zeros(n)
    b_bin[np.argsort(-b_cont)[:num_ones]] = 1

    obj_bin = objective(b_bin)
    obj_cont = objective(b_cont)
    return b_bin, b_cont, obj_bin, obj_cont


def solve(
    obs_path,
    prune_frac,
    out_path,
    task_id=None,
):
    """
    End-to-end HOPE solver: build F-matrices, solve per-layer QPs, save
    the pruning set.

    Args:
        obs_path: Path to HDF5 observations from calibration.
        prune_frac: Fraction of experts to prune per layer (0 < frac < 1),
            or an integer count to prune per layer.
        out_path: Path to save the output JSON pruning set.
        task_id: Task ID in the HDF5. If None, uses the first available.
    """
    with h5py.File(obs_path, "r") as f:
        available_tasks = list(f.keys())
        if task_id is None:
            task_id = available_tasks[0]
        elif task_id not in available_tasks:
            raise ValueError(
                "Task '%s' not found. Available: %s"
                % (task_id, available_tasks)
            )

        # Discover layers
        layer_keys = [
            k for k in f[task_id].keys()
             if k.startswith("layer_") and "-" not in k
        ]
        num_layers = len(layer_keys)
        num_experts = f[task_id]["layer_0"]["norm_prod_sums"].shape[0]

        num_prune = (
            int(prune_frac * num_experts) if prune_frac < 1
            else int(prune_frac)
        )
        assert 0 < num_prune < num_experts

        print("Pruning %d/%d experts per layer across %d layers" % (
            num_prune, num_experts, num_layers
        ))

        pruneset = {}
        log_data = {
            "obs_path": obs_path,
            "prune_frac": prune_frac,
            "task_id": task_id,
            "num_layers": num_layers,
            "num_experts": num_experts,
            "layers": {},
        }

        for layer_i in range(num_layers):
            matrix = build_f_matrix(f, task_id, layer_i)
            print("Solving QP for layer %d ..." % layer_i)
            b_bin, b_cont, obj_bin, obj_cont = solve_qp(matrix, num_prune)

            # b=1 entries are the experts to prune
            pruned = np.where(b_bin > 0.5)[0].tolist()
            pruneset[str(layer_i)] = pruned

            log_data["layers"][str(layer_i)] = {
                "num_experts": num_experts,
                "num_pruned": len(pruned),
                "pruned_experts": pruned,
                "bin_objective_val": float(obj_bin),
                "cont_objective_val": float(obj_cont),
                "cont_solution": b_cont.tolist(),
            }

            print(
                "  Layer %d: pruning %d/%d experts, "
                "obj_bin=%.6e, obj_cont=%.6e"
                % (layer_i, len(pruned), num_experts, obj_bin, obj_cont)
            )

    # Save pruning set
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as out_f:
        json.dump(pruneset, out_f, indent=2)
    print("Pruning set saved to %s" % out_path)

    # Save log
    log_path = out_path + ".log.json"
    with open(log_path, "w") as log_f:
        json.dump(log_data, log_f, indent=2)
    print("Log saved to %s" % log_path)
