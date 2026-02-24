# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
from copy import deepcopy


USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   hybridfactory prime-init -h: convert Transformer to Hybrid       |\n"
    + "|   hybridfactory prime-unfuse -h: convert fused to standard Hybrid  |\n"
    + "|   hybridfactory train -h: train models                             |\n"
    + "|   hybridfactory train-multinode -h: multi-node distributed train   |\n"
    + "|   hybridfactory export -h: merge LoRA adapters and export model    |\n"
    + "|   hybridfactory api -h: launch an OpenAI-style API server          |\n"
    + "|   hybridfactory chat -h: launch a chat interface in CLI            |\n"
    + "|   hybridfactory preprocess -h: sharded preprocess and tokenize     |\n"
    + "|   hybridfactory env: show environment info                         |\n"
    + "|   hybridfactory version: show version info                         |\n"
    + "| Hint: You can use `hmf` as a shortcut for `hybridfactory`.         |\n"
    + "-" * 70
)


def launch():
    from .extras import logging
    from .extras.env import VERSION, print_env
    from .extras.misc import find_available_port, get_device_count, is_env_enabled, use_kt, use_ray

    logger = logging.get_logger(__name__)
    WELCOME = (
        "-" * 58
        + "\n"
        + f"| Welcome to Hybrid Model Factory, version {VERSION}"
        + " " * (21 - len(VERSION))
        + "|\n|"
        + " " * 56
        + "|\n"
        + "| Project page: https://github.com/awslabs/hybrid-model-factory |\n"
        + "-" * 58
    )

    command = sys.argv.pop(1) if len(sys.argv) > 1 else "help"
    if is_env_enabled("USE_MCA"):  # force use torchrun
        os.environ["FORCE_TORCHRUN"] = "1"

    if command == "train" and (
        is_env_enabled("FORCE_TORCHRUN") or (get_device_count() > 1 and not use_ray() and not use_kt())
    ):
        # launch distributed training
        nnodes = os.getenv("NNODES", "1")
        node_rank = os.getenv("NODE_RANK", "0")
        nproc_per_node = os.getenv("NPROC_PER_NODE", str(get_device_count()))
        master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
        master_port = os.getenv("MASTER_PORT", str(find_available_port()))
        logger.info_rank0(f"Initializing {nproc_per_node} distributed tasks at: {master_addr}:{master_port}")
        if int(nnodes) > 1:
            logger.info_rank0(f"Multi-node training enabled: num nodes: {nnodes}, node rank: {node_rank}")

        # elastic launch support
        max_restarts = os.getenv("MAX_RESTARTS", "0")
        rdzv_id = os.getenv("RDZV_ID")
        min_nnodes = os.getenv("MIN_NNODES")
        max_nnodes = os.getenv("MAX_NNODES")

        env = deepcopy(os.environ)
        if is_env_enabled("OPTIM_TORCH", "1"):
            # optimize DDP, see https://zhuanlan.zhihu.com/p/671834539
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        if rdzv_id is not None:
            # launch elastic job with fault tolerant support when possible
            # see also https://docs.pytorch.org/docs/stable/elastic/train_script.html
            rdzv_nnodes = nnodes
            # elastic number of nodes if MIN_NNODES and MAX_NNODES are set
            if min_nnodes is not None and max_nnodes is not None:
                rdzv_nnodes = f"{min_nnodes}:{max_nnodes}"

            process = subprocess.run(
                (
                    "torchrun --nnodes {rdzv_nnodes} --nproc-per-node {nproc_per_node} "
                    "--rdzv-id {rdzv_id} --rdzv-backend c10d --rdzv-endpoint {master_addr}:{master_port} "
                    "--max-restarts {max_restarts} {file_name} {args}"
                )
                .format(
                    rdzv_nnodes=rdzv_nnodes,
                    nproc_per_node=nproc_per_node,
                    rdzv_id=rdzv_id,
                    master_addr=master_addr,
                    master_port=master_port,
                    max_restarts=max_restarts,
                    file_name=__file__,
                    args=" ".join(sys.argv[1:]),
                )
                .split(),
                env=env,
                check=True,
            )
        else:
            # NOTE: DO NOT USE shell=True to avoid security risk
            process = subprocess.run(
                (
                    "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                    "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
                )
                .format(
                    nnodes=nnodes,
                    node_rank=node_rank,
                    nproc_per_node=nproc_per_node,
                    master_addr=master_addr,
                    master_port=master_port,
                    file_name=__file__,
                    args=" ".join(sys.argv[1:]),
                )
                .split(),
                env=env,
                check=True,
            )

        sys.exit(process.returncode)

    elif command == "api":
        from .api.app import run_api

        run_api()

    elif command == "prime-init":
        from .priming.hybridize_model import load_config, verify_hybrid_config, hybridize_model

        import argparse

        parser = argparse.ArgumentParser(description="Convert a Transformer model to a Hybrid model.")
        parser.add_argument("config", help="Path to the YAML config file for hybridization")
        args = parser.parse_args()

        hybrid_config = load_config(args.config)
        verify_hybrid_config(hybrid_config)
        hybridize_model(hybrid_config)

    elif command == "prime-unfuse":
        from .priming.fused_to_standard import unfuse_model

        import argparse

        parser = argparse.ArgumentParser(description="Convert a fused Hybrid model to standard Hybrid model.")
        parser.add_argument("model_dir", help="Directory of the fused model")
        parser.add_argument("--save_dir", type=str, default=None, help="Output directory (default: <model_dir>_unfused)")
        parser.add_argument("--save_max_shard_size", type=str, default="5GB", help="Max shard size for saving")
        args = parser.parse_args()

        save_dir = args.save_dir or args.model_dir.rstrip("/") + "_unfused"
        unfuse_model(args.model_dir, save_dir, args.save_max_shard_size)

    elif command == "chat":
        from .chat.chat_model import run_chat

        run_chat()

    elif command == "eval":
        raise NotImplementedError("Evaluation will be deprecated in the future.")

    elif command == "export":
        from .train.tuner import export_model

        export_model()

    elif command == "train":
        from .train.tuner import run_exp

        run_exp()

    elif command == "train-multinode":
        import argparse

        parser = argparse.ArgumentParser(
            description="Launch multi-node distributed training.",
            usage="hmf train-multinode config master_addr nnodes node_rank [--master-port PORT]"
        )
        parser.add_argument("config", help="Path to the YAML config file")
        parser.add_argument("master_addr", help="IP address of the master node")
        parser.add_argument("nnodes", type=int, help="Total number of nodes")
        parser.add_argument("node_rank", type=int, help="Rank of this node (0 to nnodes-1)")
        parser.add_argument("--master-port", type=int, default=29500, help="Port for communication (default: 29500)")
        args = parser.parse_args()

        # Locate the multinode launcher script
        script_path = os.path.join(os.path.dirname(__file__), "scripts", "multinode_launcher.sh")
        if not os.path.exists(script_path):
            print(f"Error: multinode_launcher.sh not found at {script_path}")
            sys.exit(1)

        # Call the bash script
        result = subprocess.run([
            "bash", script_path,
            args.config,
            args.master_addr,
            str(args.nnodes),
            str(args.node_rank),
            str(args.master_port),
        ])
        sys.exit(result.returncode)

    elif command == "preprocess":
        from .data.sharded_preprocess_and_tokenize import run_preprocess

        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("config", help="YAML config file")
        parser.add_argument("--shard_size", type=int, default=75000)
        parser.add_argument("--parallel_jobs", type=int, default=128)
        parser.add_argument("--no-cleanup", dest="cleanup", action="store_false")
        args = parser.parse_args()
        sys.exit(run_preprocess(args.config, args.shard_size, args.parallel_jobs, args.cleanup))

    elif command == "env":
        print_env()

    elif command == "version":
        print(WELCOME)

    elif command == "help":
        print(USAGE)

    else:
        print(f"Unknown command: {command}.\n{USAGE}")


if __name__ == "__main__":
    from hmf.train.tuner import run_exp  # use absolute import

    run_exp()
