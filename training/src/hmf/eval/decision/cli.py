"""
The command lines of decision model evaluation:

    hmf decision-eval calibrate MODEL --adapter_name_or_path ADAPTER --data FILE      (see calibrate.py)
    hmf decision-eval evaluate MODEL --adapter_name_or_path ADAPTER --data FILE       (see evaluate.py)
    hmf decision-eval compare RUN_A RUN_B                                             (see jevbench.py)
    hmf decision-eval run SUITE.yaml key=value ...                                    (see suite.py)
    hmf decision-serve MODEL --adapter_name_or_path ADAPTER                           (see server.py)

Each takes --help for its options.
"""


def eval_main() -> None:
    import fire

    from .calibrate import calibrate
    from .evaluate import evaluate
    from .jevbench import main as compare
    from .suite import run_suite

    fire.Fire(
        {"calibrate": calibrate, "evaluate": evaluate, "compare": compare, "run": run_suite},
        name="hmf decision-eval",
    )


def serve_main() -> None:
    import fire

    from .server import serve

    fire.Fire(serve, name="hmf decision-serve")
