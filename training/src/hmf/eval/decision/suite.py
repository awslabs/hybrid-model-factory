"""
Evaluation suites, which run calibration, a list of benchmarks, and JevBench from one YAML file with one
command, writing every result to one folder.

    hmf decision-eval run examples/decision/eval/strands_v19.yaml model_name_or_path=... \\
        adapter_name_or_path=... output_dir=reports/my-model

Values on the command line (key=value, e.g. jevbench.max_len=3072) override the file. A suite file has:

    model_name_or_path: the decision model (required)
    adapter_name_or_path: its LoRA adapter (the training output_dir), if any
    output_dir: where the reports go (required)
    max_len, prompt_format, force_bos, ordinal_smoothing, seed, dataset_dir: as for calibrate and evaluate
    calibration: {dataset or data, split, limit}: fitted first, and used by everything after
    benchmarks: [{name, dataset or data, split, limit}, ...]: each evaluated in turn
    jevbench: {dir, commit, max_len, port, compare: [{name, results, run}, ...]}: JevBench on a server the
        run starts and stops

Every section is optional, and the run does only what the file has (at least one of calibration,
benchmarks and jevbench). A data file is a dataset name of data/dataset_info.json (dataset) or a path
(data). Without a calibration section, everything uses the model's saved calibration if it has one,
else temperature 1.

Into output_dir, the run writes calibration.json (a copy of the calibration saved in the adapter's
folder), <name>.json for each benchmark (as evaluate writes it), jevbench/ (JevBench's results and
summary, and compare_<name>.json for each comparison), and summary.md (a table of every result), and it
prints the table.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from ...extras import logging
from .calibrate import calibrate
from .evaluate import evaluate
from .jevbench import compare, load_results
from .predictor import DecisionPredictor, calibration_path, load_calibration


logger = logging.get_logger(__name__)


SECTIONS = ("calibration", "benchmarks", "jevbench")
JEVBENCH_PARTS = ("original", "easy", "hard")  # the public tasks, in the order JevBench lists them


def load_suite(path: str, overrides: Sequence[str] = ()) -> Dict[str, Any]:
    """A suite file, with key=value overrides, checked."""
    from omegaconf import OmegaConf

    suite = OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(list(overrides)))
    suite = OmegaConf.to_container(suite, resolve=True)
    for key in ("model_name_or_path", "output_dir"):
        if not suite.get(key):
            raise ValueError(f"{path}: set {key} (in the file, or {key}=... on the command line).")
    if not any(suite.get(section) for section in SECTIONS):
        raise ValueError(f"{path}: nothing to run: add a calibration, benchmarks or jevbench section.")
    return suite


def resolve_data(entry: Dict[str, Any], dataset_dir: str = "data") -> str:
    """The data file of a section, given by its dataset (a name in dataset_info.json) or its data (a path)."""
    if ("dataset" in entry) == ("data" in entry):
        raise ValueError(f"Give either dataset (a name in {dataset_dir}/dataset_info.json) or data (a path): {entry}")
    if "data" in entry:
        return entry["data"]
    from ...data.parser import get_dataset_list

    [attr] = get_dataset_list([entry["dataset"]], dataset_dir)
    return os.path.join(dataset_dir, attr.dataset_name)


def _port_is_free(port: int) -> bool:
    """Whether nothing accepts connections on 127.0.0.1:port (a port a server just closed is free)."""
    with socket.socket() as probe:
        probe.settimeout(2)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def _check_jevbench(jevbench: Dict[str, Any], output_dir: str) -> None:
    """Checks the JevBench section's checkout, commit, output folder, and port before anything runs."""
    directory = jevbench.get("dir")
    if not directory or not os.path.isdir(os.path.join(directory, "datasets", "public")):
        raise ValueError(f"jevbench.dir must be a JevBench checkout (with datasets/public/), got {directory!r}.")
    commit = jevbench.get("commit")
    if commit:
        head = subprocess.run(
            ["git", "-C", directory, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        if head != commit:
            raise ValueError(
                f"The JevBench checkout {directory} is at {head}, not at jevbench.commit {commit}. Check out that "
                f"commit (git -C {directory} checkout {commit}), or change jevbench.commit."
            )
    if os.path.abspath(output_dir).startswith(os.path.abspath(directory) + os.sep):
        raise ValueError("output_dir must be outside the JevBench checkout (JevBench refuses to write into it).")
    for reference in jevbench.get("compare") or []:
        if not os.path.isfile(str(reference.get("results"))):
            raise ValueError(f"jevbench.compare {reference.get('name')!r}: no results file at {reference.get('results')!r}.")
    port = int(jevbench.get("port", 8099))
    if not _port_is_free(port):
        raise ValueError(f"Something already listens on 127.0.0.1:{port}: stop it, or set jevbench.port.")


def run_jevbench(jevbench: Dict[str, Any], predictor: DecisionPredictor, output_dir: str) -> Dict[str, Any]:
    """
    JevBench's public tasks against the model. It starts a server on 127.0.0.1, runs JevBench's harness from
    its checkout (nothing installed), and stops the server, also if the harness fails.

    Returns:
        JevBench's summary, and the comparison with each reference run.
    """
    import uvicorn

    from .server import DecisionServer, create_app

    directory, port = jevbench["dir"], int(jevbench.get("port", 8099))
    folder = os.path.join(output_dir, "jevbench")
    os.makedirs(folder, exist_ok=True)
    tasks = os.path.join(folder, "tasks.jsonl")
    with open(tasks, "w", encoding="utf-8") as out:
        for part in JEVBENCH_PARTS:
            with open(os.path.join(directory, "datasets", "public", f"{part}.jsonl"), encoding="utf-8") as f:
                out.write(f.read())

    server = DecisionServer(predictor, max_len=jevbench.get("max_len"))
    config = uvicorn.Config(create_app(server), host="127.0.0.1", port=port, log_level="warning")
    http = uvicorn.Server(config)
    thread = threading.Thread(target=http.run, daemon=True)
    thread.start()
    env = {**os.environ, "PYTHONPATH": directory}
    results, ledger, summary = (
        os.path.join(folder, name) for name in ("results.jsonl", "ledger.jsonl", "summary.json")
    )
    try:
        for _ in range(600):
            if http.started or not thread.is_alive():
                break
            time.sleep(0.1)
        if not http.started:
            raise RuntimeError(f"The server did not start on 127.0.0.1:{port}.")
        for name in (results, ledger):  # JevBench refuses to append to an earlier run's files
            if os.path.exists(name):
                os.remove(name)
        shutil.rmtree(os.path.join(folder, "raw"), ignore_errors=True)
        subprocess.run(
            [sys.executable, "-m", "jevbench.cli", "run", "--tasks", tasks, "--adapter", "typesafe",
             "--endpoint", f"http://127.0.0.1:{port}", "--key-env", "", "--model", server.model_name,
             "--run-label", server.model_name, "--cost-basis", "no_billable_account_public_endpoint",
             "--reserve-usd", "0", "--results", results, "--raw-dir", os.path.join(folder, "raw"),
             "--ledger", ledger],
            env=env, check=True,
        )
    finally:
        http.should_exit = True
        thread.join(timeout=30)
    subprocess.run(
        [sys.executable, "-m", "jevbench.cli", "summarize", "--tasks", tasks, "--results", results,
         "--ledger", ledger, "--public-export", summary],
        env=env, check=True,
    )
    with open(summary, encoding="utf-8") as f:
        report: Dict[str, Any] = {"summary": json.load(f), "max_len": server.max_len, "comparisons": {}}

    ours = load_results(results)
    for reference in jevbench.get("compare") or []:
        result = compare(load_results(reference["results"], reference.get("run")), ours)
        with open(os.path.join(folder, f"compare_{reference['name']}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        report["comparisons"][reference["name"]] = result
    return report


def _summary_table(results: Dict[str, Any]) -> str:
    """The results as a Markdown table."""
    lines = []
    calibration = results.get("calibration")
    if calibration:
        by_kind = ", ".join(f"{k} {v:.3f}" for k, v in calibration["temperature_by_kind"].items())
        lines += [
            f"Calibration on {calibration['num_rows']:,} rows: overall temperature {calibration['temperature']:.3f}; "
            f"by kind {by_kind or 'none'}; ECE {calibration['before']['ece']:.3f} before, "
            f"{calibration['after']['ece']:.3f} after.",
            "",
        ]
    else:
        lines += [f"Calibration: {results['calibration_used']}.", ""]
    if results.get("benchmarks"):
        lines += [
            "| benchmark | rows | skipped | accuracy | ECE | NLL | by task |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, report in results["benchmarks"].items():
            o = report["overall"]
            tasks = ", ".join(f"{t} {b['accuracy']:.3f}" for t, b in report["by_task"].items())
            lines.append(
                f"| {name} | {o['n']:,} | {report['num_skipped']} | {o['accuracy']:.3f} | {o['ece']:.3f} | "
                f"{o['nll']:.3f} | {tasks} |"
            )
        lines.append("")
    jevbench = results.get("jevbench")
    if jevbench:
        s = jevbench["summary"]
        lines.append(
            f"JevBench (window {jevbench['max_len']}): {s.get('n_correct')} of {s.get('n_attempted')} tasks correct."
        )
        for name, c in jevbench["comparisons"].items():
            net = c["b_correct"] - c["a_correct"]
            lines.append(
                f"- against {name}: {c['a_correct']} for {name}, {c['b_correct']} here (net {net:+d}); "
                f"{len(c['a_only'])} tasks only {name} answers, {len(c['b_only'])} only this model; "
                f"McNemar p = {c['p']:.4f}"
            )
    return "\n".join(lines)


def run_suite(path: str, *overrides: str) -> Dict[str, Any]:
    """
    Runs an evaluation suite (see the module's docstring).

    Args:
        path: The suite file.
        overrides: key=value overrides of the file.

    Returns:
        Every result, which is the calibration, each benchmark's report, and JevBench's.
    """
    suite = load_suite(path, overrides)
    model, adapter, output_dir = suite["model_name_or_path"], suite.get("adapter_name_or_path"), suite["output_dir"]
    dataset_dir = suite.get("dataset_dir", "data")
    options = {
        "prompt_format": suite.get("prompt_format", "strands"),
        "force_bos": suite.get("force_bos", False),
        "max_len": suite.get("max_len", 4096),
    }
    jevbench = suite.get("jevbench")
    # Every check before the first model is loaded
    calibration_data = resolve_data(suite["calibration"], dataset_dir) if suite.get("calibration") else None
    benchmarks = {b["name"]: (b, resolve_data(b, dataset_dir)) for b in suite.get("benchmarks") or []}
    if jevbench:
        _check_jevbench(jevbench, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    results: Dict[str, Any] = {}
    seed = suite.get("seed", 0)
    # The model is loaded once, for calibration, the benchmarks, and JevBench
    predictor = DecisionPredictor(model, adapter, calibration=None, **options)
    saved = calibration_path(model, adapter)
    if calibration_data is not None:
        section = suite["calibration"]
        results["calibration"] = calibrate(
            model, calibration_data, adapter, split=section.get("split", "calib"), limit=section.get("limit", 4000),
            seed=seed, ordinal_smoothing=suite.get("ordinal_smoothing", 0.1), predictor=predictor,
        )
        shutil.copy(saved, os.path.join(output_dir, "calibration.json"))
    # From here on, the model answers at its calibration, either the one just fitted or the one it had
    predictor.calibration = load_calibration(saved)
    results["calibration_used"] = saved if predictor.calibration is not None else "none (temperature 1)"
    results["benchmarks"] = {}
    for name, (section, data) in benchmarks.items():
        results["benchmarks"][name] = evaluate(
            model, data, adapter, split=section.get("split", "test"), limit=section.get("limit", 4000), seed=seed,
            out=os.path.join(output_dir, f"{name}.json"), predictor=predictor,
        )
    if jevbench:
        results["jevbench"] = run_jevbench(jevbench, predictor, output_dir)

    table = _summary_table(results)
    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(f"# {os.path.basename(adapter or model)}\n\n{table}\n")
    print(f"\n{table}\n\nWritten to {output_dir}")
    return results
