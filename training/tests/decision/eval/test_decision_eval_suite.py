"""
Tests for evaluation suites (hmf/eval/decision/suite.py) and the decision-eval and decision-serve commands,
with conftest.py's tiny model:

    - reading a suite file, with overrides, required keys, a file with nothing to run, and data by dataset name or
      path;
    - partial suites run only their sections, such as only benchmarks or only calibration;
    - a full suite with JevBench, against a stand-in for JevBench's harness (same command line, asks the
      suite's server), which the suite starts and stops;
    - the JevBench section is checked before anything runs, including the checkout, its commit, the output
      folder, the port, and the reference runs;
    - the commands, through HMF's launcher.
"""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from hmf.eval.decision.predictor import CALIBRATION_FILE
from hmf.eval.decision.suite import load_suite, resolve_data, run_suite


def _suite(tmp_path, trained, **sections) -> str:
    suite = {"model_name_or_path": trained["model_dir"], "adapter_name_or_path": trained["adapter_dir"],
             "output_dir": str(tmp_path / "reports"), **sections}
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(suite))
    return str(path)


def _no_saved_calibration(trained) -> None:
    path = Path(trained["adapter_dir"]) / CALIBRATION_FILE
    if path.exists():
        path.unlink()


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


# Reading a suite


def test_load_suite(tmp_path, trained):
    path = _suite(tmp_path, trained, benchmarks=[{"name": "toy", "data": trained["data"]}], max_len=4096)
    assert load_suite(path, ["max_len=512", "output_dir=elsewhere"])["max_len"] == 512

    with pytest.raises(ValueError, match="set model_name_or_path"):
        load_suite(path, ["model_name_or_path=null"])
    with pytest.raises(ValueError, match="nothing to run"):
        load_suite(_suite(tmp_path, trained))


def test_resolve_data(tmp_path):
    (tmp_path / "dataset_info.json").write_text(json.dumps({"toy": {"file_name": "folder/toy.jsonl"}}))
    assert resolve_data({"data": "a/b.jsonl"}) == "a/b.jsonl"
    assert resolve_data({"dataset": "toy"}, str(tmp_path)) == os.path.join(str(tmp_path), "folder/toy.jsonl")
    for entry in ({"data": "x", "dataset": "toy"}, {"name": "nothing"}):
        with pytest.raises(ValueError, match="either dataset"):
            resolve_data(entry, str(tmp_path))


# Partial suites


def test_only_benchmarks(tmp_path, trained):
    _no_saved_calibration(trained)
    path = _suite(tmp_path, trained, benchmarks=[{"name": "toy", "data": trained["data"], "split": "all", "limit": 0}])

    results = run_suite(path)

    reports = tmp_path / "reports"
    assert sorted(os.listdir(reports)) == ["summary.md", "toy.json"]  # no calibration, no JevBench
    assert results["calibration_used"] == "none (temperature 1)"
    assert results["benchmarks"]["toy"]["overall"]["n"] == 91 and results["benchmarks"]["toy"]["num_skipped"] == 0
    summary = (reports / "summary.md").read_text()
    assert "Calibration: none (temperature 1)" in summary and "| toy | 91 |" in summary


def test_only_calibration(tmp_path, trained):
    path = _suite(tmp_path, trained, calibration={"data": trained["data"], "split": "all", "limit": 0})

    results = run_suite(path)

    reports = tmp_path / "reports"
    assert sorted(os.listdir(reports)) == ["calibration.json", "summary.md"]
    saved = Path(trained["adapter_dir"]) / CALIBRATION_FILE
    assert json.loads((reports / "calibration.json").read_text()) == json.loads(saved.read_text())
    assert results["calibration"]["num_rows"] == 91
    assert "Calibration on 91 rows" in (reports / "summary.md").read_text()


# JevBench


# A stand-in for JevBench's harness, with its command line. `run` asks the server each task (choice
# questions) and records whether the answer is the task's, and `summarize` counts the correct answers
FAKE_CLI = """
import argparse, json, urllib.request

def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    run = commands.add_parser("run")
    for flag in ("--tasks", "--adapter", "--endpoint", "--key-env", "--model", "--run-label", "--cost-basis",
                 "--reserve-usd", "--results", "--raw-dir", "--ledger"):
        run.add_argument(flag)
    summarize = commands.add_parser("summarize")
    for flag in ("--tasks", "--results", "--ledger", "--public-export"):
        summarize.add_argument(flag)
    args = parser.parse_args()
    if args.command == "run":
        with open(args.results, "w") as out:
            for line in open(args.tasks):
                task = json.loads(line)
                body = json.dumps({"state": task["state"], "questions": {"decision": task["question"]}}).encode()
                request = urllib.request.Request(args.endpoint + "/v1/systemone", data=body,
                                                 headers={"Content-Type": "application/json"})
                answer = json.load(urllib.request.urlopen(request))["answers"]["decision"]
                correct = answer["choice"] == task["answer"]
                out.write(json.dumps({"task_id": task["id"], "ok": True, "correct": correct}) + "\\n")
        open(args.ledger, "w").close()
    else:
        records = [json.loads(line) for line in open(args.results)]
        json.dump({"n_correct": sum(r["correct"] for r in records), "n_attempted": len(records)},
                  open(args.public_export, "w"))

if __name__ == "__main__":
    main()
"""


@pytest.fixture
def fake_jevbench(tmp_path) -> dict:
    """A JevBench checkout (a git repository) with 6 public tasks and the stand-in harness."""
    root = tmp_path / "jevbench"
    (root / "datasets" / "public").mkdir(parents=True)
    (root / "jevbench").mkdir()
    (root / "jevbench" / "__init__.py").write_text("")
    (root / "jevbench" / "cli.py").write_text(FAKE_CLI)
    for part, offset in (("original", 0), ("easy", 2), ("hard", 4)):
        question = {"type": "choice", "instructions": "question ?", "criteria": {"refund": None, "ignore": None}}
        tasks = [{"id": f"t{offset + i}", "state": "state state", "answer": "refund", "question": question}
                 for i in range(2)]
        (root / "datasets" / "public" / f"{part}.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    git = ["git", "-C", str(root), "-c", "user.email=test@example.com", "-c", "user.name=test"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "tasks"], check=True)
    commit = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    reference = tmp_path / "reference.jsonl"  # a reference run that answers every task correctly
    reference.write_text("".join(json.dumps({"task_id": f"t{i}", "correct": True}) + "\n" for i in range(6)))
    return {"dir": str(root), "commit": commit, "reference": str(reference)}


def test_full_suite_with_jevbench(tmp_path, trained, fake_jevbench):
    port = _free_port()
    path = _suite(
        tmp_path, trained,
        calibration={"data": trained["data"], "split": "calib", "limit": 0},
        benchmarks=[{"name": "toy", "data": trained["data"], "split": "test", "limit": 0}],
        jevbench={"dir": fake_jevbench["dir"], "commit": fake_jevbench["commit"], "max_len": 200, "port": port,
                  "compare": [{"name": "reference", "results": fake_jevbench["reference"]}]},
    )

    results = run_suite(path)

    reports = tmp_path / "reports"
    assert {"calibration.json", "toy.json", "summary.md", "jevbench"} <= set(os.listdir(reports))
    assert {"tasks.jsonl", "results.jsonl", "summary.json", "compare_reference.json"} <= set(
        os.listdir(reports / "jevbench")
    )
    assert results["jevbench"]["summary"]["n_attempted"] == 6 and results["jevbench"]["max_len"] == 200
    comparison = results["jevbench"]["comparisons"]["reference"]
    assert comparison["n"] == 6 and comparison["a_correct"] == 6
    summary = (reports / "summary.md").read_text()
    assert "JevBench (window 200)" in summary and "against reference" in summary
    # The server is stopped, so nothing accepts connections on its port anymore
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", port)) != 0


@pytest.mark.parametrize("problem", ["no checkout", "other commit", "output inside", "port taken", "no reference"])
def test_jevbench_section_is_checked_first(tmp_path, trained, fake_jevbench, problem):
    jevbench = {"dir": fake_jevbench["dir"], "commit": fake_jevbench["commit"], "port": _free_port(),
                "compare": [{"name": "reference", "results": fake_jevbench["reference"]}]}
    overrides, message, holder = [], None, None
    if problem == "no checkout":
        jevbench["dir"], message = str(tmp_path / "missing"), "must be a JevBench checkout"
    elif problem == "other commit":
        jevbench["commit"], message = "0" * 40, "Check out that commit"
    elif problem == "output inside":
        overrides, message = [f"output_dir={fake_jevbench['dir']}/reports"], "outside the JevBench checkout"
    elif problem == "port taken":
        holder = socket.socket()
        holder.bind(("127.0.0.1", jevbench["port"]))
        holder.listen()
        message = "already listens"
    else:
        jevbench["compare"][0]["results"], message = str(tmp_path / "missing.jsonl"), "no results file"
    path = _suite(tmp_path, trained, jevbench=jevbench)
    try:
        with pytest.raises(ValueError, match=message):
            run_suite(path, *overrides)
    finally:
        if holder is not None:
            holder.close()
    assert not (tmp_path / "reports").exists()  # nothing ran


# The commands


def _launch(*argv: str) -> subprocess.CompletedProcess:
    code = f"import sys; sys.argv = {['hmf', *argv]!r}; from hmf.launcher import launch; launch()"
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_decision_eval_command(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps({"task_id": "t0", "correct": True}) + "\n")
    b.write_text(json.dumps({"task_id": "t0", "correct": False}) + "\n")
    result = _launch("decision-eval", "compare", str(a), str(b))
    assert result.returncode == 0, result.stderr
    assert "exact McNemar p = 1.0000" in result.stdout


def test_decision_serve_command_refuses_other_hosts(trained):
    result = _launch("decision-serve", trained["model_dir"], "--host", "0.0.0.0")
    assert result.returncode != 0 and "Refusing to listen on 0.0.0.0" in result.stderr
