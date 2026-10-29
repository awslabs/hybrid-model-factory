"""
Tests for the decision server (hmf/eval/decision/server.py) and the comparison of JevBench runs
(hmf/eval/decision/jevbench.py), with conftest.py's tiny model, on the CPU:

    - the protocol, with each answer's fields, and invalid requests refused with 422;
    - the answers equal DecisionPredictor's for the same questions, calibrated or not, and a
      request's questions are answered as if asked alone;
    - prompts longer than the window, whose state is cut and options kept, and which --strict_window refuses;
    - the server listens on 127.0.0.1 only;
    - the exact McNemar test.
"""

import json
import math
import socket
import threading
import time
from pathlib import Path

import pytest

from hmf.eval.decision.jevbench import compare, load_results, mcnemar_exact
from hmf.eval.decision.metrics import choice_confidence, score_confidence
from hmf.eval.decision.predictor import DecisionPredictor
from hmf.eval.decision.server import (
    DecisionRequest,
    DecisionServer,
    WindowError,
    create_app,
    is_loopback,
    question_row,
    serve,
)


QUESTIONS = {
    "yes_no": {"type": "noul", "instructions": "question ?", "criteria": {"true": "high"}},
    "pick": {"type": "choice", "instructions": "question ?",
             "criteria": {"refund": "refund", "ignore": None, "escalate": "escalate"}},
    "rate": {"type": "score", "instructions": "question ?", "criteria": ["low", "low", "high", "high"]},
}


def _server(trained, **kwargs) -> DecisionServer:
    calibration = kwargs.pop("calibration", None)
    max_len = kwargs.pop("max_len", 4096)
    predictor = DecisionPredictor(
        trained["model_dir"], trained["adapter_dir"], device="cpu", calibration=calibration, max_len=max_len
    )
    return DecisionServer(predictor, **kwargs)


def _request(questions=QUESTIONS, state="state state") -> DecisionRequest:
    return DecisionRequest(state=state, questions=questions)


# The protocol


def test_answers(trained):
    reply = _server(trained).answer(_request())

    assert set(reply) == {"model", "answers", "usage"} and set(reply["answers"]) == set(QUESTIONS)
    assert reply["usage"]["output_tokens"] == 3 and reply["usage"]["input_tokens"] > 0

    noul = reply["answers"]["yes_no"]
    assert noul["type"] == "noul" and 0 <= noul["noul"] <= 1 and set(noul) == {"type", "noul"}

    choice = reply["answers"]["pick"]
    probs = choice["probabilities"]
    assert list(probs) == ["refund", "ignore", "escalate"]  # the request's order
    assert sum(probs.values()) == pytest.approx(1, abs=1e-3)
    assert choice["choice"] == max(probs, key=probs.get)
    assert choice["confidence"] == pytest.approx(choice_confidence(list(probs.values())), abs=1e-3)

    score = reply["answers"]["rate"]
    levels = score["probabilities"]
    assert list(levels) == ["0", "1", "2", "3"]
    assert score["legend"] == {"0": "low", "1": "low", "2": "high", "3": "high"}
    assert score["score"] == pytest.approx(sum(int(k) * p for k, p in levels.items()), abs=1e-3)
    assert score["confidence"] == pytest.approx(score_confidence(list(levels.values()), 0.1), abs=1e-3)


@pytest.mark.parametrize(
    "body, message",
    [
        ({"state": "s", "questions": {}}, "questions"),
        ({"state": "s", "questions": {"q": {"type": "choice", "instructions": "q", "criteria": {"a": None}}}},
         "2 to 255"),
        ({"state": "s", "questions": {"q": {"type": "score", "instructions": "q", "criteria": ["l"] * 11}}},
         "2 to 10"),
        ({"state": "s", "questions": {"q": {"type": "noul", "instructions": "q", "criteria": {"maybe": "x"}}}},
         "noul criteria"),
        ({"state": "s", "questions": {"q": {"type": "rank", "instructions": "q"}}}, "type"),
        ({"state": "s", "questions": {"q": {"type": "noul", "instructions": "q"}}, "images": ["abc"]}, "images"),
    ],
)
def test_invalid_requests_are_refused(trained, body, message):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    client = TestClient(create_app(_server(trained)))
    response = client.post("/v1/systemone", json=body)
    assert response.status_code == 422 and message in response.text


def test_health(trained):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    health = TestClient(create_app(_server(trained))).get("/health").json()
    assert health["status"] == "ok" and health["model_name_or_path"] == trained["model_dir"]
    assert health["adapter_name_or_path"] == trained["adapter_dir"] and health["max_length"] == 4096
    assert health["calibrated"] is False
    assert health["temperature_by_kind"] == {"noul": 1.0, "choice": 1.0, "score": 1.0}


# Consistency


@pytest.mark.parametrize("calibrated", [False, True])
def test_answers_equal_the_predictors(trained, tmp_path, calibrated):
    calibration = None
    if calibrated:
        calibration = str(tmp_path / "calibration.json")
        Path(calibration).write_text(json.dumps({"temperature": 1.5, "temperature_by_kind": {"choice": 0.7}}))
    server = _server(trained, calibration=calibration)
    request = _request()

    reply = server.answer(request)

    for name, question in request.questions.items():
        [probs] = server.predictor.probabilities([question_row(request.state, question)])
        answer = reply["answers"][name]
        if question.type == "noul":
            assert answer["noul"] == pytest.approx(probs[1], abs=1e-4)  # options shown as [false, true]
        else:
            assert list(answer["probabilities"].values()) == pytest.approx(probs, abs=1e-4)


def test_questions_together_are_answered_as_alone(trained):
    server = _server(trained)
    together = server.answer(_request())["answers"]
    for name, question in QUESTIONS.items():
        alone = server.answer(_request({name: question}))["answers"][name]
        assert set(alone) == set(together[name])
        for key, value in alone.items():
            if isinstance(value, float):
                assert value == pytest.approx(together[name][key], abs=1e-4), (name, key)
            elif isinstance(value, dict) and key == "probabilities":
                assert list(value.values()) == pytest.approx(list(together[name][key].values()), abs=1e-4)
            else:
                assert value == together[name][key], (name, key)


# The window


def test_long_states_are_cut_to_fit(trained):
    server = _server(trained, max_len=80)
    long_state = "state " * 300
    rendered, states = zip(*(server._render(question_row(long_state, q)) for q in _request().questions.values()))

    encoded = server._encode(list(rendered), list(states))

    for e in encoded:
        assert len(e.input_ids) <= 80
        # The options and the answer marker survive, in order, after the state
        assert all(0 < i < e.answer_idx for i in e.opt_idx) and e.answer_idx == len(e.input_ids) - 1
    reply = server.answer(_request(state=long_state))  # answered
    assert set(reply["answers"]) == set(QUESTIONS)


def test_strict_window_refuses_long_prompts(trained):
    server = _server(trained, max_len=80, strict_window=True)
    with pytest.raises(WindowError, match="exceeds the context window of 80 tokens"):
        server.answer(_request(state="state " * 300))
    server.answer(_request())  # a short prompt is answered


def test_options_that_cannot_fit_are_refused(trained):
    server = _server(trained, max_len=8)  # too small for the options themselves
    with pytest.raises(WindowError, match="do not fit"):
        server.answer(_request({"pick": QUESTIONS["pick"]}, state="state " * 50))


# Listening on this machine only


@pytest.mark.parametrize(
    "host, loopback",
    [("127.0.0.1", True), ("localhost", True), ("::1", True), ("127.0.0.2", True),
     ("0.0.0.0", False), ("::", False), ("10.0.0.5", False), ("example.com", False)],
)
def test_is_loopback(host, loopback):
    assert is_loopback(host) is loopback


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.0.0.5"])
def test_other_hosts_are_refused(trained, host):
    # Refused before the model is loaded
    with pytest.raises(ValueError, match="Refusing to listen on .* ssh -N -L"):
        serve(trained["model_dir"], trained["adapter_dir"], host=host)


def test_the_server_listens_on_127_0_0_1(trained):
    import uvicorn

    with socket.socket() as probe:  # a free port
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = uvicorn.Config(create_app(_server(trained)), host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.1)
        assert server.started
        addresses = {sock.getsockname()[0] for s in server.servers for sock in s.sockets}
        assert addresses == {"127.0.0.1"}
        import urllib.request

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as response:
            assert json.load(response)["status"] == "ok"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


# Comparing JevBench runs


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 11) == pytest.approx(1.0)  # an even split
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 2**6)  # every discordant task one way
    assert mcnemar_exact(2, 10) == pytest.approx(2 * sum(math.comb(12, i) for i in range(3)) / 2**12)


def test_compare_runs(tmp_path):
    a = tmp_path / "a.jsonl"
    a.write_text("".join(json.dumps({"task_id": f"t{i}", "correct": c}) + "\n"
                         for i, c in enumerate([True, True, False, False, None])))
    b = tmp_path / "b.csv"
    b.write_text("run,task_id,correct\n" + "".join(f"v19,t{i},{c}\n" for i, c in enumerate([1, 0, 1, 0, 1]))
                 + "other,t0,0\n")

    result = compare(load_results(str(a)), load_results(str(b), run="v19"))

    assert result["n"] == 5 and result["a_correct"] == 2 and result["b_correct"] == 3
    assert result["a_only"] == ["t1"] and result["b_only"] == ["t2", "t4"]  # a task without an answer is wrong
    assert result["both_right"] == 1 and result["both_wrong"] == 1
    with pytest.raises(ValueError, match="name the run"):
        load_results(str(b))
