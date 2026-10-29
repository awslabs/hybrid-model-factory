"""
A decision server for a trained decision model, using JevBench's "typesafe" protocol (POST
/v1/systemone), which Strands Decider's server also serves, so that JevBench's harness can run against it.

    hmf decision-serve path/to/decision-model --adapter_name_or_path path/to/lora

The server listens on 127.0.0.1 only, so only programs on the same machine can reach it, and it has
no authentication. Any other --host is refused. To query it from another machine, forward the port
over SSH (ssh -N -L 8099:127.0.0.1:8099 <server-machine>). In a container, do not publish its port.

The protocol. A request holds a state (text or JSON) and one or more named questions about it:

    POST /v1/systemone
    {"state": "...", "model": "...", "questions": {"decision": {"type": "choice", "instructions": "...",
                                                                 "criteria": {"refund": "...", "ignore": null}}}}

    - noul: a yes/no statement; criteria, optional, describes "true" and "false".
    - choice: one of 2 to 255 named options; criteria maps each name to a description (or null).
    - score: a rating; criteria lists 2 to 10 levels, lowest first.

The reply holds one answer per question (probabilities rounded to 4 decimals):

    {"model": "...", "answers": {"decision": {...}}, "usage": {"input_tokens": n, "output_tokens": n},
     "latency_ms": t}

    - noul: {"type": "noul", "noul": P(true)}
    - choice: {"type": "choice", "choice": the most probable option, "probabilities": {name: p},
      "confidence": c}
    - score: {"type": "score", "score": the expected level, "legend": {"0": level 0's text, ...},
      "probabilities": {"0": p, ...}, "confidence": c}

Confidence is defined in metrics.py. GET /health describes the model served. An invalid request is
refused with status 422 and the reason.

Each question is rendered with the model's prompt format, its options in the order the request lists
them, and answered at the model's calibrated temperatures (see calibrate.py), or at temperature 1
with --raw. The questions of a request are answered in one forward pass.

A prompt longer than the window (--max_len) is shortened as Strands Decider's server shortens it,
unless --strict_window is set. The question and its options keep all their tokens (up to 3/4 of the
window), and the state is cut to fit, from its end. Without truncation, a prompt is tokenized exactly
as in training.
"""

import ipaddress
import os
import threading
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Annotated

from ...data.data_utils import flatten_decision_row
from ...data.decision_prompt import (
    NOUL_OPTION_NAMES,
    EncodedPrompt,
    RenderedPrompt,
    _last_token_in_span,
    encode_decision_prompt,
    render_content,
)
from ...extras import logging
from .metrics import DEFAULT_ORDINAL_SMOOTHING, apply_temperature, choice_confidence, score_confidence
from .predictor import DecisionPredictor


logger = logging.get_logger(__name__)


DEFAULT_PORT = 8099
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10
# The largest share of the window a question and its options may take before the state is cut
MAX_QUESTION_FRACTION = 0.75

# A state, instructions, or option description, as text or JSON (rendered as indented JSON)
Content = Union[str, Dict[str, Any], List[Any]]


# The protocol


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: Content
    criteria: Optional[Dict[str, Optional[Content]]] = None

    @field_validator("criteria")
    @classmethod
    def _check_names(cls, criteria):
        if criteria is not None and not set(criteria) <= set(NOUL_OPTION_NAMES):
            raise ValueError(f"noul criteria must be named from {list(NOUL_OPTION_NAMES)}, got {sorted(criteria)}")
        return criteria


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Content
    criteria: Dict[str, Optional[Content]]

    @field_validator("criteria")
    @classmethod
    def _check_size(cls, criteria):
        if not 2 <= len(criteria) <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice needs 2 to {MAX_CHOICE_OPTIONS} options, got {len(criteria)}")
        return criteria


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Content
    criteria: List[str]

    @field_validator("criteria")
    @classmethod
    def _check_levels(cls, criteria):
        if not MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS:
            raise ValueError(f"score needs {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels, got {len(criteria)}")
        return criteria


Question = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")]


class DecisionRequest(BaseModel):
    state: Content
    questions: Dict[str, Question] = Field(..., min_length=1)
    model: Optional[str] = None
    images: List[str] = Field(default_factory=list)

    @field_validator("images")
    @classmethod
    def _no_images(cls, images):
        if images:
            raise ValueError("images are not supported: this server answers about text states only")
        return images


def question_row(state: Content, question: Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]) -> Dict[str, Any]:
    """A question as a decision row (see data/README.md), with its options in the order the request lists them."""
    if question.type == "noul":
        options = [[name, desc] for name, desc in (question.criteria or {}).items()]
    elif question.type == "choice":
        options = [[name, desc] for name, desc in question.criteria.items()]
    else:
        options = [[str(level), desc] for level, desc in enumerate(question.criteria)]
    return {"kind": question.type, "state": state, "instructions": question.instructions, "options": options}


# Answering


class WindowError(ValueError):
    """A prompt that does not fit the window (with --strict_window), or whose options do not survive truncation."""


class DecisionServer:
    """
    Answers requests of the protocol with a trained decision model.

    Args:
        predictor: The model (see predictor.py), with its calibration unless raw.
        raw: Whether to answer at temperature 1, ignoring the calibration. Defaults to False.
        strict_window: Whether to refuse prompts longer than the window rather than cut their state.
            Defaults to False.
        model_name: The name replies give as their model. Defaults to the adapter's folder name, else
            the model's.
        max_len: The context window, in tokens. Defaults to the predictor's.
    """

    def __init__(
        self,
        predictor: DecisionPredictor,
        raw: bool = False,
        strict_window: bool = False,
        model_name: Optional[str] = None,
        max_len: Optional[int] = None,
    ):
        self.predictor = predictor
        self.raw = raw
        self.strict_window = strict_window
        self.tokenizer = predictor.model.tokenizer
        self.prompt = predictor.model.prompt  # the prompt format and force_bos, as in training
        self.max_len = max_len or predictor.max_len
        self.ordinal_smoothing = (predictor.calibration or {}).get("ordinal_smoothing", DEFAULT_ORDINAL_SMOOTHING)
        source = predictor.adapter_name_or_path or predictor.model_name_or_path
        self.model_name = model_name or os.path.basename(source.rstrip("/"))
        self._lock = threading.Lock()  # one request at a time on the model

    def temperature(self, kind: str) -> float:
        return 1.0 if self.raw else self.predictor.temperature(kind)

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "model": self.model_name,
            "model_name_or_path": self.predictor.model_name_or_path,
            "adapter_name_or_path": self.predictor.adapter_name_or_path,
            "max_length": self.max_len,
            "calibrated": self.predictor.calibration is not None and not self.raw,
            "temperature_by_kind": {kind: self.temperature(kind) for kind in ("noul", "choice", "score")},
            "strict_window": self.strict_window,
            "device": self.predictor.device,
        }

    def _render(self, row: Dict[str, Any]) -> Tuple[RenderedPrompt, str]:
        flat = flatten_decision_row(row)
        options = [tuple(option) for option in flat["options"] or []]
        rendered = self.prompt.prompt_format.render(flat["state"], flat["instructions"], flat["kind"], options)
        return rendered, flat["state"]

    def _encode(self, rendered: List[RenderedPrompt], states: List[str]) -> List[EncodedPrompt]:
        """The questions' prompts as model inputs, shortened to the window if they do not fit."""
        encoded = [encode_decision_prompt(self.tokenizer, r, force_bos=self.prompt.force_bos) for r in rendered]
        if self.max_len is None or all(len(e.input_ids) <= self.max_len for e in encoded):
            return encoded  # tokenized exactly as in training
        if self.strict_window:
            longest = max(len(e.input_ids) for e in encoded)
            raise WindowError(f"prompt of {longest} tokens exceeds the context window of {self.max_len} tokens")
        return self._truncate(rendered, states)

    def _truncate(self, rendered: List[RenderedPrompt], states: List[str]) -> List[EncodedPrompt]:
        """
        Strands Decider's rule, where the questions (all of the prompt after the state) keep their tokens, up
        to MAX_QUESTION_FRACTION of the window, and the state takes the rest of the window, cut from
        its end. A question longer than its share loses its first tokens; its options and the answer
        marker, at its end, are kept.
        """
        prefixes, questions = [], []
        for r, state in zip(rendered, states):
            # The prompt format starts with the state block, which is what gets cut
            prefix = f"<state>\n{render_content(state)}\n</state>\n"
            if not r.text.startswith(prefix):
                raise WindowError("this prompt format cannot be shortened: it does not start with the state")
            split = len(prefix)
            prefixes.append(prefix)
            questions.append(self.tokenizer(r.text[split:], add_special_tokens=False, return_offsets_mapping=True))

        longest = max(len(q["input_ids"]) for q in questions)
        reserve = min(longest, max(1, int(self.max_len * MAX_QUESTION_FRACTION)))
        state_ids = self.tokenizer(
            prefixes[0], add_special_tokens=True, truncation=True, max_length=max(1, self.max_len - reserve)
        )["input_ids"]
        bos = self.tokenizer.bos_token_id
        if self.prompt.force_bos and bos is not None and (not state_ids or state_ids[0] != bos):
            state_ids = [bos] + state_ids[: max(0, self.max_len - reserve - 1)]

        encoded = []
        for r, prefix, q in zip(rendered, prefixes, questions):
            cut = max(0, len(q["input_ids"]) - reserve)
            ids, offsets = q["input_ids"][cut:], q["offset_mapping"][cut:]
            spans = [(start - len(prefix), end - len(prefix)) for start, end in r.option_spans]
            opt_idx = [_last_token_in_span(offsets, start, end) for start, end in spans]
            answer_idx = _last_token_in_span(offsets, r.answer_span[0] - len(prefix), r.answer_span[1] - len(prefix))
            if min(opt_idx + [answer_idx]) < 0:
                raise WindowError("the question and its options do not fit the context window")
            base = len(state_ids)
            encoded.append(EncodedPrompt(state_ids + ids, [base + i for i in opt_idx], base + answer_idx))
        return encoded

    @torch.no_grad()
    def answer(self, request: DecisionRequest) -> Dict[str, Any]:
        """The reply to a request."""
        names = list(request.questions)
        questions = [request.questions[name] for name in names]
        rendered, states = zip(*(self._render(question_row(request.state, q)) for q in questions))
        encoded = self._encode(list(rendered), list(states))

        seq_len = max(len(e.input_ids) for e in encoded)
        num_options = max(len(e.opt_idx) for e in encoded)
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        input_ids = torch.full((len(encoded), seq_len), pad, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), seq_len), dtype=torch.long)
        opt_idx = torch.full((len(encoded), num_options), -1, dtype=torch.long)
        for i, e in enumerate(encoded):
            input_ids[i, : len(e.input_ids)] = torch.tensor(e.input_ids)
            attention_mask[i, : len(e.input_ids)] = 1
            opt_idx[i, : len(e.opt_idx)] = torch.tensor(e.opt_idx)
        device = self.predictor.device
        out = self.predictor.model.model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            opt_idx=opt_idx.to(device),
            answer_idx=torch.tensor([e.answer_idx for e in encoded]).to(device),
        )

        answers = {}
        for i, (name, question) in enumerate(zip(names, questions)):
            log_probs = out.log_probs[i, : len(encoded[i].opt_idx)].float().tolist()
            probs = apply_temperature(log_probs, self.temperature(question.type))
            answers[name] = self._to_answer(question, probs)
        return {
            "model": self.model_name,
            "answers": answers,
            "usage": {"input_tokens": int(attention_mask.sum()), "output_tokens": len(names)},
        }

    def _to_answer(self, question, probs: List[float]) -> Dict[str, Any]:
        if question.type == "noul":
            # Options are shown as [false, true] whatever the request's criteria list
            return {"type": "noul", "noul": round(probs[NOUL_OPTION_NAMES.index("true")], 4)}
        if question.type == "choice":
            by_name = dict(zip(question.criteria, probs))
            return {
                "type": "choice",
                "choice": max(by_name, key=by_name.get),
                "probabilities": {name: round(p, 4) for name, p in by_name.items()},
                "confidence": round(choice_confidence(probs), 4),
            }
        return {
            "type": "score",
            "score": round(sum(level * p for level, p in enumerate(probs)), 4),
            "legend": {str(level): text for level, text in enumerate(question.criteria)},
            "probabilities": {str(level): round(p, 4) for level, p in enumerate(probs)},
            "confidence": round(score_confidence(probs, self.ordinal_smoothing), 4),
        }


# The HTTP server


def is_loopback(host: str) -> bool:
    """Whether host is this machine only (localhost, 127.0.0.0/8, or ::1)."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def create_app(server: DecisionServer):
    """The FastAPI app, with POST /v1/systemone and GET /health."""
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="HMF decision server", description="JevBench's typesafe protocol (POST /v1/systemone).")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return server.health()

    @app.post("/v1/systemone")
    def systemone(request: DecisionRequest) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            with server._lock:
                reply = server.answer(request)
        except WindowError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        reply["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return reply

    return app


def serve(
    model_name_or_path: str,
    adapter_name_or_path: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    raw: bool = False,
    calibration: Optional[str] = "auto",
    strict_window: bool = False,
    max_len: int = 4096,
    prompt_format: str = "strands",
    force_bos: bool = False,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> None:
    """
    Serves a trained decision model on 127.0.0.1.

    Args:
        model_name_or_path: The decision model (from hmf decision-init).
        adapter_name_or_path: Its LoRA adapter, the training output_dir. Defaults to None.
        host: The address to listen on: 127.0.0.1 (the default), localhost or ::1. Other addresses
            are refused, as the server has no authentication; to query it from another machine,
            forward the port over SSH.
        port: The port to listen on. Defaults to 8099.
        raw: Whether to answer at temperature 1, ignoring the model's calibration. Defaults to False.
        calibration: "auto" (decision_calibration.json in the adapter's folder, else in the model's,
            if there is one), the path of a calibration file, or None (temperature 1). Defaults to
            "auto".
        strict_window: Whether to refuse (status 422) a prompt longer than max_len, instead of cutting
            its state to fit. Benchmarks that forbid shortening their prompts need it. Defaults to
            False.
        max_len: The context window, in tokens: the longest prompt answered. Defaults to 4096.
        prompt_format: The prompt format of training (decision_prompt_format). Defaults to "strands".
        force_bos: Whether prompts start with the BOS token (decision_force_bos in training).
            Defaults to False.
        model_name: The name replies give as their model. Defaults to the adapter's folder name, else
            the model's.
        device: Where to run the model. Defaults to the first GPU if there is one, else the CPU.
    """
    if not is_loopback(host):
        raise ValueError(
            f"Refusing to listen on {host}: the server has no authentication, so it listens on this machine "
            f"only (127.0.0.1). To query it from another machine, forward the port over SSH: "
            f"ssh -N -L {port}:127.0.0.1:{port} <this machine>."
        )
    import uvicorn

    predictor = DecisionPredictor(
        model_name_or_path, adapter_name_or_path, device=device, prompt_format=prompt_format, force_bos=force_bos,
        max_len=max_len, calibration=calibration,
    )
    server = DecisionServer(predictor, raw=raw, strict_window=strict_window, model_name=model_name)
    logger.info_rank0(f"Serving {server.model_name} on http://{host}:{port} (POST /v1/systemone, GET /health).")
    uvicorn.run(create_app(server), host=host, port=port, workers=1)


if __name__ == "__main__":
    import fire

    fire.Fire(serve)
