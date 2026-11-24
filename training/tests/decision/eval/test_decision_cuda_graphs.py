"""
Tests for answering through CUDA graphs (eval/decision/graphs.py): the runner on tiny Qwen3 and Qwen3.5 decision
models, causal and bidirectional, and the predictor on the eval tests' tiny trained model. They need a GPU and the
flash_attn package, except the check that the runner refuses a model it cannot record.
"""

import pytest
import torch
import transformers

import hmf.model.hybrid_zoo.models.model_register  # noqa: F401  # Registers decision models
from hmf.eval.decision.graphs import DecisionGraphRunner
from hmf.eval.decision.predictor import DecisionPredictor
from hmf.model.hybrid_zoo.models.decision.configuration_decision import DecisionConfig
from hmf.model.hybrid_zoo.models.decision.modeling_decision import DecisionModel


VOCAB_SIZE = 128
PAD = 1
SEED = 0

BACKBONES = {
    "qwen3": lambda: transformers.Qwen3Config(
        vocab_size=VOCAB_SIZE, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, tie_word_embeddings=True,
    ),
    "qwen3_5_text": lambda: getattr(transformers, "Qwen3_5TextConfig")(
        vocab_size=VOCAB_SIZE, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, layer_types=["linear_attention", "full_attention"], tie_word_embeddings=True,
    ),
}


def _needs_flash_attention() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs need a GPU")
    pytest.importorskip("flash_attn")


def _model(name: str, attention: str) -> DecisionModel:
    """A tiny decision model in bfloat16 on the GPU, with flash attention."""
    _needs_flash_attention()
    if name == "qwen3_5_text" and not hasattr(transformers, "Qwen3_5TextConfig"):
        pytest.skip("this transformers has no Qwen3.5")
    torch.manual_seed(SEED)
    config = DecisionConfig(text_config=BACKBONES[name](), decision_head_dim=16, decision_attention=attention)
    model = DecisionModel(config).eval().to("cuda", torch.bfloat16)
    model.model.set_attn_implementation("flash_attention_2")
    return model


def _prompts(lengths, num_options, seed=SEED):
    """Random prompts of the given lengths, each with num_options options before its answer marker, last."""
    generator = torch.Generator().manual_seed(seed)
    input_ids, opt_idx, answer_idx = [], [], []
    for length, options in zip(lengths, num_options):
        input_ids.append(torch.randint(2, VOCAB_SIZE, (length,), generator=generator).tolist())
        opt_idx.append([length - 1 - options + k for k in range(options)])
        answer_idx.append(length - 1)
    return input_ids, opt_idx, answer_idx


def _masked(model, input_ids, opt_idx, answer_idx):
    """The answers of the ordinary pass (padding mask, no graph), padded to the longest prompt."""
    length, options = max(map(len, input_ids)), max(map(len, opt_idx))
    ids = torch.full((len(input_ids), length), PAD, dtype=torch.long)
    mask = torch.zeros_like(ids)
    opts = torch.full((len(input_ids), options), -1, dtype=torch.long)
    for j, row in enumerate(input_ids):
        ids[j, : len(row)] = torch.tensor(row)
        mask[j, : len(row)] = 1
        opts[j, : len(opt_idx[j])] = torch.tensor(opt_idx[j])
    with torch.no_grad():
        out = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), opt_idx=opts.cuda(),
                    answer_idx=torch.tensor(answer_idx).cuda())
    return [out.log_probs[j, : len(opt_idx[j])].float().cpu().tolist() for j in range(len(input_ids))]


def _close(a, b, tol=1e-2):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        torch.testing.assert_close(torch.tensor(x), torch.tensor(y), rtol=tol, atol=tol)


CASES = [
    ([40], [3]),  # one prompt
    ([200, 130, 17], [3, 5, 2]),  # three prompts of different lengths, in a batch of four
    ([300, 90], [9, 2]),  # more options than the smallest option bucket
]


@pytest.mark.parametrize("name", list(BACKBONES))
@pytest.mark.parametrize("attention", ["causal", "bidirectional"])
@pytest.mark.parametrize("lengths,num_options", CASES)
def test_graphs_match_the_ordinary_pass(name, attention, lengths, num_options):
    model = _model(name, attention)
    runner = DecisionGraphRunner(model, PAD)
    prompts = _prompts(lengths, num_options)
    graphed = runner(*prompts)
    assert len(runner.graphs) == 1
    _close(graphed, _masked(model, *prompts))
    # A second call replays the recorded graph and gives the same answers exactly
    assert runner(*prompts) == graphed


@pytest.mark.parametrize("name", list(BACKBONES))
def test_graphs_replay_with_new_inputs_and_other_shapes_in_between(name):
    model = _model(name, "bidirectional")
    runner = DecisionGraphRunner(model, PAD)
    first, second = _prompts([100], [3], seed=1), _prompts([100], [3], seed=2)
    other = _prompts([500, 20], [4, 4], seed=3)
    runner(*first)
    runner(*other)
    # The first bucket's graph, replayed on new inputs after another graph ran, answers them correctly
    _close(runner(*second), _masked(model, *second))
    _close(runner(*other), _masked(model, *other))
    assert len(runner.graphs) == 2


@pytest.mark.parametrize("name", list(BACKBONES))
def test_answers_do_not_depend_on_the_batch_or_the_bucket(name):
    model = _model(name, "bidirectional")
    runner = DecisionGraphRunner(model, PAD)
    input_ids, opt_idx, answer_idx = _prompts([150, 60, 33], [3, 4, 2])
    together = runner(input_ids, opt_idx, answer_idx)
    alone = [runner([input_ids[j]], [opt_idx[j]], [answer_idx[j]])[0] for j in range(3)]
    _close(together, alone)


@pytest.mark.parametrize("name", list(BACKBONES))
def test_batches_outside_every_bucket_run_without_a_graph(name):
    model = _model(name, "bidirectional")
    runner = DecisionGraphRunner(model, PAD, max_batch=2, max_length=256, option_buckets=(4,))
    for prompts in (_prompts([50] * 3, [3] * 3), _prompts([300], [3]), _prompts([50], [6])):
        _close(runner(*prompts), _masked(model, *prompts))
    assert runner.graphs == {}


def test_bucket():
    runner = DecisionGraphRunner.__new__(DecisionGraphRunner)
    runner.max_batch, runner.length_multiple, runner.max_length, runner.option_buckets = 8, 128, 4096, (8, 16, 32)
    assert runner.bucket(1, 1, 1) == (1, 128, 8)
    assert runner.bucket(3, 129, 9) == (4, 256, 16)
    assert runner.bucket(8, 4096, 32) == (8, 4096, 32)
    assert runner.bucket(9, 100, 3) is None
    assert runner.bucket(1, 4097, 3) is None
    assert runner.bucket(1, 100, 33) is None


def test_warm_up_records_the_graphs_ahead():
    runner = DecisionGraphRunner(_model("qwen3", "causal"), PAD)
    runner.warm_up([100, 128, 300], batches=(1, 2))
    assert set(runner.graphs) == {(1, 128, 8), (1, 384, 8), (2, 128, 8), (2, 384, 8)}


def test_warm_up_keeps_no_activations():
    model = _model("qwen3_5_text", "bidirectional")
    runner = DecisionGraphRunner(model, PAD)
    runner.warm_up([128])
    before = torch.cuda.memory_allocated()
    # Graphs recorded outside the runner's own no_grad, as the server's warm-up records them, keep no activations
    # for a backward pass, so the memory in use barely grows with more graphs
    runner.warm_up(range(256, 2048 + 128, 128))
    assert all(not entry.log_probs.requires_grad for entry in runner.graphs.values())
    assert torch.cuda.memory_allocated() - before < 64 * 2**20


def test_graphs_make_the_cpu_wait_only_to_read_the_answers():
    runner = DecisionGraphRunner(_model("qwen3_5_text", "bidirectional"), PAD)
    prompts = _prompts([120], [3])
    runner(*prompts)
    entry = next(iter(runner.graphs.values()))
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        entry.graph.replay()
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_the_runner_refuses_a_model_it_cannot_record():
    torch.manual_seed(SEED)
    model = DecisionModel(DecisionConfig(text_config=BACKBONES["qwen3"](), decision_head_dim=16)).eval()
    with pytest.raises(ValueError, match="GPU"):
        DecisionGraphRunner(model, PAD)
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="flash attention"):
            DecisionGraphRunner(model.to("cuda"), PAD)  # Loaded with transformers' default attention


def test_the_predictor_answers_the_same_with_and_without_graphs(trained):
    _needs_flash_attention()
    rows = [trained["row"](i) for i in range(40)]
    graphed = DecisionPredictor(trained["model_dir"], trained["adapter_dir"], device="cuda", cuda_graphs=True)
    plain = DecisionPredictor(trained["model_dir"], trained["adapter_dir"], device="cuda")
    assert graphed.model.graphs is not None and plain.model.graphs is None
    with_graphs, without = graphed.log_probs(rows), plain.log_probs(rows)
    _close(with_graphs, without, tol=2e-2)
    _close(graphed.probabilities(rows), plain.probabilities(rows), tol=2e-2)


def test_the_predictor_skips_the_same_long_rows_with_graphs(trained):
    _needs_flash_attention()
    rows = [trained["row"](0), {**trained["row"](1), "state": "state " * 300}, trained["row"](2)]
    graphed = DecisionPredictor(trained["model_dir"], device="cuda", max_len=64, cuda_graphs=True)
    answers = graphed.log_probs(rows)
    assert answers[1] is None and answers[0] is not None and answers[2] is not None


def test_the_server_answers_through_the_graphs(trained, monkeypatch):
    _needs_flash_attention()
    from hmf.eval.decision.server import DecisionRequest, DecisionServer

    questions = {
        "yes_no": {"type": "noul", "instructions": "question ?", "criteria": {"true": "high"}},
        "pick": {"type": "choice", "instructions": "question ?",
                 "criteria": {"refund": "refund", "ignore": None, "escalate": "escalate"}},
        "rate": {"type": "score", "instructions": "question ?", "criteria": ["low", "low", "high", "high"]},
    }
    request = DecisionRequest(state="state state refund", questions=questions)
    graphed = DecisionServer(DecisionPredictor(trained["model_dir"], trained["adapter_dir"], device="cuda",
                                               calibration=None, cuda_graphs=True))
    plain = DecisionServer(DecisionPredictor(trained["model_dir"], trained["adapter_dir"], device="cuda",
                                             calibration=None))
    replays = []
    original = torch.cuda.CUDAGraph.replay
    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", lambda self: (replays.append(1), original(self))[1])
    with_graphs, without = graphed.answer(request), plain.answer(request)
    assert replays, "the server answered without replaying a graph"
    assert with_graphs["usage"] == without["usage"]
    a, b = with_graphs["answers"], without["answers"]
    _close([[a["yes_no"]["noul"]]], [[b["yes_no"]["noul"]]], tol=2e-2)
    for name in ("pick", "rate"):
        _close([list(a[name]["probabilities"].values())], [list(b[name]["probabilities"].values())], tol=2e-2)
    _close([[a["rate"]["score"]]], [[b["rate"]["score"]]], tol=5e-2)
