"""
Sources of target distributions over a decision row's options, used by the labeling pass
(data/decision_label.py) to add training targets beyond the gold label.

A causal LM gives a distribution over a row's options when its prompt labels each option with a
single token, as its next-token probabilities for those labels, at the position where it answers.
CausalLMScorer computes them. It serves two purposes:

    - The anchor: the student's base model reading each tokenized training prompt, whose options
      are labeled by the prompt format (1, 2, ...). The prompt lists the options in the row's
      augmented order, so the result is in that order.
    - A teacher model: any causal LM reading each original row through a teacher prompt
      (TEACHER_PROMPTS, selected by decision_teacher_prompt). See ModelTeacher.

Teachers can also be:

    - DecisionModelTeacher: a trained decision model's own answers (e.g. to replay a parent model).
    - TeacherFile: probabilities computed elsewhere and stored in a teacher file.

Teachers see each original row once and give its distribution in original order, which is the order of
its options in its data file, except for noul rows, which are always [false, true] (see
data/decision_prompt.py). to_augmented_order maps it to a training row's augmented order.
"""

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Sequence, Tuple, Type

import torch

from ..extras import logging
from .data_utils import flatten_decision_row
from .decision_prompt import NOUL_OPTION_NAMES, encode_decision_prompt, get_prompt_format


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from ..hparams import DataArguments
    from .decision_prompt import EncodedPrompt


logger = logging.get_logger(__name__)


def to_augmented_order(probs: Sequence[float], option_order: Sequence[int]) -> List[float]:
    """
    Reorders a distribution from a row's original option order to its augmented order.

    Args:
        probs: One probability per option, in original order.
        option_order: The row's option order: option_order[k] is the original index of the option
            shown at position k.

    Returns:
        The probabilities in augmented order.
    """
    return [probs[i] for i in option_order]


def _batches(lengths: Sequence[int], max_batch_tokens: int, pad_to_multiple_of: int) -> Iterator[Tuple[List[int], int]]:
    """
    Groups sequences into batches of similar length, longest first.

    Yields:
        The indices of each batch's sequences, and the batch's padded length (a multiple of
        pad_to_multiple_of, so that kernels compiled per sequence length are compiled only a few times).
    """
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    start = 0
    while start < len(order):
        seq_len = -(-lengths[order[start]] // pad_to_multiple_of) * pad_to_multiple_of
        batch = order[start : start + max(1, max_batch_tokens // seq_len)]
        start += len(batch)
        yield batch, seq_len


def _pad(sequences: Sequence[Sequence[int]], seq_len: int, pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pads token ids to seq_len, and returns them with their attention mask."""
    input_ids = torch.full((len(sequences), seq_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), seq_len), dtype=torch.long)
    for j, ids in enumerate(sequences):
        input_ids[j, : len(ids)] = torch.tensor(ids)
        attention_mask[j, : len(ids)] = 1
    return input_ids, attention_mask


def _default_dtype(device: str) -> torch.dtype:
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


@dataclass(frozen=True)
class ScoringQuery:
    """
    A prompt a causal LM answers with one of its options' labels.

    Args:
        input_ids: Token ids of the prompt.
        read_idx: The position whose next-token probabilities are read (where the model answers).
        labels: The label of each option, in the order the prompt shows them.
    """

    input_ids: List[int]
    read_idx: int
    labels: List[str]


class CausalLMScorer:
    """
    A frozen causal LM's distribution over the options of each prompt, given by the softmax of its
    next-token logits for the options' labels, at the query's read position. A label can be read
    only if it is a single token that decodes back to itself, so a prompt with a label that is not
    gets no distribution (e.g. "10" for most tokenizers). If the model soft-caps its logits (e.g.
    Gemma), the cap is applied, as in its own output.

    The model can be one of HMF's hybrid models. Prompts are right-padded, and like decision models
    (see uses_padding_mask), hybrid models are run without the padding mask.

    Args:
        model_name_or_path: The causal LM, with its LM head.
        device: Where to run the model.
        dtype: The model's dtype. Defaults to bfloat16 on GPU and float32 on CPU.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        dtype: Optional[torch.dtype] = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..model.hybrid_zoo.models import model_register  # noqa: F401  # Registers HMF's hybrid models
        from ..model.hybrid_zoo.models.decision.modeling_decision import uses_padding_mask

        self.name = model_name_or_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, dtype=dtype or _default_dtype(device)
        )
        self.model.to(device).eval()
        self.pass_attention_mask = uses_padding_mask(self.model)
        self.output_weight = self.model.get_output_embeddings().weight  # [vocab_size, hidden_size]
        self.softcap = getattr(self.model.config.get_text_config(), "final_logit_softcapping", None)
        self._label_ids: Dict[str, Optional[int]] = {}

    def label_id(self, label: str) -> Optional[int]:
        """The token of a label, or None if the label is not a single token that decodes back to itself."""
        if label not in self._label_ids:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            ok = len(ids) == 1 and ids[0] != self.tokenizer.unk_token_id and self.tokenizer.decode(ids) == label
            self._label_ids[label] = ids[0] if ok else None
        return self._label_ids[label]

    @torch.no_grad()
    def __call__(
        self,
        queries: Sequence[ScoringQuery],
        max_batch_tokens: int = 16384,
        pad_to_multiple_of: int = 256,
    ) -> List[Optional[List[float]]]:
        """
        Args:
            queries: The prompts.
            max_batch_tokens: Padded tokens per forward pass. Defaults to 16384.
            pad_to_multiple_of: Prompts are padded to a multiple of this length. Defaults to 256.

        Returns:
            Each prompt's distribution over its options, in the order the prompt shows them, or
            None if one of its labels cannot be read.
        """
        results: List[Optional[List[float]]] = [None] * len(queries)
        label_ids = [[self.label_id(label) for label in query.labels] for query in queries]
        rows = [i for i in range(len(queries)) if None not in label_ids[i]]

        decoder = self.model.get_decoder()
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        lengths = [len(queries[i].input_ids) for i in rows]
        for batch, seq_len in _batches(lengths, max_batch_tokens, pad_to_multiple_of):
            batch = [rows[b] for b in batch]
            input_ids, attention_mask = _pad([queries[i].input_ids for i in batch], seq_len, pad_token_id)
            attention_mask = attention_mask.to(self.device) if self.pass_attention_mask else None
            hidden = decoder(
                input_ids=input_ids.to(self.device), attention_mask=attention_mask, use_cache=False
            ).last_hidden_state  # [batch, seq_len, hidden_size]

            for j, i in enumerate(batch):
                state = hidden[j, queries[i].read_idx].float()  # [hidden_size]
                logits = self.output_weight[label_ids[i]].float() @ state  # [num_options]
                if self.softcap:
                    logits = self.softcap * torch.tanh(logits / self.softcap)
                results[i] = torch.softmax(logits, dim=-1).tolist()
        return results


# Teacher prompts


@dataclass(frozen=True)
class TeacherQuery:
    """
    An original row put to a teacher model.

    Args:
        query: The prompt, and the labels of the options it shows.
        original_order: original_order[k] is the original index of the option shown at position k.
    """

    query: ScoringQuery
    original_order: List[int]


class TeacherPrompt(ABC):
    """
    How a teacher model is asked an original decision row, which is the prompt it reads and the
    labels of its options. Teacher prompts are registered in TEACHER_PROMPTS and selected with
    decision_teacher_prompt.
    """

    @classmethod
    def from_data_args(cls, data_args: "DataArguments") -> "TeacherPrompt":
        """The teacher prompt for a config. Defaults to one without settings."""
        return cls()

    @abstractmethod
    def build(self, tokenizer: "PreTrainedTokenizer", row: Dict[str, Any]) -> Optional[TeacherQuery]:
        """
        Args:
            tokenizer: The teacher's tokenizer.
            row: The original row, as stored in its data file (see read_decision_rows).

        Returns:
            The TeacherQuery, or None if the row cannot be put in this prompt (e.g. it has more
            options than the prompt has labels).
        """
        ...


class DecisionTeacherPrompt(TeacherPrompt):
    """
    The decision prompt the student trains on (decision_prompt_format), with the options in original
    order and the question's original phrasing. The teacher answers with the options' labels in
    that format (1, 2, ...) after the answer marker. Suits any causal LM, including base models.

    Args:
        prompt_format: The prompt format's name.
        force_bos: Whether to start the prompt with the BOS token (see decision_force_bos).
    """

    def __init__(self, prompt_format: str, force_bos: bool = False):
        self.prompt_format = get_prompt_format(prompt_format)
        self.force_bos = force_bos

    @classmethod
    def from_data_args(cls, data_args: "DataArguments") -> "DecisionTeacherPrompt":
        return cls(data_args.decision_prompt_format, data_args.decision_force_bos)

    def encode(self, tokenizer: "PreTrainedTokenizer", row: Dict[str, Any]) -> "EncodedPrompt":
        """The row rendered as in training, in original order, and tokenized."""
        row = flatten_decision_row(row)
        options = [tuple(option) for option in row["options"] or []]
        rendered = self.prompt_format.render(row["state"], row["instructions"], row["kind"], options)
        return encode_decision_prompt(tokenizer, rendered, force_bos=self.force_bos)

    def build(self, tokenizer: "PreTrainedTokenizer", row: Dict[str, Any]) -> Optional[TeacherQuery]:
        encoded = self.encode(tokenizer, row)
        num_options = len(encoded.opt_idx)
        labels = [self.prompt_format.option_label(k) for k in range(num_options)]
        return TeacherQuery(ScoringQuery(encoded.input_ids, encoded.answer_idx, labels), list(range(num_options)))


def _as_text(content: Any) -> str:
    """Strings as they are; None as an empty string; other content as JSON."""
    if content is None:
        return ""
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


class StrandsTeacherPrompt(TeacherPrompt):
    """
    The teacher prompt of Strands Decider, which reads a frozen instruct model's answer from its
    probabilities for the options' letters. It goes through the teacher's chat template, with
    thinking disabled:

        system: SYSTEM
        user:   {"evidence": <state>, "criterion": <instructions>,
                 "options": [{"letter": "A", "description": "<name>: <description>"}, ...]}

    The teacher answers with an option's letter, LABELS[k] for the option shown at position k, so
    rows with more options than LABELS get no teacher. The options are described as Strands does:
    noul as "true: ..." then "false: ...", choice as "<name>: <description>", score as
    "<level>: <description>". Suits instruction-tuned models with a chat template.
    """

    LABELS = "ABCDEFGHIJKLMNOP"
    SYSTEM = (
        "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
        "Respond with only its uppercase letter, with no explanation or reasoning."
    )

    @classmethod
    def options(cls, kind: str, options: Sequence[Sequence[Any]]) -> Tuple[List[str], List[int]]:
        """
        The description of each option as shown, and the original index of each shown option.

        Args:
            kind: One of DECISION_KINDS.
            options: The row's (name, description) pairs, as stored.
        """
        if kind == "noul":
            # Shown true first; the original order is [false, true]
            descriptions = {name: desc for name, desc in options}
            shown = []
            for name in ("true", "false"):
                desc = descriptions.get(name)
                shown.append(f"{name}: " + (_as_text(desc) if desc is not None else f"The proposition is {name}."))
            return shown, [NOUL_OPTION_NAMES.index("true"), NOUL_OPTION_NAMES.index("false")]
        if kind == "choice":
            return [f"{name}: " + (_as_text(desc) or name) for name, desc in options], list(range(len(options)))
        return [f"{level}: {_as_text(desc)}" for level, (_, desc) in enumerate(options)], list(range(len(options)))

    def build(self, tokenizer: "PreTrainedTokenizer", row: Dict[str, Any]) -> Optional[TeacherQuery]:
        if tokenizer.chat_template is None:
            raise ValueError("decision_teacher_prompt: strands needs a teacher with a chat template.")
        descriptions, original_order = self.options(row["kind"], row["options"] or [])
        if len(descriptions) > len(self.LABELS):
            return None

        payload = {
            "evidence": row["state"],
            "criterion": row["instructions"],
            "options": [{"letter": self.LABELS[k], "description": desc} for k, desc in enumerate(descriptions)],
        }
        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        input_ids = tokenizer.encode(str(text), add_special_tokens=False)
        labels = list(self.LABELS[: len(descriptions)])
        return TeacherQuery(ScoringQuery(input_ids, len(input_ids) - 1, labels), original_order)


# Teacher prompts, selected by decision_teacher_prompt
TEACHER_PROMPTS: Dict[str, Type[TeacherPrompt]] = {
    "decision": DecisionTeacherPrompt,
    "strands": StrandsTeacherPrompt,
}


def get_teacher_prompt(data_args: "DataArguments") -> TeacherPrompt:
    """The teacher prompt selected by decision_teacher_prompt."""
    name = data_args.decision_teacher_prompt
    if name not in TEACHER_PROMPTS:
        raise ValueError(f"Unknown decision_teacher_prompt '{name}'. Supported prompts: {list(TEACHER_PROMPTS)}")
    return TEACHER_PROMPTS[name].from_data_args(data_args)


# Teachers


class ModelTeacher:
    """
    A causal LM teacher. Each original row is put to the model with a teacher prompt, and its
    distribution over the options' labels (see CausalLMScorer) is mapped back to original order.

    Args:
        scorer: The teacher model.
        prompt: The teacher prompt.
        max_len: Prompts longer than this many tokens get no teacher (they are not truncated, which
            would cut the evidence). Defaults to None (no limit).
    """

    def __init__(self, scorer: CausalLMScorer, prompt: TeacherPrompt, max_len: Optional[int] = None):
        self.scorer = scorer
        self.prompt = prompt
        self.max_len = max_len

    def __call__(self, rows: Sequence[Dict[str, Any]]) -> List[Optional[List[float]]]:
        """Each original row's distribution in original order, or None if it gets no teacher."""
        built = [self.prompt.build(self.scorer.tokenizer, row) for row in rows]
        keep = [
            i for i, b in enumerate(built)
            if b is not None and (self.max_len is None or len(b.query.input_ids) <= self.max_len)
        ]
        scored = self.scorer([built[i].query for i in keep])

        results: List[Optional[List[float]]] = [None] * len(rows)
        for i, probs in zip(keep, scored):
            if probs is not None:
                original = [0.0] * len(probs)
                for position, index in enumerate(built[i].original_order):
                    original[index] = probs[position]
                results[i] = original
        return results


class DecisionModelTeacher:
    """
    A trained decision model as the teacher, which gives its own distribution (temperature 1) over each original
    row's options, rendered with the student's prompt format (decision_prompt_format) in original
    order and with the question's original phrasing. Training toward it keeps a new model close to
    an earlier one (replay).

    Args:
        model_name_or_path: The decision model (an hmf_decision checkpoint).
        device: Where to run the model.
        adapter_name_or_path: A LoRA adapter trained on the model, merged before labeling. Defaults
            to None.
        prompt_format: The prompt format's name. Defaults to "strands".
        force_bos: Whether to start prompts with the BOS token (see decision_force_bos). Defaults to
            False.
        max_len: Prompts longer than this many tokens get no teacher. Defaults to None (no limit).
        dtype: The model's dtype. Defaults to bfloat16 on GPU and float32 on CPU.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        adapter_name_or_path: Optional[str] = None,
        prompt_format: str = "strands",
        force_bos: bool = False,
        max_len: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        from transformers import AutoModel, AutoTokenizer

        from ..model.hybrid_zoo.models import model_register  # noqa: F401  # Registers decision models

        self.name = model_name_or_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path, dtype=dtype or _default_dtype(device))
        if adapter_name_or_path is not None:
            from peft import PeftModel

            # Merging folds the LoRA weights into the backbone and puts the trained head in place
            model = PeftModel.from_pretrained(model, adapter_name_or_path).merge_and_unload()
        self.model = model.to(device).eval()
        self.prompt = DecisionTeacherPrompt(prompt_format, force_bos)
        self.max_len = max_len

    def __call__(
        self,
        rows: Sequence[Dict[str, Any]],
        max_batch_tokens: int = 16384,
        pad_to_multiple_of: int = 256,
    ) -> List[Optional[List[float]]]:
        """Each original row's distribution in original order, or None if its prompt is too long."""
        log_probs = self.log_probs(rows, max_batch_tokens, pad_to_multiple_of)
        return [None if lp is None else [math.exp(x) for x in lp] for lp in log_probs]

    @torch.no_grad()
    def log_probs(
        self,
        rows: Sequence[Dict[str, Any]],
        max_batch_tokens: int = 16384,
        pad_to_multiple_of: int = 256,
    ) -> List[Optional[List[float]]]:
        """
        Each original row's log-probabilities over its options (temperature 1), in original order, or
        None if its prompt is too long.
        """
        encoded = [self.prompt.encode(self.tokenizer, row) for row in rows]
        keep = [i for i, e in enumerate(encoded) if self.max_len is None or len(e.input_ids) <= self.max_len]
        encoded = [encoded[i] for i in keep]

        results: List[Optional[List[float]]] = [None] * len(rows)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        lengths = [len(e.input_ids) for e in encoded]
        for batch, seq_len in _batches(lengths, max_batch_tokens, pad_to_multiple_of):
            input_ids, attention_mask = _pad([encoded[b].input_ids for b in batch], seq_len, pad_token_id)
            num_options = max(len(encoded[b].opt_idx) for b in batch)
            opt_idx = torch.full((len(batch), num_options), -1, dtype=torch.long)
            for j, b in enumerate(batch):
                opt_idx[j, : len(encoded[b].opt_idx)] = torch.tensor(encoded[b].opt_idx)
            answer_idx = torch.tensor([encoded[b].answer_idx for b in batch])

            out = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                opt_idx=opt_idx.to(self.device),
                answer_idx=answer_idx.to(self.device),
            )
            for j, b in enumerate(batch):
                results[keep[b]] = out.log_probs[j, : len(encoded[b].opt_idx)].float().tolist()
        return results


def is_decision_model(model_name_or_path: str) -> bool:
    """Whether a checkpoint is a decision model (an hmf_decision checkpoint)."""
    from transformers import AutoConfig

    from ..model.hybrid_zoo.models import model_register  # noqa: F401  # Registers decision models

    return AutoConfig.from_pretrained(model_name_or_path).model_type == "hmf_decision"


def load_teacher_model(data_args: "DataArguments", device: str):
    """
    The teacher model of a config (decision_teacher_model), which is a DecisionModelTeacher for a
    decision model and otherwise a ModelTeacher with the teacher prompt decision_teacher_prompt.
    """
    path = data_args.decision_teacher_model
    if is_decision_model(path):
        logger.info_rank0(f"Teacher: the decision model {path}.")
        return DecisionModelTeacher(
            path,
            device,
            adapter_name_or_path=data_args.decision_teacher_adapter,
            prompt_format=data_args.decision_prompt_format,
            force_bos=data_args.decision_force_bos,
            max_len=data_args.decision_teacher_max_len,
        )

    if data_args.decision_teacher_adapter is not None:
        raise ValueError("decision_teacher_adapter is for decision models; the teacher model is a causal LM.")
    logger.info_rank0(f"Teacher: the causal LM {path}, with the {data_args.decision_teacher_prompt} prompt.")
    return ModelTeacher(
        CausalLMScorer(path, device),
        get_teacher_prompt(data_args),
        max_len=data_args.decision_teacher_max_len,
    )


class TeacherFile:
    """
    A teacher file, which holds probabilities over the options of some rows, computed elsewhere (e.g. by a
    stronger model, or by a previous decision model). It is a JSON Lines file with one line per
    row:

        {"source": "strands-decider/multistep_v14.jsonl", "row_index": 17, "probs": [0.82, 0.11, 0.07]}

    - source: the dataset's file_name in dataset_info.json (the tokenized data's source column).
    - row_index: the row's position among the non-empty lines of that file, counting from 0.
    - probs: one probability per option, in the row's original order (the order its options are
      listed in the data file; for noul rows, [false, true]), summing to 1.

    Rows not in the file get no teacher.

    Args:
        path: The teacher file.
    """

    def __init__(self, path: str):
        self.path = path
        self.probs: Dict[Tuple[str, int], List[float]] = {}
        with open(path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                entry = json.loads(line)
                missing = {"source", "row_index", "probs"} - set(entry)
                if missing:
                    raise ValueError(f"{path} line {line_number}: missing {sorted(missing)}.")
                key = (entry["source"], int(entry["row_index"]))
                if key in self.probs:
                    raise ValueError(f"{path} line {line_number}: a second entry for {key}.")
                probs = [float(p) for p in entry["probs"]]
                if abs(sum(probs) - 1.0) > 1e-3:
                    raise ValueError(f"{path} line {line_number}: probs sum to {sum(probs):.4f}, not 1.")
                self.probs[key] = probs
        logger.info_rank0(f"Teacher file {path}: {len(self.probs):,} rows.")

    def get(self, source: str, row_index: int) -> Optional[List[float]]:
        """The row's probabilities in original order, or None if the file has none for it."""
        return self.probs.get((source, row_index))
