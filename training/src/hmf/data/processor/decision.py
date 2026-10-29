"""
Dataset processor for decision models. It renders each decision row with a prompt format, tokenizes it,
and locates the token positions the decision model reads (see data/decision_prompt.py).

Training rows are augmented, so the model learns to read the options from the prompt rather than
from their position. An augmentation of a row is one augmented copy of it, rendered with
its options in one order and one phrasing of its question.

    - noul and choice options are shuffled, and the label follows its option.
    - score options keep their order (it carries meaning), but the whole rubric is reversed, with
      the label, with probability decision_reverse_score_prob.
    - One of the row's phrasings of the question is sampled.

Each training row is tokenized decision_num_augmentations times, as augmentations 0, 1, ...; the
trainer reads augmentation e in epoch e. The randomness is seeded per row and augmentation, so the
output is the same whatever the number of workers or shards. Evaluation rows have a single copy,
number 0, without augmentation.
"""

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Optional

from ...extras import logging
from ..decision_prompt import NOUL_OPTION_NAMES, adds_bos_token, encode_decision_prompt, get_prompt_format
from .processor_utils import DatasetProcessor


logger = logging.get_logger(__name__)


@dataclass
class DecisionDatasetProcessor(DatasetProcessor):
    """
    Args:
        is_eval: Whether the rows are evaluation rows, rendered once and without augmentation.
            Defaults to False.
        seed: Seed of the augmentations. Defaults to 0.
    """

    is_eval: bool = False
    seed: int = 0

    def _option_order(self, kind: str, num_options: int, rng: random.Random) -> Optional[List[int]]:
        """The rendered option order (option_order[k] = canonical index shown at k), or None for canonical."""
        if self.is_eval or not self.data_args.decision_shuffle_options:
            return None
        if kind == "score":
            if rng.random() < self.data_args.decision_reverse_score_prob:
                return list(reversed(range(num_options)))
            return None
        order = list(range(num_options))
        rng.shuffle(order)
        return order

    def _instructions(self, instructions: str, variants: List[str], rng: random.Random) -> str:
        """The canonical phrasing of the question, or for training rows one sampled from all phrasings."""
        if self.is_eval or not self.data_args.decision_vary_instructions or not variants:
            return instructions
        return rng.choice([instructions] + variants)

    def _label_dist(self, kind: str, label: int, order: List[int]) -> List[float]:
        """
        Target distribution over the rendered options. One-hot, except for score rows, where
        decision_ordinal_smoothing of the mass goes to the gold level's neighbors. Smoothing is
        applied to levels, then mapped to rendered positions, so a reversed rubric still spreads
        the mass to the neighboring levels.
        """
        level_mass = {label: 1.0}
        smoothing = self.data_args.decision_ordinal_smoothing
        if kind == "score" and smoothing > 0:
            neighbors = [level for level in (label - 1, label + 1) if 0 <= level < len(order)]
            if neighbors:
                level_mass = {label: 1.0 - smoothing, **{level: smoothing / len(neighbors) for level in neighbors}}

        dist = [0.0] * len(order)
        for level, mass in level_mass.items():
            dist[order.index(level)] += mass
        return dist

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        prompt_format = get_prompt_format(self.data_args.decision_prompt_format)
        num_augmentations = 1 if self.is_eval else self.data_args.decision_num_augmentations
        force_bos = self.data_args.decision_force_bos

        if not force_bos and self.tokenizer.bos_token_id is not None and not adds_bos_token(self.tokenizer):
            logger.warning_rank0_once(
                f"The tokenizer has a BOS token ({self.tokenizer.bos_token}) but does not add it to decision "
                "prompts. If the model expects one (e.g. an instruction-tuned model whose chat template adds "
                "it), set decision_force_bos: true."
            )

        model_inputs = defaultdict(list)
        num_too_long = 0
        for i in range(len(examples["_kind"])):
            kind, label = examples["_kind"][i], examples["_label"][i]
            options = [tuple(option) for option in examples["_options"][i]]
            num_options = len(NOUL_OPTION_NAMES) if kind == "noul" else len(options)
            if not 0 <= label < num_options:
                raise ValueError(f"Label {label} is out of range for {num_options} options (row {i} of the batch).")

            for augmentation in range(num_augmentations):
                # Seeded by the row's identity, not its position, so any sharding gives the same result
                rng = random.Random(
                    f"{self.seed}:{examples['_source'][i]}:{examples['_row_index'][i]}:{augmentation}"
                )
                order = self._option_order(kind, num_options, rng)
                instructions = self._instructions(
                    examples["_instructions"][i], examples["_instruction_variants"][i] or [], rng
                )
                rendered = prompt_format.render(
                    examples["_state"][i], instructions, kind, options, option_order=order
                )
                encoded = encode_decision_prompt(self.tokenizer, rendered, force_bos=force_bos)
                if len(encoded.input_ids) > self.data_args.cutoff_len:
                    num_too_long += 1
                    continue

                order = order if order is not None else list(range(num_options))
                model_inputs["input_ids"].append(encoded.input_ids)
                model_inputs["attention_mask"].append([1] * len(encoded.input_ids))
                model_inputs["opt_idx"].append(encoded.opt_idx)
                model_inputs["answer_idx"].append(encoded.answer_idx)
                model_inputs["label"].append(order.index(label))
                model_inputs["label_dist"].append(self._label_dist(kind, label, order))
                model_inputs["option_order"].append(order)
                model_inputs["kind"].append(kind)
                model_inputs["task"].append(examples["_task"][i])
                model_inputs["weight"].append(examples["_weight"][i])
                model_inputs["source"].append(examples["_source"][i])
                model_inputs["row_index"].append(examples["_row_index"][i])
                model_inputs["augmentation"].append(augmentation)

        if num_too_long:
            logger.warning_rank0(
                f"Dropped {num_too_long} decision prompts longer than cutoff_len ({self.data_args.cutoff_len})."
            )

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        input_ids = example["input_ids"]
        print("input_ids:\n{}".format(input_ids))
        print("inputs:\n{}".format(self.tokenizer.decode(input_ids, skip_special_tokens=False)))
        print("option tokens (opt_idx):")
        for k, position in enumerate(example["opt_idx"]):
            print(f"  {k}: {position} {self.tokenizer.decode([input_ids[position]])!r}")
        answer_idx = example["answer_idx"]
        print(f"answer token (answer_idx): {answer_idx} {self.tokenizer.decode([input_ids[answer_idx]])!r}")
        print(f"label: {example['label']}  label_dist: {example['label_dist']}")
        print(f"option_order: {example['option_order']}")
        print(f"kind: {example['kind']}  task: {example['task']}  augmentation: {example['augmentation']}")
