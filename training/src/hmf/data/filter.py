import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import Counter
from typing import TYPE_CHECKING, Any

from .data_utils import Role
from ..extras import logging

if TYPE_CHECKING:
    from ..hparams import DataArguments
    from .parser import DatasetAttr


logger = logging.get_logger(__name__)


def parse_filter_spec(filter_spec: str) -> tuple[str, dict[str, int]]:
    """Parse filter specification string.

    Example: 'repetition_ngram10_repeat3_window1024'
    Returns: ('repetition', {'ngram': 10, 'repeat': 3, 'window': 1024})
    """
    parts = filter_spec.split("_")
    pattern = r"^([a-z]+)(\d+)$"

    name_parts = []
    params = {}

    for part in parts:
        match = re.match(pattern, part)
        if match:
            key, value = match.groups()
            params[key] = int(value)
        else:
            name_parts.append(part)

    filter_name = "_".join(name_parts)
    return filter_name, params


@dataclass
class DataFilter(ABC):
    """Base class for dataset filters."""

    dataset_attr: "DatasetAttr"
    data_args: "DataArguments"
    params: dict[str, int] = field(default_factory=dict)

    @abstractmethod
    def __call__(self, aligned_messages: list[dict[str, Any]]) -> bool:
        """
        Filter aligned messages.

        Returns:
            bool: True if the data should be filtered out, False to keep.
        """
        ...

@dataclass
class RepetitionDataFilter(DataFilter):
    """Filter that removes examples with excessive repetition in assistant content.

    Uses word-level n-grams (ignores punctuation/formatting naturally) plus
    a simple check for long runs of identical alphanumeric characters.

    Params:
        ngram: word n-gram size (default: 3)
        repeat: max allowed repeats (default: 5)
        maxrun: max consecutive identical alphanumeric chars, 0 = off (default: 50)

    Usage: --filter repetition_ngram3_repeat5_maxrun50
    """

    def __call__(self, aligned_messages: list[dict[str, Any]]) -> bool:
        ngram_size = self.params.get("ngram", 3)
        max_repeats = self.params.get("repeat", 5)
        max_char_run = self.params.get("maxrun", 50)
        window_size = self.params.get("window", 0)

        for message in aligned_messages:
            if message["role"] in (Role.ASSISTANT.value, Role.FUNCTION.value):
                content = message["content"]
                
                if max_char_run > 0 and self._has_char_run(content, max_char_run):
                    return True
                
                if self._has_word_repetition(content, ngram_size, max_repeats, window_size):
                    return True
        return False

    def _has_char_run(self, text: str, max_run: int) -> bool:
        """Check for consecutive identical alphanumeric characters."""
        count = 1
        for i in range(1, len(text)):
            if text[i] == text[i - 1] and text[i].isalnum():
                count += 1
                if count > max_run:
                    return True
            else:
                count = 1
        return False

    def _has_word_repetition(self, text: str, ngram_size: int, max_repeats: int, window_size: int = 0) -> bool:
        """Check for repeated word n-grams within sliding window."""
        words = text.split()
        if len(words) < ngram_size:
            return False
        
        ngrams = [tuple(words[i:i + ngram_size]) for i in range(len(words) - ngram_size + 1)]
        
        # No window - check globally
        if window_size <= 0 or window_size >= len(ngrams):
            counts: dict[tuple[str, ...], int] = {}
            for ngram in ngrams:
                count = counts.get(ngram, 0) + 1
                if count > max_repeats:
                    return True
                counts[ngram] = count
            return False
        
        # Sliding window
        window = Counter(ngrams[:window_size])
        if max(window.values()) > max_repeats:
            return True
        
        for i in range(window_size, len(ngrams)):
            old = ngrams[i - window_size]
            window[old] -= 1
            if window[old] == 0:
                del window[old]
            
            new = ngrams[i]
            window[new] += 1
            if window[new] > max_repeats:
                return True
        
        return False

    def find_offending_ngrams(
        self, aligned_messages: list[dict[str, Any]], max_results: int = 5, context_chars: int = 100
    ) -> list[dict[str, Any]]:
        """Find offending patterns for debugging."""
        ngram_size = self.params.get("ngram", 3)
        max_repeats = self.params.get("repeat", 5)
        max_char_run = self.params.get("maxrun", 50)

        results = []

        for message in aligned_messages:
            if message["role"] not in (Role.ASSISTANT.value, Role.FUNCTION.value):
                continue

            content = message["content"]

            # Check char runs
            if max_char_run > 0:
                for char, count, pos in self._find_char_runs(content, max_char_run):
                    start = max(0, pos - context_chars)
                    end = min(len(content), pos + count + context_chars)
                    results.append({
                        "type": "char_run",
                        "ngram": f"{char!r} x {count}",
                        "count": count,
                        "position": pos,
                        "context_before": content[start:pos],
                        "context_after": content[pos + count:end],
                    })
                    if len(results) >= max_results:
                        return results

            # Check word ngrams
            for ngram, count, pos in self._find_word_repetitions(content, ngram_size, max_repeats):
                ngram_str = " ".join(ngram)
                start = max(0, pos - context_chars)
                end = min(len(content), pos + len(ngram_str) + context_chars)
                results.append({
                    "type": "word_ngram",
                    "ngram": ngram_str,
                    "count": count,
                    "position": pos,
                    "context_before": content[start:pos],
                    "context_after": content[pos + len(ngram_str):end],
                })
                if len(results) >= max_results:
                    return results

        return results

    def _find_char_runs(self, text: str, max_run: int) -> list[tuple[str, int, int]]:
        """Find alphanumeric char runs exceeding max_run."""
        results = []
        count = 1
        start = 0

        for i in range(1, len(text)):
            if text[i] == text[i - 1] and text[i].isalnum():
                count += 1
            else:
                if count > max_run:
                    results.append((text[i - 1], count, start))
                count = 1
                start = i

        if count > max_run and text and text[-1].isalnum():
            results.append((text[-1], count, start))

        return results

    def _find_word_repetitions(
        self, text: str, ngram_size: int, max_repeats: int
    ) -> list[tuple[tuple[str, ...], int, int]]:
        """Find word n-grams exceeding max_repeats."""
        words = text.split()
        if len(words) < ngram_size:
            return []

        ngram_counts: dict[tuple[str, ...], int] = {}
        ngram_first_pos: dict[tuple[str, ...], int] = {}
        
        # Track character position
        word_positions = []
        pos = 0
        for word in words:
            idx = text.find(word, pos)
            word_positions.append(idx)
            pos = idx + len(word)

        for i in range(len(words) - ngram_size + 1):
            ngram = tuple(words[i:i + ngram_size])
            if ngram not in ngram_counts:
                ngram_counts[ngram] = 0
                ngram_first_pos[ngram] = word_positions[i]
            ngram_counts[ngram] += 1

        results = [
            (ngram, count, ngram_first_pos[ngram])
            for ngram, count in ngram_counts.items()
            if count > max_repeats
        ]
        results.sort(key=lambda x: -x[1])
        return results
     

DATASET_FILTERS: dict[str, type[DataFilter]] = {
    "repetition": RepetitionDataFilter,
}


def register_dataset_filter(name: str, dataset_filter: type[DataFilter]) -> None:
    """Register a new dataset filter."""
    if name in DATASET_FILTERS:
        raise ValueError(f"Dataset filter {name} already exists.")
    DATASET_FILTERS[name] = dataset_filter


def get_dataset_filter(
    filter_spec: str, dataset_attr: "DatasetAttr", data_args: "DataArguments"
) -> DataFilter:
    """Get a dataset filter instance from a filter specification string."""
    filter_name, params = parse_filter_spec(filter_spec)

    if filter_name not in DATASET_FILTERS:
        raise ValueError(
            f"Dataset filter '{filter_name}' not found. Available: {list(DATASET_FILTERS.keys())}"
        )

    return DATASET_FILTERS[filter_name](dataset_attr, data_args, params)