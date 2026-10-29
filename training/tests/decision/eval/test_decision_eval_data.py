"""
Tests for reading decision data files for evaluation, and for the split of a file into a calibration
half and a test half (hmf/eval/decision/data.py). That the halves are exactly Strands Decider's is
tested in tests/decision/test_strands_decider_reproduction.py.
"""

import json
import random

import pytest

from hmf.eval.decision.data import partition_rows, read_rows, sample_rows


ROWS = [{"i": i} for i in range(20)]
CALIB = [0, 3, 4, 5, 9, 10, 11, 12, 13, 15]  # the calibration half of ROWS (seed 0)


def test_the_halves_are_disjoint_and_cover_the_file():
    calib = [row["i"] for row in partition_rows(ROWS, "calib")]
    test = [row["i"] for row in partition_rows(ROWS, "test")]
    assert sorted(calib + test) == list(range(20)) and not set(calib) & set(test)
    # The half depends on the position, not the content, so duplicated rows still split the same way
    assert len(partition_rows([{"same": True}] * 20, "calib")) == len(CALIB)
    # Another seed gives other halves
    assert [row["i"] for row in partition_rows(ROWS, "calib", seed=1)] != CALIB


def test_unknown_split():
    with pytest.raises(ValueError, match="split must be one of"):
        partition_rows(ROWS, "train")


def test_sampling():
    assert sample_rows(ROWS, 0) == ROWS and sample_rows(ROWS, 50) == ROWS  # all rows
    # A fixed shuffle, then the first rows (Strands Decider's sampling)
    expected = list(ROWS)
    random.Random(0).shuffle(expected)
    assert sample_rows(ROWS, 5) == expected[:5]
    assert sample_rows(ROWS, 5) == sample_rows(ROWS, 5)  # deterministic
    assert sample_rows(ROWS, 5, seed=1) != sample_rows(ROWS, 5)


def test_read_rows(tmp_path):
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text(json.dumps({"a": 1}) + "\n\n" + json.dumps({"a": 2}) + "\n")  # an empty line is not a row
    assert read_rows(str(jsonl)) == [{"a": 1}, {"a": 2}]
    as_list = tmp_path / "rows.json"
    as_list.write_text(json.dumps([{"a": 1}, {"a": 2}]))
    assert read_rows(str(as_list)) == [{"a": 1}, {"a": 2}]
