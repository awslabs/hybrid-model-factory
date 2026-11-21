"""
Tests for a decision model's saved prompt settings (decision_prompt.json, see eval/decision/predictor.py), which
check the values a predictor uses and the files hmf export takes along into a merged model. No model is loaded.
"""

import json

import pytest

from hmf.eval.decision import predictor
from hmf.eval.decision.predictor import (
    AUTO,
    CALIBRATION_FILE,
    DEFAULT_PROMPT,
    PROMPT_FILE,
    resolve_prompt_settings,
    save_prompt_settings,
)
from hmf.train.tuner import _copy_decision_files


def _resolve(model, adapter=None, **given):
    settings = {"prompt_format": AUTO, "force_bos": AUTO, "max_len": AUTO, "option_labels": AUTO}
    settings.update(given)
    return resolve_prompt_settings(str(model), str(adapter) if adapter else None, **settings)


def test_a_model_without_the_file_gets_the_defaults(tmp_path):
    assert _resolve(tmp_path) == DEFAULT_PROMPT


def test_saved_settings_are_used(tmp_path):
    save_prompt_settings(str(tmp_path), "read_twice", True, 8192)

    assert _resolve(tmp_path) == {
        "prompt_format": "read_twice", "force_bos": True, "max_len": 8192, "option_labels": "numbers"
    }


def test_adapter_settings_come_from_the_adapter_folder(tmp_path):
    model, adapter = tmp_path / "model", tmp_path / "adapter"
    model.mkdir()
    adapter.mkdir()
    save_prompt_settings(str(model), "read_twice", True, 8192)

    # The model's file describes the model alone, so an adapter without one gets the defaults
    assert _resolve(model, adapter) == DEFAULT_PROMPT
    save_prompt_settings(str(adapter), "question_first", False, 2048)
    assert _resolve(model, adapter)["prompt_format"] == "question_first"


def test_a_value_given_is_used_and_warned_about_if_it_differs(tmp_path, monkeypatch):
    save_prompt_settings(str(tmp_path), "read_twice", False, 8192)
    warnings = []
    monkeypatch.setattr(predictor.logger, "warning_rank0", warnings.append)

    settings = _resolve(tmp_path, prompt_format="strands", max_len=None)

    assert settings == {"prompt_format": "strands", "force_bos": False, "max_len": None, "option_labels": "numbers"}
    # The prompt format and the window differ from the file's, but the BOS setting does not
    assert len(warnings) == 2
    assert "prompt_format='strands' differs from 'read_twice'" in warnings[0]


def test_option_labels_come_from_the_file_else_from_the_head(tmp_path):
    # A file saved before option labels existed: its model's options are numbered
    (tmp_path / PROMPT_FILE).write_text(json.dumps({"prompt_format": "strands", "force_bos": False, "max_len": 4096}))
    assert _resolve(tmp_path)["option_labels"] == "numbers"

    # A model with a label head and no file yet (as hmf decision-init leaves it) is asked with letters
    model = tmp_path / "label_model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "hmf_decision", "decision_option_labels": "letters"}))
    assert _resolve(model)["option_labels"] == "letters"
    save_prompt_settings(str(model), "strands", False, 4096, "letters")
    assert json.loads((model / PROMPT_FILE).read_text())["option_labels"] == "letters"
    assert _resolve(model)["option_labels"] == "letters"


def _files(folder):
    save_prompt_settings(str(folder), "read_twice", False, 8192)
    (folder / CALIBRATION_FILE).write_text(json.dumps({"temperature": 1.5}))


@pytest.mark.parametrize("merged", ["one adapter", "no adapter", "two adapters"])
def test_export_takes_the_files_of_the_exported_model(tmp_path, merged):
    model, export = tmp_path / "model", tmp_path / "export"
    adapters = [tmp_path / "adapter1", tmp_path / "adapter2"]
    for folder in [model, export] + adapters:
        folder.mkdir()
    (model / PROMPT_FILE).write_text(json.dumps({"prompt_format": "strands", "force_bos": False, "max_len": 4096}))
    for adapter in adapters:
        _files(adapter)

    if merged == "one adapter":
        _copy_decision_files(str(model), [str(adapters[0])], str(export))
        assert json.loads((export / PROMPT_FILE).read_text())["prompt_format"] == "read_twice"
        assert (export / CALIBRATION_FILE).exists()
    elif merged == "no adapter":
        # Exporting the model alone (e.g., in another dtype) keeps its own files
        _copy_decision_files(str(model), None, str(export))
        assert json.loads((export / PROMPT_FILE).read_text())["prompt_format"] == "strands"
        assert not (export / CALIBRATION_FILE).exists()
    else:
        # Neither adapter's files describe a merge of both
        _copy_decision_files(str(model), [str(a) for a in adapters], str(export))
        assert not (export / PROMPT_FILE).exists() and not (export / CALIBRATION_FILE).exists()
