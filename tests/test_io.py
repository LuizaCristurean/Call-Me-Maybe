"""Tests for reading the input files and writing the results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.cli import load_inputs
from src.io_utils import (
    load_functions_definition,
    load_test_prompts,
    write_results,
)
from src.models import FunctionCall, OutputEntries, OutputEntry

FUNCTION = (
    '[{"name": "fn_a", "description": "d",'
    ' "parameters": {"x": {"type": "number"}},'
    ' "returns": {"type": "number"}}]'
)


def test_load_functions(tmp_path: Path) -> None:
    path = tmp_path / "functions.json"
    path.write_text(FUNCTION)
    functions = load_functions_definition(path)
    assert functions.root[0].name == "fn_a"
    assert functions.root[0].parameters["x"].type.value == "number"


def test_load_prompts(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text('[{"prompt": "hello"}]')
    assert load_test_prompts(path).root[0].prompt == "hello"


def test_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not json")
    with pytest.raises(ValidationError):
        load_test_prompts(path)


def test_wrong_structure_raises(tmp_path: Path) -> None:
    path = tmp_path / "wrong.json"
    path.write_text('{"a": 1}')
    with pytest.raises(ValidationError):
        load_test_prompts(path)


def test_missing_files_exit(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        load_inputs(tmp_path / "a.json", tmp_path / "b.json")


def test_write_results_format(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "out.json"
    call = FunctionCall(name="fn_a", parameters={"x": 1.0})
    entries = OutputEntries(root=[OutputEntry.from_call("hi", call)])
    write_results(out, entries)
    data = json.loads(out.read_text())
    assert data == [
        {"prompt": "hi", "name": "fn_a", "parameters": {"x": 1.0}}
    ]
