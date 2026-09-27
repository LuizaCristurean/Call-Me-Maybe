"""Reading the two input JSON files, writing the output JSON file.

Each function here does exactly one thing: turn bytes on disk into a
validated pydantic object, or turn a validated pydantic object into
bytes on disk. No decoding logic and no business logic live here --
that belongs to schema_builder.py, constrained_decoder.py and
pipeline.py. If a file is malformed, the pydantic ValidationError
raised by model_validate_json is left to propagate -- the caller
decides how to report it, we don't swallow it here.
"""

from __future__ import annotations
from pathlib import Path
from .models import FunctionsDefinition, OutputEntries, TestPrompts


def load_functions_definition(path: Path) -> FunctionsDefinition:
    """Read and validate functions_definition.json."""
    return FunctionsDefinition.model_validate_json(path.read_text())


def load_test_prompts(path: Path) -> TestPrompts:
    """Read and validate function_calling_tests.json."""
    return TestPrompts.model_validate_json(path.read_text())


def write_results(path: Path, entries: OutputEntries) -> None:
    """Write the final results as pretty-printed JSON.

    Creates the parent directory (data/output/, typically) if it
    doesn't exist yet, so a fresh checkout can run without manual setup.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(entries.model_dump_json(indent=2))
