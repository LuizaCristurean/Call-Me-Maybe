"""Command-line entry point.

Parses the CLI's arguments and loads the two input files through
io_utils. Owns error handling for anything that can go wrong with
those files -- a missing path, invalid JSON, or data that doesn't
match the expected schema -- all reported as one clear line, never
as a raw Python traceback (the subject requires the program to never
crash on bad input, only report the problem and stop).
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from pydantic import ValidationError
from .io_utils import load_functions_definition, load_test_prompts
from .models import FunctionsDefinition, TestPrompts

DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
DEFAULT_TESTS = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the program's command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="call-me-maybe",
        description=(
            "Translate a natural-language prompt into a "
            "structured function call."
        ),
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS,
        help="Path to functions_definition.json (the available functions).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_TESTS,
        help="Path to function_calling_tests.json (the prompts to process).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write function_calling_results.json.",
    )
    return parser.parse_args(argv)


def load_inputs(
    functions_path: Path, tests_path: Path
) -> tuple[FunctionsDefinition, TestPrompts]:
    """Load and validate both input files, exiting with a
    clear message on failure."""
    try:
        functions = load_functions_definition(functions_path)
    except FileNotFoundError:
        sys.exit(f"error: functions file not found: {functions_path}")
    except ValidationError as exc:
        sys.exit(
            f"error: {functions_path} does not match the "
            f"expected schema:\n{exc}"
            )

    try:
        tests = load_test_prompts(tests_path)
    except FileNotFoundError:
        sys.exit(f"error: prompts file not found: {tests_path}")
    except ValidationError as exc:
        sys.exit(
            f"error: {tests_path} does not match the "
            f"expected schema:\n{exc}"
            )

    return functions, tests


def main(argv: list[str] | None = None) -> None:
    """Program entry point, called from call_me_maybe.py."""
    args = parse_args(argv)
    functions, tests = load_inputs(args.functions_definition, args.input)
    print(
        f"Loaded {len(functions.root)} functions "
        f"and {len(tests.root)} prompts."
        )
