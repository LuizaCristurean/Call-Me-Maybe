"""Build the text prompt that is given to the language model."""

from __future__ import annotations
from .models import FunctionDef, FunctionsDefinition


def describe_function(function: FunctionDef) -> str:
    """Return a one-line description of a function and its parameters."""
    params = ", ".join(
        f"{name} ({spec.type.value})"
        for name, spec in function.parameters.items()
    )
    return f"- {function.name}: {function.description} Parameters: {params}"


def build_prompt(functions: FunctionsDefinition, request: str) -> str:
    """Return the prompt: the available functions, then the request."""
    lines = [describe_function(function) for function in functions.root]
    return (
        "Convert the request into a function call. Answer with JSON: "
        '{"name": <function>, "parameters": {<parameter>: <value>}}\n'
        "Copy numbers and text from the request exactly as written. "
        "For a regex, use a pattern like \\\\d+ (backslashes doubled).\n"
        "Available functions:\n"
        + "\n".join(lines)
        + f"\n\nRequest: {request}\nCall: "
    )
