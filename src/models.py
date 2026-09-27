"""Pydantic data models shared across the project.

These describe the three JSON shapes we read or produce:
  * a function's schema, as it appears in ``functions_definition.json``;
  * a single test prompt, as it appears in ``function_calling_tests.json``;
  * one entry of our own output, ``function_calling_results.json``.

Every field is validated on construction (that's the whole point of using
pydantic here instead of raw dicts) -- a malformed functions_definition.json
fails loudly and early, with a clear message, instead of crashing deep
inside the decoder later.
"""

from __future__ import annotations

from enum import Enum
from typing import Union

from pydantic import BaseModel, Field, RootModel


# The JSON scalar types our functions can take as parameters / return.
# Kept as a closed enum (rather than a bare str) so an unknown type in
# functions_definition.json ("array", say) fails validation immediately
# with a clear error, instead of silently being treated as valid.
class ParamType(str, Enum):
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"


# A Python value of one of the types above -- what ends up in the
# "parameters" object of a single output entry.
JSONScalar = Union[float, str, bool]


class ParameterSpec(BaseModel):
    """One parameter's type, as declared inside a function's "parameters"."""

    type: ParamType


class ReturnSpec(BaseModel):
    """A function's declared return type (informational -- we never call
    the function ourselves, we only report what *should* be called)."""

    type: ParamType


class FunctionDef(BaseModel):
    """One entry of functions_definition.json: a function the LLM may call."""

    name: str
    description: str
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: ReturnSpec


class FunctionsDefinition(RootModel[list[FunctionDef]]):
    """The whole functions_definition.json file: a JSON array of FunctionDef.
    """

    pass


class TestPrompt(BaseModel):
    """One entry of function_calling_tests.json."""

    prompt: str


class TestPrompts(RootModel[list[TestPrompt]]):
    """The whole function_calling_tests.json file: a JSON array of TestPrompt.
    """

    pass


class FunctionCall(BaseModel):
    """What constrained decoding produces for one prompt: which function,
    with which arguments -- before we know which prompt it answers."""

    name: str
    parameters: dict[str, JSONScalar]


class OutputEntry(BaseModel):
    """One entry of function_calling_results.json -- exactly the three keys
    the subject requires, nothing more, nothing less."""

    prompt: str
    name: str
    parameters: dict[str, JSONScalar]

    @classmethod
    def from_call(cls, prompt: str, call: FunctionCall) -> "OutputEntry":
        """Build an OutputEntry by pairing a prompt with the call
        decoded for it."""
        return cls(prompt=prompt, name=call.name, parameters=call.parameters)


class OutputEntries(RootModel[list[OutputEntry]]):
    """The whole function_calling_results.json file: a JSON array of
    OutputEntry."""

    pass
