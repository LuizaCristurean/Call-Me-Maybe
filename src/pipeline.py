"""Turn a request into a function call using constrained decoding."""

from __future__ import annotations
import json
from llm_sdk import Small_LLM_Model
from .constrained_decoder import ConstrainedDecoder
from .models import (
    FunctionCall,
    FunctionsDefinition,
    JSONScalar,
    OutputEntries,
    OutputEntry,
    ParamType,
    TestPrompts,
)
from .prompt import build_prompt
from .vocab import load_vocab


def clean_string(value: str, request: str) -> str:
    """Drop a leading space that is only a tokenizer artifact.

    Tokens carry their own leading space (for example ' *'). If the
    value starts with a space that never appears in the request, the
    space was not asked for.
    """
    if value.startswith(" ") and value.strip() and value not in request:
        return value.lstrip()
    return value


def write_value(
    decoder: ConstrainedDecoder, kind: ParamType, more: bool, request: str
) -> JSONScalar:
    """Write one parameter value of the given type and return it."""
    if kind is ParamType.NUMBER:
        return float(decoder.number(more))
    if kind is ParamType.STRING:
        decoder.force(' "')
        text = decoder.string()
        decoder.force('"')
        value: str = json.loads(f'"{text}"')
        return clean_string(value, request)
    return decoder.choose([" true", " false"]).strip() == "true"


def generate_call(
    decoder: ConstrainedDecoder, functions: FunctionsDefinition, request: str
) -> FunctionCall:
    """Write one function call: the name, then each parameter."""
    by_name = {function.name: function for function in functions.root}
    decoder.force('{"name": "')
    name = decoder.choose([key + '"' for key in by_name])[:-1]
    params = list(by_name[name].parameters.items())
    values: dict[str, JSONScalar] = {}
    lead = ', "parameters": {'
    for index, (param, spec) in enumerate(params):
        decoder.force(f'{lead}"{param}":')
        lead = ", "
        more = index < len(params) - 1
        values[param] = write_value(decoder, spec.type, more, request)
    if not params:
        decoder.force(lead)
    decoder.force("}}")
    return FunctionCall(name=name, parameters=values)


def run_pipeline(
    model: Small_LLM_Model,
    functions: FunctionsDefinition,
    prompts: TestPrompts,
) -> OutputEntries:
    """Turn every test prompt into a function call."""
    vocab = load_vocab(model)
    entries: list[OutputEntry] = []
    for item in prompts.root:
        prompt = build_prompt(functions, item.prompt)
        decoder = ConstrainedDecoder(model, vocab, prompt)
        call = generate_call(decoder, functions, item.prompt)
        entries.append(OutputEntry.from_call(item.prompt, call))
        print(f"[{len(entries)}/{len(prompts.root)}] {item.prompt}")
    return OutputEntries(root=entries)
