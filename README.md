*This project has been created as part of the 42 curriculum by lcristur.*

# Call Me Maybe

## Description

Call Me Maybe turns a natural-language request into a structured function
call. Given a list of function definitions and a list of prompts, it asks a
small language model (Qwen3-0.6B) which function to call and with which
arguments, and writes the answers as JSON.

The main point of the project is **constrained decoding**: the model is never
trusted to write valid JSON on its own. At every step the program removes
all the tokens that would break the expected structure, so the output is
always valid JSON that matches the function definitions, even though the
model is small.

For example, the prompt `What is the sum of 2 and 3?` becomes:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

## Instructions

Requirements: Python 3 and [uv](https://docs.astral.sh/uv/). The first run
downloads the model (about 1.2 GB) into the Hugging Face cache.

```bash
make install        # create .venv and install the dependencies (uv sync)
make run            # run on the default input files
make lint           # flake8 + mypy
make lint-strict    # flake8 + mypy --strict
make debug          # run under pdb
make clean          # remove caches
```

Command line options (all optional):

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

The input files are JSON arrays. The output is a JSON array with one entry
per prompt, each with the keys `prompt`, `name` and `parameters`. If the
output folder does not exist, it is created. Missing files, invalid JSON and
files with a wrong structure give a one-line `error:` message instead of a
traceback.

## Resources

- Qwen3-0.6B model card: https://huggingface.co/Qwen/Qwen3-0.6B
- Hugging Face Transformers documentation: https://huggingface.co/docs/transformers
- JSON specification: https://www.json.org
- pydantic documentation: https://docs.pydantic.dev
- uv documentation: https://docs.astral.sh/uv/

**Use of AI.** Claude was used as a tutor: to explain concepts (tokens,
logits, constrained decoding, pydantic, uv), to discuss design options and
to review results. The code was typed and tested by the author, step by step,
and each design decision in this file was checked on real model output.

## Algorithm explanation

A language model does not write text directly. For a given sequence of token
ids it returns one score (logit) for every token in its vocabulary (151,936
positions for Qwen3-0.6B). Normal generation takes the best score, appends
that token and repeats.

Constrained decoding changes one thing: before taking the best score, every
token that would make the output invalid is excluded. Excluding a token is the
same as setting its logit to minus infinity. The model still decides, but only
among tokens that keep the output valid.

The output of one call is built from two kinds of pieces:

1. **Fixed text**, written by the program without asking the model:
   `{"name": "`, `, "parameters": {"a":`, `}}`, and so on. Nothing needs to be
   decided there.
2. **Decision points**, where the model chooses among allowed tokens:
   - *Function name*: the allowed tokens are those that keep the text typed so
     far a prefix of one of the real function names (each followed by a closing
     quote). The model can only end on a name that exists.
   - *Number*: the text must stay a prefix of a valid JSON number (optional
     minus, digits, optional dot and digits). The separator token (`,` or `}`)
     becomes allowed only when the number written so far is complete, so a
     number can never end as `-` or `3.`.
   - *String*: any token is allowed, except those with control characters or an
     invalid escape. A backslash must be followed by a valid escape character.
     Tokens that contain the closing quote (for example `",`) end the string,
     and only the part before the quote is kept.
   - *Boolean*: the model chooses between `true` and `false`.

For each prompt the program builds a text prompt with the function list and
the request, lets the model pick the function name, then goes through the
parameters of that function in the order of its definition and fills each one
according to its type. The values are collected in a Python object, which is
then serialized with a JSON library, so the final file is always well formed.

## Design decisions

- **Token texts come from the tokenizer.** `vocab.py` builds a list where the
  index is the token id and the value is `decode([id])`. The raw
  `vocab.json` uses an internal byte encoding (for example `Ġ` for a space), so
  decoding every id is simpler and safer than converting it by hand.
- **Allowed sets are computed from text.** Every rule works on the text of the
  tokens, not on characters, because one token can hold several characters.
  A token is allowed if the text typed so far plus the token is still valid.
- **The string rule is the JSON string rule.** The model writes strings in
  JSON style (a backslash is written twice), so the program accepts valid JSON
  escapes and converts the text with `json.loads`. A regex such as `\d+` is
  written by the model as `\\d+` and stored as `\d+`.
- **Closing tokens.** The model prefers to close a string with a token such as
  `",` rather than a lone `"`. Measuring the logits showed the lone quote about
  18 points lower, so a string using only the lone quote never ended. Tokens
  with a closing quote are therefore allowed, and the rest of the token is
  discarded.
- **Forced text follows natural tokenization.** Fixed pieces are written in one
  piece (for example `, "b":`) so that their tokens look like text the model
  has seen. Writing the space before a number separately made the model avoid
  the minus sign, because the natural token for a negative number is ` -`.
  The number rule accepts one optional leading space for that reason.
- **Leading space clean-up.** Many tokens carry their own leading space
  (` *`). If a string value starts with a space that does not appear in the
  request, the space is removed. This is the only place where a value is
  changed after generation.
- **Safety limits.** A number can use at most 30 tokens and a string at most
  120. A number that is cut incomplete is completed with a `0`, and a string
  that ends on a lone backslash is completed, so the output stays valid.
- **Errors per prompt.** If one prompt fails, the error goes to `stderr` and
  the program moves on to the next one.
- **Data models.** The input and output files are read and written through
  pydantic models (`models.py`), so wrong inputs are detected with a clear
  message.

Known limitations: tokens made of an incomplete UTF-8 byte sequence are
excluded, so rare symbols such as some emoji can be lost or shortened. Only
one function is chosen per prompt. The model is small, so on ambiguous
requests it can choose a wrong value; the constraints guarantee the form of
the output, not that every value is the one the user meant.

## Performance analysis

- **Speed.** The 11 default prompts are processed in about 30 seconds in total
  on a MacBook Air (Apple GPU through the `mps` backend), including loading the
  model. The limit for the project is 5 minutes.
- **Cost per token.** Each generated token needs one forward pass of the
  model. Fixed text needs none, which is why the program writes the structure
  itself instead of asking the model to generate it. A call with two numeric
  parameters needs only a few dozen model evaluations.
- **Masks are cheap.** The string token groups are computed once per prompt by
  scanning the 150,000 tokens. Number and name rules scan the vocabulary at each
  step, which takes a small fraction of the time of a forward pass.
- **Reliability.** The output is valid JSON for every prompt that succeeds, and
  the function name always exists in the definitions. On the 11 default
  prompts all function names and values are correct.

## Challenges faced

- **Tokens are not characters.** The first experiments in a Python shell
  showed that a word and its leading space are one token, that each digit is
  its own token, and that the vocabulary file stores spaces as `Ġ`. This
  decided the whole design.
- **Strings that never ended.** On a regex prompt the program waited forever.
  Reading the logits showed that the model wanted `",` and `\\`, both of which
  the first version excluded. This led to JSON escapes and closing tokens.
- **Lost minus sign.** `-2.5` became `2.5` because the forced space before the
  number was a token the model rarely sees before a minus. Dropping the forced
  space and accepting it inside the number fixed it.
- **A stray space in a value.** The token ` *` made `replacement` equal to
  ` *`. A prompt example did not help, so the clean-up described above was
  added.
- **Tooling.** The virtual environment inside an iCloud-synced folder made the
  first import of the large libraries take minutes, so the project lives
  outside such folders.

## Testing strategy

- The default input files are run after each change and the results are read
  prompt by prompt.
- Extra files in `tests/` cover cases that the default files do not:
  negative and decimal numbers, large numbers, booleans, a function without
  parameters, quotes and non-ASCII characters in the text, an empty prompt.
- The error paths are tested by hand: a missing file, a file that is not JSON,
  a file with the wrong structure and an empty list of functions (every prompt
  is reported and skipped, and the output is an empty list).
- The small rules (name prefixes, number prefixes, string tokens) were first
  tried on a tiny made-up vocabulary, where the expected result can be written
  down in advance, and then on the real model.
- `make lint-strict` (flake8 and mypy in strict mode) must pass.

## Example usage

```bash
$ make run
[1/11] What is the sum of 2 and 3?
...
Wrote 11 results to data/output/function_calling_results.json
```

Part of `data/output/function_calling_results.json`:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
    "name": "fn_substitute_string_with_regex",
    "parameters": {
      "source_string": "Hello 34 I'm 233 years old",
      "regex": "\\d+",
      "replacement": "NUMBERS"
    }
  }
]
```

Using other files:

```bash
make run ARGS="--functions_definition my_functions.json --input my_prompts.json --output out.json"
```
