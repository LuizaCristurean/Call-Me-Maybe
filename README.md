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

Supported parameter types: `number`, `integer`, `string` and `boolean`.

## Instructions

Requirements: Python 3.10 or later and [uv](https://docs.astral.sh/uv/). The
first run downloads the model (about 1.2 GB) into the Hugging Face cache.

```bash
make install        # create .venv and install the dependencies (uv sync)
make run            # run on the default input files
make test           # run the unit tests (pytest)
make lint           # flake8 + mypy
make lint-strict    # flake8 + mypy --strict
make debug          # run under pdb
make clean          # remove caches
```

The program is started with:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

All three options are optional (the values above are the defaults). A fourth
option, `--verbose`, prints every decoding step (see the bonus section).

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
- pytest documentation: https://docs.pytest.org
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
   - *Number and integer*: the text must stay a prefix of a valid JSON number
     (optional minus, digits, and for numbers an optional dot and digits). The
     separator token (`,` or `}`) becomes allowed only when the number written
     so far is complete, so a number can never end as `-` or `3.`.
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
  `vocab.json` uses an internal byte encoding (for example `G` with a dot for a
  space), so decoding every id is simpler and safer than converting it by hand.
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
- **pydantic everywhere.** The input and output files are read and written
  through pydantic models (`models.py`), and the decoder classes
  (`ConstrainedDecoder`, `StringTokens`) are pydantic models too, as the
  subject requires. Wrong inputs are detected with a clear message.
- **Precomputed groups.** The classification of the vocabulary for strings is
  computed once and shared by all prompts.

Known limitations: tokens made of an incomplete UTF-8 byte sequence are
excluded, so rare symbols such as some emoji can be lost or shortened. Only
one function is chosen per prompt. Only the types `number`, `integer`,
`string` and `boolean` are supported; other types give a schema error when the
definitions are loaded. The model is small, so on ambiguous requests it can
choose a wrong value; the constraints guarantee the form of the output, not
that every value is the one the user meant.

## Performance analysis

- **Speed.** The 11 default prompts are processed in about 35 seconds in total
  on a MacBook Air (Apple GPU through the `mps` backend), including loading the
  model. The limit for the project is 5 minutes.
- **Cost per token.** Each generated token needs one forward pass of the
  model. Fixed text needs none, which is why the program writes the structure
  itself instead of asking the model to generate it. A call with two numeric
  parameters needs only a few dozen model evaluations.
- **Precomputation.** The vocabulary texts and the string token groups are
  computed once per run, not once per prompt. Most of the remaining time is
  spent in the model, not in the rules.
- **Reliability.** The output is valid JSON for every prompt that succeeds, and
  the function name always exists in the definitions. On the 11 default
  prompts all function names and values are correct.

## Challenges faced

- **Tokens are not characters.** The first experiments in a Python shell
  showed that a word and its leading space are one token, that each digit is
  its own token, and that the vocabulary file stores spaces in an internal
  form. This decided the whole design.
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
- A pytest suite (`make test`, 13 tests) covers the token rules on tiny
  made-up vocabularies, where the expected result can be written down in
  advance, and the reading, validation and writing of the files, including
  the error cases.
- Extra files in `tests/` cover cases that the default files do not:
  negative and decimal numbers, large numbers, booleans, an integer, a
  function without parameters, quotes and non-ASCII characters in the text,
  an empty prompt, and an empty list of functions.
- `make lint-strict` (flake8 and mypy in strict mode) must pass.

Commands:

```bash
make test
make run ARGS="--functions_definition tests/functions.json --input tests/prompts.json --output data/output/extra_results.json"
make run ARGS="--functions_definition tests/empty_functions.json --input tests/prompts.json --output data/output/empty_results.json"
make run ARGS="--input tests/missing.json"
```

## Bonus features

Each bonus can be shown with one command.

**1. Test suite.** 13 unit tests, no model needed.

```bash
make test
```

Expected: `13 passed`.

**2. Visualization of the generation process.** `--verbose` prints every
step: the text written by the program (`forced`), how many tokens were
allowed, the token chosen, and a note when the model preferred a token that
the constraints blocked.

```bash
make run ARGS="--verbose --functions_definition tests/functions.json --input tests/one.json --output data/output/one_result.json"
```

In the output, look for lines like `(model preferred '",', blocked)`. They show
constrained decoding at work.

**3. Performance optimization (caching).** The vocabulary and the string
token groups are built once in `run_pipeline` and shared by all prompts.

```bash
grep -n "load_vocab\|StringTokens.from_vocab" src/pipeline.py
```

**4. Error recovery per prompt.** A prompt that fails is reported and skipped,
and the run goes on. With an empty list of functions every prompt fails, and
the program still finishes and writes a valid empty result.

```bash
make run ARGS="--functions_definition tests/empty_functions.json --input tests/prompts.json --output data/output/empty_results.json"
cat data/output/empty_results.json
```

**5. How encode and decode integrate with constrained decoding.** `encode`
turns the prompt and the fixed text into token ids; `decode` gives the text of
every token, which is what the allowed-token rules use. This prints the ids of
a sentence and the text of each id:

```bash
uv run python -c "from llm_sdk import Small_LLM_Model as M; m = M(); ids = m.encode('What is the sum of 2 and 3?')[0].tolist(); print(ids); print([m.decode([i]) for i in ids])"
```

The `--verbose` output above shows the other half: the fixed text is encoded
(`forced`) and every token chosen by the model is shown as decoded text.

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