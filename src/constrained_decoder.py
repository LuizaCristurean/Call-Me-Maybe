"""Constrained decoding: pick tokens the model is allowed to emit."""

from __future__ import annotations
from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ConfigDict
import re


ESCAPE_CHARS = '"\\/bfnrt'
MAX_STRING_TOKENS = 120
MAX_NUMBER_TOKENS = 30


def select_token(logits: list[float], allowed: set[int]) -> int:
    """Return the allowed token id with the highest logit score.

    This is the same as setting every disallowed logit to -inf and
    taking the argmax, but it only looks at the allowed ids.
    """
    if not allowed:
        raise ValueError("no token is allowed at this step")
    return max(allowed, key=lambda token_id: logits[token_id])


def allowed_for_choices(
    vocab: list[str], typed: str, choices: list[str]
) -> set[int]:
    """Return the token ids that keep `typed` a prefix of some choice."""
    allowed: set[int] = set()
    for token_id, text in enumerate(vocab):
        if not text:
            continue
        candidate = typed + text
        if any(choice.startswith(candidate) for choice in choices):
            allowed.add(token_id)
    return allowed


NUMBER_PREFIX = re.compile(r" ?-?([0-9]+(\.[0-9]*)?)?")
NUMBER_COMPLETE = re.compile(r" ?-?[0-9]+(\.[0-9]+)?")
INTEGER_PREFIX = re.compile(r" ?-?[0-9]*")
INTEGER_COMPLETE = re.compile(r" ?-?[0-9]+")


def can_extend_number(text: str, integer: bool = False) -> bool:
    """Return True if `text` can still grow into a valid JSON number."""
    pattern = INTEGER_PREFIX if integer else NUMBER_PREFIX
    return pattern.fullmatch(text) is not None


def is_complete_number(text: str, integer: bool = False) -> bool:
    """Return True if `text` is already a valid JSON number."""
    pattern = INTEGER_COMPLETE if integer else NUMBER_COMPLETE
    return pattern.fullmatch(text) is not None


def allowed_for_number(
    vocab: list[str],
    typed: str,
    end_ids: set[int],
    integer: bool = False,
) -> set[int]:
    """Return the token ids allowed while writing a JSON number.

    `end_ids` are the tokens that may close the number (for example
    the tokens `,` and `}`); they are allowed only once the number
    written so far is complete.
    """
    allowed: set[int] = set()
    for token_id, text in enumerate(vocab):
        if text and can_extend_number(typed + text, integer):
            allowed.add(token_id)
    if is_complete_number(typed, integer):
        allowed |= end_ids
    return allowed


def scan_string_text(
    text: str, escaped: bool
) -> tuple[int, bool, bool] | None:
    """Check `text` as part of a JSON string value.

    `escaped` says the previous token ended with a lone backslash.
    Returns None if the text cannot be inside a string. Otherwise it
    returns (end, closed, escaped): `end` is the index of the closing
    quote (or the text length), `closed` says a closing quote was
    found, and `escaped` says the text ends with a lone backslash.
    """
    for index, char in enumerate(text):
        if escaped:
            if char not in ESCAPE_CHARS:
                return None
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return index, True, False
        elif char == "\ufffd" or ord(char) < 32:
            return None
    return len(text), False, escaped


class StringTokens(BaseModel):
    """Groups the vocabulary by how a token behaves inside a string."""

    dangling: set[int]
    closers: dict[int, str]
    after_escape: set[int]
    open_ids: set[int]

    @classmethod
    def from_vocab(cls, vocab: list[str]) -> StringTokens:
        """Classify every token of the vocabulary once."""
        plain: set[int] = set()
        dangling: set[int] = set()
        closers: dict[int, str] = {}
        after_escape: set[int] = set()
        for token_id, text in enumerate(vocab):
            if not text:
                continue
            result = scan_string_text(text, False)
            if result is not None:
                end, closed, escaped = result
                if closed:
                    closers[token_id] = text[:end]
                elif escaped:
                    dangling.add(token_id)
                else:
                    plain.add(token_id)
            result = scan_string_text(text, True)
            if result is not None and not result[1] and not result[2]:
                after_escape.add(token_id)
        return cls(
            dangling=dangling,
            closers=closers,
            after_escape=after_escape,
            open_ids=plain | dangling | set(closers),
        )


class ConstrainedDecoder(BaseModel):
    """Writes the output one token at a time, following the rules."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Small_LLM_Model
    vocab: list[str]
    strings: StringTokens
    comma_id: int
    brace_id: int
    ids: list[int]
    generated: str = ""
    verbose: bool = False

    @classmethod
    def create(
        cls,
        model: Small_LLM_Model,
        vocab: list[str],
        prompt: str,
        strings: StringTokens | None = None,
        verbose: bool = False,
    ) -> ConstrainedDecoder:
        """Build a decoder that starts from `prompt`."""
        return cls(
            model=model,
            vocab=vocab,
            strings=strings or StringTokens.from_vocab(vocab),
            comma_id=vocab.index(","),
            brace_id=vocab.index("}"),
            ids=model.encode(prompt)[0].tolist(),
            verbose=verbose,
        )

    def force(self, text: str) -> None:
        """Append fixed text: no choice, so no model scores needed."""
        if self.verbose:
            print(f"  forced {text!r}")
        self.ids += self.model.encode(text)[0].tolist()
        self.generated += text

    def show_step(
        self, logits: list[float], allowed: set[int], token_id: int
    ) -> None:
        """Print one decoding step (used by --verbose)."""
        wanted = max(range(len(logits)), key=lambda i: logits[i])
        line = f"  {len(allowed):>6} allowed -> {self.vocab[token_id]!r}"
        if wanted != token_id:
            line += f"   (model preferred {self.vocab[wanted]!r}, blocked)"
        print(line)

    def best(self, allowed: set[int]) -> int:
        """Return the allowed token the model scores highest."""
        logits = self.model.get_logits_from_input_ids(self.ids)
        token_id = select_token(logits, allowed)
        if self.verbose:
            self.show_step(logits, allowed, token_id)
        return token_id

    def append(self, token_id: int) -> None:
        """Add one token to the text written so far."""
        self.ids.append(token_id)
        self.generated += self.vocab[token_id]

    def pick(self, allowed: set[int]) -> int:
        """Let the model choose among `allowed`, and append the token."""
        token_id = self.best(allowed)
        self.append(token_id)
        return token_id

    def choose(self, choices: list[str]) -> str:
        """Write one of `choices` token by token; return the one written."""
        typed = ""
        while typed not in choices:
            allowed = allowed_for_choices(self.vocab, typed, choices)
            token_id = self.pick(allowed)
            typed += self.vocab[token_id]
        return typed

    def number(self, more: bool, integer: bool = False) -> str:
        """Write a JSON number; stop when the model picks the separator.

        `more` tells whether another parameter follows (`,`) or not (`}`).
        With `integer` the number has no decimal part.
        The separator itself is not written; the caller forces it.
        """
        end_id = self.comma_id if more else self.brace_id
        typed = ""
        for _ in range(MAX_NUMBER_TOKENS):
            allowed = allowed_for_number(
                self.vocab, typed, {end_id}, integer
            )
            token_id = self.best(allowed)
            if token_id == end_id:
                break
            self.append(token_id)
            typed += self.vocab[token_id]
        if not is_complete_number(typed, integer):
            self.force("0")
            typed += "0"
        return typed

    def string(self) -> str:
        """Write the content of a JSON string (still escaped).

        The closing quote is not written; the caller forces it.
        """
        typed = ""
        escaped = False
        for _ in range(MAX_STRING_TOKENS):
            if escaped:
                allowed = self.strings.after_escape
            else:
                allowed = self.strings.open_ids
            token_id = self.best(allowed)
            if not escaped and token_id in self.strings.closers:
                prefix = self.strings.closers[token_id]
                if prefix:
                    self.force(prefix)
                return typed + prefix
            self.append(token_id)
            typed += self.vocab[token_id]
            escaped = not escaped and token_id in self.strings.dangling
        if escaped:
            self.force("\\")
            typed += "\\"
        return typed
