"""Token vocabulary: the text of every token id of the model."""

from __future__ import annotations
from llm_sdk import Small_LLM_Model


def load_vocab(model: Small_LLM_Model) -> list[str]:
    """Return the text of every token; the list index is the token id.

    Ids with no text (special tokens, padding) map to an empty string.
    """
    size = len(model.get_logits_from_input_ids([0]))
    return [model.decode([i]) for i in range(size)]
