import re

_SENTENCE_END_RE = re.compile(r'[.!?]["\')]?\s+')


def mention_context(text: str, start: int, end: int) -> tuple:
    """Full sentence containing the mention, split into (before, mention, after) so
    the mention can be bolded inline without cutting off the rest of the sentence."""
    sentence_start = 0
    for m in _SENTENCE_END_RE.finditer(text, 0, start):
        sentence_start = m.end()

    end_match = _SENTENCE_END_RE.search(text, end)
    sentence_end = end_match.end() if end_match else len(text)

    before = text[sentence_start:start].lstrip()
    after = text[end:sentence_end].rstrip()
    return before, text[start:end], after
