"""
Prompt template + single-call extraction logic for LLM-based coreference
annotation of one abstract.

Produces three things per abstract, matched against fastcoref's actual
predicted clusters downstream (see utils/gold_compare.py):
  - clusters: real multi-mention (2+) coreference chains — matches what
    fastcoref itself outputs, so the two are directly comparable.
  - singleton_mentions: substantive entities mentioned exactly once. These
    exist purely as a negative signal — if fastcoref groups one of these
    together with anything else, that's a false merge worth flagging.
  - self_reference_mentions: every occurrence of author self-reference
    ("we"/"our"/"us"/"I"/"my") — trivial, always-correct coreference on its
    own, deliberately kept out of `clusters`, but tracked here so a fastcoref
    cluster that mixes one of these into a real entity chain (a genuine
    error) can also be flagged.

Mentions in all three lists are verbatim substrings of the abstract, so they
can be matched/located the same way the dashboard's mention spans are.

Uses the Claude Messages API with Pydantic-backed structured outputs
(`client.messages.parse`) so the response is guaranteed to match the schema
below — no manual JSON-parsing/retry-on-malformed-output logic needed.
"""

from __future__ import annotations

from typing import List

import anthropic
from pydantic import BaseModel, Field

# Model is explicit and overridable (see run_batch.py --model) rather than
# hardcoded, since the right cost/quality tradeoff for a 6000+ abstract batch
# job is a call the user/team should make, not one this script makes for them.
DEFAULT_MODEL = "claude-opus-5"

SYSTEM_PROMPT = """\
You are annotating scientific-paper abstracts for coreference resolution, in \
a format that will be directly compared against an automatic coreference \
model's (fastcoref) output. Precision on the schema matters as much as the \
linguistic judgment.

Task: read the title and abstract given by the user, and produce THREE \
things:

1. "clusters" — every coreference CLUSTER: a group of 2 or more text spans \
("mentions") that all refer to the same real-world entity (a proposed \
method/system, a dataset, a model, a metric, or similar) — exactly the kind \
of chain a coreference resolution model is scored on.

2. "singleton_mentions" — every OTHER substantive entity (a method, \
dataset, metric, component, concept, etc.) that is mentioned EXACTLY ONCE in \
the whole abstract, and is not part of any cluster above and not author \
self-reference. List the verbatim mention text for each. Purpose: these \
exist purely as a negative check — a correct coreference system should \
NEVER link one of these to anything else, since by definition nothing else \
in the text refers to it. If an automatic model groups one of these \
together with other mentions, that's a false merge. Use judgment: you don't \
need to list every single noun phrase in the abstract, just entities \
specific/nameable enough that a coreference model might plausibly (and \
wrongly) try to link them to something. Skip generic/function words.

3. "self_reference_mentions" — every verbatim occurrence of author \
self-reference ("we"/"us"/"our"/"I"/"my"/"the authors"), in the order they \
appear, however many times it recurs. Do NOT put these into "clusters" \
even though they corefer with each other — track them here instead. \
Purpose: same as singletons — if an automatic model links one of these into \
a real entity's cluster instead of keeping author self-reference separate, \
that's an error worth flagging.

Rules for "clusters", in priority order:

1. ONLY multi-mention clusters. An entity mentioned exactly once belongs in \
"singleton_mentions" instead (or is skipped, if it's too generic to be \
worth tracking) — never invent a second mention to pad a cluster to size 2.

2. EXCLUDE author self-reference from "clusters" — track it in \
"self_reference_mentions" instead (see above), no matter how many times it \
recurs.

3. Mentions must be EXACT, VERBATIM substrings of the abstract text — same \
casing, same punctuation, same whitespace as they appear. Do not paraphrase, \
normalize, truncate, or correct typos in a mention. If you cannot copy a \
span exactly as it appears, do not include it. This applies to all three \
output lists, not just "clusters".

4. List each cluster's mentions in the order they appear in the text \
(left to right). Same for "self_reference_mentions".

5. "label" is a short human-readable name for what the cluster refers to — \
normally its longest or most identifying mention (e.g. a method's full name \
if an acronym recurs, or "the model" if that's the clearest handle).

Typical clusters in this corpus (domain adaptation / ML abstracts): the \
paper's named method/system and its later short-form, acronym, or pronoun \
references ("it", "which", "the proposed X"); a dataset or benchmark name \
referenced more than once; "the model"/"the framework"/"our approach" \
referring back to one named system (but NOT the bare "we/our/us" author \
chain itself — rule 2); pronoun chains ("it"/"its"/"they"/"which") pointing \
to one clear antecedent.

If no qualifying cluster remains after the rules above, return an empty \
list for "clusters" — do not force a result. Same for the other two lists \
if nothing qualifies.\
"""


def build_user_content(title: str, abstract: str) -> str:
    return f"TITLE: {title}\n\nABSTRACT: {abstract}"


class Cluster(BaseModel):
    label: str = Field(description="Short human-readable name for the entity this cluster refers to.")
    mentions: List[str] = Field(
        description="Verbatim substrings of the abstract, in order of appearance, all referring to the same entity."
    )


class AbstractClusters(BaseModel):
    clusters: List[Cluster] = Field(
        description="Every multi-mention coreference cluster found, excluding author self-reference."
    )
    singleton_mentions: List[str] = Field(
        default_factory=list,
        description=(
            "Substantive entities mentioned exactly once — a negative check: none of these should "
            "ever appear grouped with another mention in a correct coreference output."
        ),
    )
    self_reference_mentions: List[str] = Field(
        default_factory=list,
        description=(
            "Every verbatim occurrence of author self-reference (we/us/our/I/my/the authors), in "
            "order of appearance — kept out of 'clusters' but tracked so a model mixing one of these "
            "into a real entity's cluster can be flagged."
        ),
    )


def extract_clusters(
    client: anthropic.Anthropic,
    title: str,
    abstract: str,
    *,
    model: str = DEFAULT_MODEL,
):
    """One extraction call for one abstract. Raises on API error — caller decides retry policy.

    Returns the raw parsed response (not just `.parsed_output`) so the caller can also
    read `.usage` for cost tracking across a batch."""
    response = client.messages.parse(
        model=model,
        max_tokens=4096,
        # The system prompt is byte-identical across every call in the batch —
        # caching it means every abstract after the first pays ~0.1x for it
        # instead of full price. See shared/prompt-caching.md in the claude-api skill.
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": build_user_content(title, abstract)}],
        output_format=AbstractClusters,
    )
    return response
