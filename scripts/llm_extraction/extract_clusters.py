"""
Prompt template + single-call extraction logic for LLM-based coreference
cluster annotation of one abstract.

This mirrors the manual annotation policy used to hand-build the gold
comparison file for abstracts 1-200:
  - Only multi-mention (2+) clusters — matches what fastcoref itself outputs,
    so the two are directly comparable.
  - Author self-reference ("we"/"our"/"us"/"I"/"my") is excluded — it's
    trivial, always-correct coreference that adds no evaluation signal.
  - Mentions are verbatim substrings of the abstract, in order of appearance,
    so they can be located/highlighted the same way the dashboard's mention
    spans are.

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

Task: read the title and abstract given by the user, and identify every \
coreference CLUSTER — a group of 2 or more text spans ("mentions") that all \
refer to the same real-world entity (a proposed method/system, a dataset, a \
model, a metric, or similar) — exactly the kind of chain a coreference \
resolution model is scored on.

Rules, in priority order:

1. ONLY multi-mention clusters. An entity mentioned exactly once has no \
coreference chain — omit it entirely. Do not invent a second mention to pad \
a cluster to size 2.

2. EXCLUDE author self-reference. Do not create a cluster for "we"/"us"/ \
"our"/"I"/"my"/"the authors" no matter how many times it recurs — this is \
trivial, always-correctly-resolved coreference that carries no evaluation \
signal, and it is deliberately excluded from this dataset.

3. Mentions must be EXACT, VERBATIM substrings of the abstract text — same \
casing, same punctuation, same whitespace as they appear. Do not paraphrase, \
normalize, truncate, or correct typos in a mention. If you cannot copy a \
span exactly as it appears, do not include it.

4. List each cluster's mentions in the order they appear in the text \
(left to right).

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

If, after applying rules 1 and 2, no qualifying cluster remains, return an \
empty list — do not force a result.\
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
        max_tokens=2048,
        # The system prompt is byte-identical across every call in the batch —
        # caching it means every abstract after the first pays ~0.1x for it
        # instead of full price. See shared/prompt-caching.md in the claude-api skill.
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": build_user_content(title, abstract)}],
        output_format=AbstractClusters,
    )
    return response
