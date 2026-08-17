"""Turn tool output into a cited answer, or abstain."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from tactistat.config import Config
from tactistat.llm.registry import chat_model

ABSTAIN_TOKEN = "INSUFFICIENT_EVIDENCE"
# Accept up to three digits so stray brackets such as [597] can be rejected.
_CITE_RE = re.compile(r"\[(\d{1,3})\]")
# Normalise fullwidth brackets emitted by gpt-oss.
_BRACKETS = str.maketrans({"【": "[", "】": "]", "［": "[", "］": "]"})
# Compounds such as scorelines and formations are checked separately.
_NUMBER_RE = re.compile(r"(?<![\d.,\-/:])\d+(?:[.,]\d+)?(?![\d.,\-/:])")
_ANY_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
# Match scorelines, but not formations, dates, or rounds.
_SCORELINE_RE = re.compile(r"(?<![\d\-–/])(\d+)\s*[-–]\s*(\d+)(?![\d\-–/])")
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
# Decimal points have no following space, so they remain intact.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_BULLET_RE = re.compile(r"^\s*(?:[-*•–—]|\d{1,2}[.)])\s+")
# Move a leading citation back to the sentence it trails.
_LEADING_CITE_RE = re.compile(r"^(?:\[\d{1,3}\]\s*)+")
# Hard boundaries split directly; `and`/`và` need the support check below so
# coordinated subjects are not mistaken for separate claims.
_CLAUSE_BOUNDARY_RE = re.compile(
    r"(?P<punct>[;:](?!\d))|\s+(?P<join>and|but|however|và|nhưng|tuy nhiên)\s+",
    re.IGNORECASE,
)
_COORDINATING_JOINS = frozenset({"and", "và"})
MIN_ANSWER_CHARS = 15
# Non-claims that need no evidence.
ACKNOWLEDGEMENTS = frozenset(
    {"yes", "no", "correct", "incorrect", "đúng", "sai", "không", "có", "đúng vậy"}
)

# How a metric is named in an answer, in either language the system accepts.
METRIC_SURFACE: dict[str, tuple[str, ...]] = {
    "goals": ("goal", "goals", "bàn", "bàn thắng"),
    "assists": ("assist", "assists", "kiến tạo"),
    "xg": ("xg", "expected goals"),
    "shots": ("shot", "shots", "cú sút", "sút"),
    "passes_attempted": ("passes", "passes attempted", "đường chuyền"),
    "passes_completed": ("completed passes", "đường chuyền chính xác"),
    "key_passes": ("key pass", "key passes", "đường chuyền quyết định"),
    "dribbles_completed": ("dribble", "dribbles", "pha rê bóng"),
    "tackles": ("tackle", "tackles", "pha tắc bóng"),
    "interceptions": ("interception", "interceptions", "pha cắt bóng"),
    "minutes": ("minute", "minutes", "phút"),
    "appearances": ("appearance", "appearances", "match", "matches", "trận"),
}

# Answer in the language the question was asked in, not the corpus language.
LANGUAGE_NAMES = {"vi": "Vietnamese", "en": "English"}

SYSTEM_PROMPT = """You answer questions about the 2022 FIFA World Cup from the evidence
below, and from nothing else.

Rules:
- Use only the evidence. Never add a fact, number or name that is not in it.
- Copy numbers from the STATS TOOL block exactly, digit for digit, keeping the
  decimal point as written. Do not round, recompute or reformat them.
- {citation_rule}
- If the evidence does not answer the question, reply with {abstain} followed by
  one sentence naming what is missing.
- Answer in {language}. Be direct; three sentences is usually enough.

Evidence:
{evidence}"""

CITE_PASSAGES = (
    "Cite every claim taken from a RAG TOOL passage with its bracketed number, like "
    "[2]. Every sentence needs its own bracket; one citation does not carry over to "
    "the next sentence. A sentence that only restates STATS TOOL numbers needs no "
    "bracket. Write no sentence you cannot cite. Square brackets are for passage "
    "numbers only; never put anything else inside them."
)
# Asking for citations with no passages to cite just invites invented ones.
CITE_NOTHING = "There are no passages to cite. Never write square brackets."

NO_EVIDENCE = "Neither tool returned usable evidence for this question."


@dataclass
class Answer:
    """A synthesised answer and its mechanical evidence checks.

    `ok` checks numeric support, claim attribution, and citation ranges—not
    whether a cited passage entails its claim. Evaluation measures faithfulness
    with a separate judge.
    """

    question: str
    text: str
    cited: list[int] = field(default_factory=list)
    abstained: bool = False
    note: str | None = None
    ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "text": self.text,
            "cited": self.cited,
            "abstained": self.abstained,
            "note": self.note,
            "ok": self.ok,
        }


def _language_name(code: str | None) -> str:
    return LANGUAGE_NAMES.get((code or "").lower(), "the same language as the question")


def _strip_unsupported(text: str, n_passages: int) -> tuple[str, list[int], int]:
    """Keep citations that point at a real passage; delete the rest."""
    cited: list[int] = []
    dropped = 0

    def keep(match: re.Match[str]) -> str:
        nonlocal dropped
        rank = int(match.group(1))
        if 1 <= rank <= n_passages:
            if rank not in cited:
                cited.append(rank)
            return match.group(0)
        dropped += 1
        return ""

    cleaned = _CITE_RE.sub(keep, text.translate(_BRACKETS))
    return re.sub(r" +([.,;:])", r"\1", cleaned).strip(), sorted(cited), dropped


def _numeric_value(token: str) -> float | None:
    """Read a number token, accepting a decimal comma."""
    try:
        return float(token.replace(",", "."))
    except ValueError:
        return None


def evidence_values(evidence: str) -> set[float]:
    """Every number the evidence states, including inside dates and scorelines."""
    return {
        value
        for token in _ANY_NUMBER_RE.findall(evidence)
        if (value := _numeric_value(token)) is not None
    }


def _surface_index() -> tuple[re.Pattern[str], dict[str, str]]:
    """Build one longest-first matcher so the nearest metric owns each number."""
    lookup = {word.lower(): metric for metric, words in METRIC_SURFACE.items() for word in words}
    alternation = "|".join(re.escape(word) for word in sorted(lookup, key=len, reverse=True))
    # A short nonnumeric gap avoids binding 2022 to a later goal count.
    pattern = re.compile(
        rf"(\d+(?:[.,]\d+)?)\s+((?:(?!\d)\S+\s+){{0,2}}?)({alternation})\b", re.IGNORECASE
    )
    return pattern, lookup


_CLAIM_RE, _SURFACE_TO_METRIC = _surface_index()


def unsupported_metric_claims(text: str, claims: dict[str, set[float]]) -> list[str]:
    """Find values that the answer binds to the wrong stats metric."""
    wrong: list[str] = []
    for match in _CLAIM_RE.finditer(text):
        token, surface = match.group(1), match.group(3).lower()
        metric = _SURFACE_TO_METRIC.get(surface)
        allowed = claims.get(metric or "")
        if not allowed:
            continue
        value = _numeric_value(token)
        if value is not None and value not in allowed:
            claim = f"{token} {metric}"
            if claim not in wrong:
                wrong.append(claim)
    return wrong


def claim_sentences(text: str) -> list[str]:
    """Split sentences while attaching a post-period citation to its claim."""
    sentences: list[str] = []
    for part in _SENTENCE_SPLIT_RE.split(text):
        part = _BULLET_RE.sub("", part).strip()
        if not part:
            continue
        lead = _LEADING_CITE_RE.match(part)
        if lead and sentences:
            sentences[-1] += " " + lead.group(0).strip()
            part = part[lead.end() :].strip()
            if not part:
                continue
        sentences.append(part)
    return sentences


def _numbers_in(text: str) -> set[float]:
    return {
        value
        for token in _ANY_NUMBER_RE.findall(text)
        if (value := _numeric_value(token)) is not None
    }


def _restates_stats(clause: str, stats_values: set[float]) -> bool:
    """Return whether every number in this clause came from the stats tool."""
    values = _numbers_in(clause)
    return bool(values) and values <= stats_values


def _asserts_nothing(clause: str) -> bool:
    """An acknowledgement ("Yes.", "Không.") supports no claim and needs none."""
    return clause.strip(" .!?;:,").lower() in ACKNOWLEDGEMENTS


def _split_clauses(sentence: str, stats_values: set[float]) -> list[str]:
    """Split claims without breaking coordinated subjects.

    `and`/`và` splits only after supported text; this catches "7 goals and
    France won" without turning "Messi and Mbappe" into two claims.
    """
    clauses: list[str] = []
    start = 0
    for boundary in _CLAUSE_BOUNDARY_RE.finditer(sentence):
        left = sentence[start : boundary.start()].strip()
        join = (boundary.group("join") or "").lower()
        if join in _COORDINATING_JOINS and not (
            _CITE_RE.search(left) or _restates_stats(left, stats_values)
        ):
            continue
        if left:
            clauses.append(left)
        start = boundary.end()
    tail = sentence[start:].strip()
    if tail:
        clauses.append(tail)
    return clauses


def _uncited_in(sentence: str, stats_values: set[float]) -> list[str]:
    """Find unsupported clauses; a citation covers preceding clauses only."""
    uncited: list[str] = []
    cited_after = False
    for clause in reversed(_split_clauses(sentence, stats_values)):
        clause = clause.strip()
        if not clause:
            continue
        if _CITE_RE.search(clause):
            cited_after = True
            continue
        if cited_after or _asserts_nothing(clause) or _restates_stats(clause, stats_values):
            continue
        uncited.append(clause)
    uncited.reverse()
    return uncited


def uncited_claims(text: str, n_passages: int, stats_values: set[float]) -> list[str]:
    """Find clauses with neither a citation nor a supported stats value.

    This checks attribution only; the evaluation judge checks entailment.
    """
    if not n_passages:
        return []
    return [
        clause
        for sentence in claim_sentences(text)
        for clause in _uncited_in(sentence, stats_values)
    ]


def reformatted_numbers(text: str, evidence: str) -> list[str]:
    """Find supported values whose decimal separator changed."""
    written = set(re.findall(r"\d+(?:[.,]\d+)?", evidence))
    return [
        token
        for token in dict.fromkeys(_NUMBER_RE.findall(text))
        if "," in token and token not in written
    ]


def unsupported_numbers(text: str, evidence: str) -> list[str]:
    """Find unsupported standalone values and scoreline components."""
    known = evidence_values(evidence)
    invented: list[str] = []
    claimed = _NUMBER_RE.findall(text) + [
        half for match in _SCORELINE_RE.findall(text) for half in match
    ]
    for token in claimed:
        if _YEAR_RE.match(token) or token in invented:
            continue
        value = _numeric_value(token)
        if value is None or value in known:
            continue
        invented.append(token)
    return invented


class Synthesizer:
    """Writes the final answer and enforces the evidence rules afterwards."""

    def __init__(self, config: Config, model: BaseChatModel | Runnable | None = None):
        self.config = config
        self.allow_abstain = bool(config["synthesis.allow_abstain"])
        self.require_citations = bool(config["synthesis.require_citations"])
        self._model = model

    @property
    def model(self) -> BaseChatModel | Runnable:
        if self._model is None:
            self._model = chat_model(self.config, "synthesis")
        return self._model

    def _prompt(
        self, question: str, evidence: str, language: str | None, n_passages: int
    ) -> list[tuple[str, str]]:
        system = SYSTEM_PROMPT.format(
            abstain=ABSTAIN_TOKEN,
            language=_language_name(language),
            citation_rule=CITE_PASSAGES if n_passages else CITE_NOTHING,
            evidence=evidence,
        )
        return [("system", system), ("human", question)]

    def answer(
        self,
        question: str,
        *,
        stats_context: str | None = None,
        rag_context: str | None = None,
        n_passages: int = 0,
        language: str | None = None,
        stats_claims: dict[str, set[float]] | None = None,
    ) -> Answer:
        """Answer from the supplied tool contexts, or say the evidence is short."""
        question = (question or "").strip()
        blocks = [block for block in (stats_context, rag_context) if block and block.strip()]
        if not question:
            return Answer(question, NO_EVIDENCE, abstained=True, ok=False, note="empty question")
        if not blocks:
            return Answer(question, NO_EVIDENCE, abstained=True, ok=False, note="no tool evidence")

        try:
            prompt = self._prompt(question, "\n\n".join(blocks), language, n_passages)
            reply = self.model.invoke(prompt)
        except Exception as exc:  # noqa: BLE001 - keep an evaluation run alive
            note = f"synthesis failed: {type(exc).__name__}: {exc}"
            return Answer(question, NO_EVIDENCE, abstained=True, ok=False, note=note)

        # Gemini returns content as a list of blocks, not a string.
        text = (getattr(reply, "text", None) or getattr(reply, "content", reply) or "").strip()
        abstained = ABSTAIN_TOKEN in text
        if abstained:
            text = text.replace(ABSTAIN_TOKEN, "").strip(" .:\n") or NO_EVIDENCE
        text, cited, dropped = _strip_unsupported(text, n_passages)

        if len(text) < MIN_ANSWER_CHARS:
            note = f"answer too short to be a claim ({len(text)} chars)"
            return Answer(question, NO_EVIDENCE, abstained=True, ok=False, note=note)

        # Abstention tokens do not bypass evidence checks.
        notes: list[str] = []
        ok = True
        evidence = "\n".join(blocks)

        if dropped:
            # A deleted citation means the claim beside it was never supported.
            notes.append(f"removed {dropped} fabricated citation(s); claims left uncited")
            ok = False

        misattributed = unsupported_metric_claims(text, stats_claims or {})
        if misattributed:
            notes.append(f"claim(s) the stats tool never computed: {', '.join(misattributed)}")
            ok = False

        invented = unsupported_numbers(text, evidence)
        if invented:
            notes.append(f"number(s) not found in the evidence: {', '.join(invented)}")
            ok = False
        elif not misattributed:
            restyled = reformatted_numbers(text, evidence)
            if restyled:
                notes.append(f"number(s) reformatted from the tool output: {', '.join(restyled)}")

        # Stats evidence cannot support uncited retrieved prose.
        if self.require_citations and n_passages and not cited:
            notes.append("answer cites no retrieved passage")
            ok = False
        elif self.require_citations:
            # Check support per clause, not merely once per answer.
            loose = uncited_claims(text, n_passages, evidence_values(stats_context or ""))
            if loose:
                shown = "; ".join(sentence[:60] for sentence in loose[:3])
                notes.append(f"{len(loose)} uncited claim(s): {shown}")
                ok = False

        if abstained and not self.allow_abstain:
            notes.append("model abstained but abstention is disabled")
            ok = False
        return Answer(
            question,
            text,
            cited=cited,
            abstained=abstained,
            ok=ok,
            note="; ".join(notes) or None,
        )
