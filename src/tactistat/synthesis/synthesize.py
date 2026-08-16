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
# Three digits so a stray [597] is caught as invalid rather than left in the
# prose looking like a citation; no passage rank ever reaches three digits.
_CITE_RE = re.compile(r"\[(\d{1,3})\]")
# gpt-oss writes citations in fullwidth CJK brackets; they are still citations.
_BRACKETS = str.maketrans({"【": "[", "】": "]", "［": "[", "］": "]"})
# Standalone numbers only: a digit inside a compound is a scoreline or a
# formation ("4-2", "4-1-4-1"), which is prose, not a tool value.
_NUMBER_RE = re.compile(r"(?<![\d.,\-/:])\d+(?:[.,]\d+)?(?![\d.,\-/:])")
_ANY_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
# Exactly two hyphenated numbers and nothing adjacent: a scoreline. Longer
# chains are formations ("4-1-4-1") or dates, and "1/8" is a round, not a score.
_SCORELINE_RE = re.compile(r"(?<![\d\-–/])(\d+)\s*[-–]\s*(\d+)(?![\d\-–/])")
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
MIN_ANSWER_CHARS = 15

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
    "[2]. Numbers from the STATS TOOL need no bracket. Square brackets are for "
    "passage numbers only; never put anything else inside them."
)
# Asking for citations with no passages to cite just invites invented ones.
CITE_NOTHING = "There are no passages to cite. Never write square brackets."

NO_EVIDENCE = "Neither tool returned usable evidence for this question."


@dataclass
class Answer:
    """A synthesised answer plus what can be checked about it."""

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


def unsupported_metric_claims(text: str, claims: dict[str, set[float]]) -> list[str]:
    """Numbers asserted *of a metric* that the stats tool never computed.

    The loose check below asks only whether a number appears somewhere in the
    evidence, which the date 2022-12-18 alone is enough to satisfy. A claim like
    "18 goals" has to be checked against the goal values specifically, so the
    number is bound to the metric it is asserted about.
    """
    wrong: list[str] = []
    for metric, allowed in claims.items():
        surfaces = METRIC_SURFACE.get(metric)
        if not surfaces or not allowed:
            continue
        pattern = "|".join(re.escape(word) for word in surfaces)
        # A number, then at most two words, then the metric it is claimed of.
        for match in re.finditer(
            rf"(\d+(?:[.,]\d+)?)\s+(?:\S+\s+){{0,2}}?(?:{pattern})\b", text, re.IGNORECASE
        ):
            token = match.group(1)
            value = _numeric_value(token)
            if value is not None and value not in allowed:
                claim = f"{token} {metric}"
                if claim not in wrong:
                    wrong.append(claim)
    return wrong


def reformatted_numbers(text: str, evidence: str) -> list[str]:
    """Right value, rewritten shape — "0,94" for a tool value of "0.94".

    Not an error in prose, but it defeats a string-equality fidelity check, so
    it is reported separately from a number the evidence never stated.
    """
    written = set(re.findall(r"\d+(?:[.,]\d+)?", evidence))
    return [
        token
        for token in dict.fromkeys(_NUMBER_RE.findall(text))
        if "," in token and token not in written
    ]


def unsupported_numbers(text: str, evidence: str) -> list[str]:
    """Numbers the answer asserts that the evidence never reported anywhere.

    Compared by value, not by substring, and applied to the two halves of a
    scoreline as well, so an invented "won 8-0" cannot hide behind punctuation.
    Years are left to `unsupported_metric_claims`, which binds them to a metric;
    here they are ordinary prose.
    """
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

        # Every guard runs on an abstention too: a model can emit the token and
        # a fabricated claim in the same reply.
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

        # Passages were retrieved and none was cited: nothing supports the prose.
        # A stats block alongside it does not make the prose supported, and an
        # abstention token does not exempt a claim that came with it.
        if self.require_citations and n_passages and not cited:
            notes.append("answer cites no retrieved passage")
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
