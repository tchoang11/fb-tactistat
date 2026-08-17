"""Classify a question and fill the slots its tool needs.

An explicit router node, rather than free tool choice, so routing accuracy can be
scored on its own in the evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from tactistat.config import Config, ConfigError
from tactistat.llm.registry import structured_model

STRATEGIES = ("few_shot", "keyword")
LABELS = ("STAT", "TACTICAL", "HYBRID")
# Preference order when the configured labels exclude the one we would repair to.
FALLBACK_LABELS = ("TACTICAL", "HYBRID", "STAT")
OPERATIONS = ("player", "ranking", "compare", "total")
MAX_TOP_N = 50
TOTAL_KEYWORDS = ("in total", "altogether", "as a team", "combined", "overall")
# A ranking question asks who leads; without one of these it asks how many.
RANKING_KEYWORDS = ("most", "top ", "leading", "highest", "best", "who ")

# Keyword baseline: surface forms that map onto a configured metric.
METRIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "goals": ("goal", "scorer", "scored", "scoring"),
    "assists": ("assist",),
    "xg": ("xg", "expected goal"),
    "shots": ("shot", "attempt on goal"),
    "passes_attempted": ("pass attempted", "passes attempted"),
    "passes_completed": ("pass completed", "passes completed", "completed pass"),
    "pass_accuracy": ("pass accuracy", "passing accuracy", "pass completion"),
    "key_passes": ("key pass", "chance"),
    "dribbles_completed": ("dribble", "take-on", "take on"),
    "tackles": ("tackle",),
    "interceptions": ("interception", "intercepted"),
}
# Capitalised in a question but never a player's name.
NON_NAME_WORDS = frozenset(
    """who which what how why when where whose whom did do does was were is are
    the a an and or but in on at to for of by from with about during after before
    many much most top best fifa world cup qatar 2022 final finals group stage
    round semi quarter knockout tournament team squad player players goals""".split()
)
# Shortest real surname in the 2022 squads is four characters (e.g. Kane).
MIN_NAME_CHARS = 4
COUNT_KEYWORDS = ("how many", "how much", "most", "top ", "highest", "total", "leading", "record")
RATE_KEYWORDS = ("per 90", "per-90", "per game", "per match", "rate", "efficiency", "productivity")
TACTICAL_KEYWORDS = tuple(
    "why, how did, how does, tactic, formation, style, press, pressing, defen, attack, "
    "strategy, approach, explain, describe, what happened, role, shape, build-up, "
    "counter, set piece, manager, coach".split(", ")
)

SYSTEM_PROMPT = """You route questions about the 2022 FIFA World Cup to one of two tools.

STAT      answerable purely from a number computed over event data.
TACTICAL  needs explanation, context or narrative from Wikipedia prose.
HYBRID    needs both: a number, and prose to interpret or explain it.

For STAT or HYBRID, fill `metric` from this list and nothing else:
{metrics}

Then choose `operation`:
  player   one named player's number          -> players=["<name>"]
  compare  two or more named players          -> players=["<a>","<b>"]
  ranking  who leads, the top N               -> players=[], set top_n
  total    a sum over a whole squad or the    -> players=[], set team for one
           whole tournament                      squad, leave null for all

`total` and `ranking` are different questions. "How many goals did Argentina
score?" is a total; "Who were Argentina's top scorers?" is a ranking. A ranking
truncated to top_n does not add up to the total.

Set `per90` only when the question asks about a rate, efficiency or
productivity, or compares players with unequal playing time; never for a total.
`stage` only when the question names one, e.g. Group Stage, Final. `opponent`
only when the question is about a single fixture: "how many goals did Argentina
score against Poland?" is team="Argentina", opponent="Poland".

For TACTICAL or HYBRID, put in `rag_query` what to search English Wikipedia for.

Examples:
Q: How many goals did Kylian Mbappe score?
-> STAT, operation=player, metric=goals, players=["Kylian Mbappe"]
Q: Who were the top scorers at the tournament?
-> STAT, operation=ranking, metric=goals, players=[], top_n=5
Q: How many goals did Argentina score?
-> STAT, operation=total, metric=goals, players=[], team="Argentina"
Q: How many goals were scored at the tournament?
-> STAT, operation=total, metric=goals, players=[], team=null
Q: How many goals did Argentina score against Poland?
-> STAT, operation=total, metric=goals, team="Argentina", opponent="Poland"
Q: Why was Morocco's defence so hard to break down?
-> TACTICAL, rag_query="Morocco defensive tactics 2022 FIFA World Cup"
Q: Was Messi the tournament's best player, and what did the numbers say?
-> HYBRID, operation=player, metric=goals, players=["Lionel Messi"],
   rag_query="Lionel Messi 2022 FIFA World Cup"
Q: Who created the most chances per 90 minutes?
-> STAT, operation=ranking, metric=key_passes, players=[], per90=true

Never answer the question; only route it.
Reply with json of the form {{"label": "...", "operation": "...", "metric": "...",
"players": ["..."], "per90": false, "top_n": 5, "team": null, "stage": null,
"opponent": null, "rag_query": null}}."""


class RouteDecision(BaseModel):
    """Flat on purpose: nested optional objects break strict JSON schema."""

    label: Literal["STAT", "TACTICAL", "HYBRID"] = Field(description="Which tool answers this.")
    operation: Literal["player", "ranking", "compare", "total"] = Field(
        default="ranking", description="Which question to ask of the statistics."
    )
    metric: str | None = Field(default=None, description="Statistic to compute, for STAT/HYBRID.")
    players: list[str] = Field(default_factory=list, description="Player names in the question.")
    per90: bool = Field(default=False, description="Normalise by minutes played.")
    top_n: int = Field(default=5, description="How many players a ranking returns.")
    team: str | None = Field(default=None, description="Restrict a ranking to one national team.")
    stage: str | None = Field(default=None, description="Exact stage, e.g. Group Stage.")
    opponent: str | None = Field(
        default=None, description="The other team, when the question is about one fixture."
    )
    rag_query: str | None = Field(default=None, description="What to search Wikipedia for.")


@dataclass
class Route:
    """A validated decision the graph can act on without further checking."""

    label: str
    query: str
    strategy: str
    stats_args: dict[str, Any] | None = None
    rag_query: str | None = None
    repairs: list[str] = field(default_factory=list)

    @property
    def needs_stats(self) -> bool:
        return self.stats_args is not None

    @property
    def needs_rag(self) -> bool:
        return self.rag_query is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "query": self.query,
            "strategy": self.strategy,
            "stats_args": self.stats_args,
            "rag_query": self.rag_query,
            "repairs": self.repairs,
        }


def router_settings(config: Config) -> tuple[str, tuple[str, ...]]:
    """Validate the router block before any model is built."""
    strategy = config["router.strategy"]
    if strategy not in STRATEGIES:
        raise ConfigError(f"Unknown router.strategy {strategy!r}; expected one of {STRATEGIES}")
    labels = tuple(config["router.labels"])
    unknown = set(labels) - set(LABELS)
    if unknown or not labels:
        raise ConfigError(f"router.labels must be a non-empty subset of {LABELS}, got {labels}")
    return strategy, labels


def _keyword_metric(text: str) -> str | None:
    """First configured metric whose surface forms appear in the question.

    Whole words only, allowing a plural: "goalkeeper" contains "goal" but asks
    about no configured metric, while "goals" and "key passes" must still match.
    """
    for metric, keywords in METRIC_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(k)}(?:e?s)?\b", text) for k in keywords):
            return metric
    return None


def _capitalised_runs(question: str) -> list[str]:
    """Capitalised runs, the only name signal available without an LLM.

    A sentence-initial word capitalises like a name, so a run containing a
    non-name token, or a single word too short to be a surname, is dropped
    rather than sent to the player resolver.
    """
    runs = re.findall(r"\b[A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)*", question)
    keep = []
    for run in runs:
        words = run.split()
        if {word.lower() for word in words} & NON_NAME_WORDS:
            continue
        if len(words) == 1 and len(run) < MIN_NAME_CHARS:
            continue
        keep.append(run)
    return keep[:4]


def _split_names_and_team(question: str, known_teams: set[str]) -> tuple[list[str], str | None]:
    """Separate national teams from player names among the capitalised runs."""
    names, team = [], None
    for run in _capitalised_runs(question):
        if run.casefold() in known_teams:
            team = team or run
        else:
            names.append(run)
    return names, team


def _keyword_opponent(question: str, known_teams: set[str]) -> str | None:
    """The team named after "against" or "vs", when it is a real team."""
    match = re.search(
        r"\b(?:against|versus|vs\.?)\s+([A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)*)", question
    )
    if match and match.group(1).casefold() in known_teams:
        return match.group(1)
    return None


def _keyword_stage(text: str) -> str | None:
    """Stages the stats tool accepts, when the question names one outright."""
    for stage in ("Group Stage", "Round of 16", "Quarter-finals", "Semi-finals", "Final"):
        if stage.lower() in text:
            return stage
    return None


class Router:
    """Routes a question, then repairs any slot the model got wrong."""

    def __init__(
        self,
        config: Config,
        model: Runnable | None = None,
        known_teams: set[str] | None = None,
    ):
        self.config = config
        self.strategy, self.labels = router_settings(config)
        self.allowed_metrics = set(config["stats_tool.metrics"])
        self.per90_metrics = set(config["stats_tool.per90_metrics"])
        # Only the keyword baseline needs these, to avoid calling a country a player.
        self.known_teams = {team.casefold() for team in known_teams or ()}
        self._model = model

    @property
    def model(self) -> Runnable:
        if self._model is None:
            self._model = structured_model(self.config, "router", RouteDecision)
        return self._model

    def _prompt(self, question: str) -> list[tuple[str, str]]:
        metrics = ", ".join(sorted(self.allowed_metrics))
        return [("system", SYSTEM_PROMPT.format(metrics=metrics)), ("human", question)]

    def _keyword_decision(self, question: str) -> RouteDecision:
        """Zero-LLM baseline, and the fallback when the router model fails."""
        text = question.lower()
        metric = _keyword_metric(text)
        counts = metric is not None and any(word in text for word in COUNT_KEYWORDS)
        tactical = any(word in text for word in TACTICAL_KEYWORDS)

        if counts and tactical:
            label = "HYBRID"
        elif counts:
            label = "STAT"
        else:
            # Prose can address an open question; a number cannot.
            label = "TACTICAL"

        names, team = _split_names_and_team(question, self.known_teams)
        if label == "TACTICAL":
            names, team = [], None
        if names:
            operation = "compare" if len(names) > 1 else "player"
        elif any(word in text for word in TOTAL_KEYWORDS) or not any(
            word in text for word in RANKING_KEYWORDS
        ):
            # "How many goals were scored?" is a total whether or not a team is
            # named; only a superlative makes it a ranking.
            operation = "total"
        else:
            operation = "ranking"
        return RouteDecision(
            label=label,
            operation=operation,
            metric=metric if label != "TACTICAL" else None,
            players=names,
            per90=any(word in text for word in RATE_KEYWORDS),
            team=team,
            stage=_keyword_stage(text) if label != "TACTICAL" else None,
            opponent=_keyword_opponent(question, self.known_teams) if label != "TACTICAL" else None,
            rag_query=question if label != "STAT" else None,
        )

    def _fallback_label(self, repairs: list[str], reason: str) -> str:
        """Degrade to a label the config actually enables."""
        for candidate in FALLBACK_LABELS:
            if candidate in self.labels:
                repairs.append(f"{reason}; used {candidate}")
                return candidate
        return self.labels[0]

    def _validate(self, decision: RouteDecision, question: str) -> Route:
        """Repair rather than trust: a wrong slot must not reach a tool."""
        repairs: list[str] = []
        label = decision.label
        if label not in self.labels:
            label = self._fallback_label(repairs, f"label {label!r} not enabled by router.labels")

        metric = decision.metric
        if label in ("STAT", "HYBRID") and metric not in self.allowed_metrics:
            label = self._fallback_label(
                repairs, f"metric {metric!r} is not a configured statistic"
            )
            metric = None

        players = [name.strip() for name in decision.players if name and name.strip()]
        operation = decision.operation if decision.operation in OPERATIONS else "ranking"
        if operation != decision.operation:
            repairs.append(f"unknown operation {decision.operation!r}; used ranking")
        # A named-player operation without a name, or a comparison of one, is
        # not answerable as asked; the player count is the more reliable signal.
        if operation == "compare" and len(players) < 2:
            operation = "player" if players else "ranking"
            repairs.append(f"compare needs two players, got {len(players)}; used {operation}")
        elif operation == "player" and not players:
            operation = "ranking"
            repairs.append("player lookup without a name; used ranking")
        elif operation == "player" and len(players) > 1:
            # Looking one player up would silently discard the others.
            operation = "compare"
            repairs.append(f"player lookup named {len(players)} players; used compare")
        elif operation in ("ranking", "total") and players:
            repairs.append(f"{operation} ignores the named players {players}")
            players = []

        per90 = decision.per90
        if per90 and operation == "total":
            repairs.append("per-90 is not defined for a squad total; used the sum")
            per90 = False
        if per90 and metric is not None and metric not in self.per90_metrics:
            repairs.append(f"per-90 is not defined for {metric!r}; used the total")
            per90 = False

        top_n = decision.top_n if isinstance(decision.top_n, int) else 5
        if not 1 <= top_n <= MAX_TOP_N:
            repairs.append(f"top_n {decision.top_n!r} out of range; used 5")
            top_n = 5

        stats_args = None
        if label in ("STAT", "HYBRID"):
            stats_args = {
                "operation": operation,
                "metric": metric,
                "players": players,
                "per90": per90,
                "top_n": top_n,
                "team": decision.team,
                "stage": decision.stage,
                "opponent": decision.opponent,
            }

        rag_query = None
        if label in ("TACTICAL", "HYBRID"):
            rag_query = (decision.rag_query or "").strip() or question
        return Route(label, question, self.strategy, stats_args, rag_query, repairs)

    def route(self, question: str) -> Route:
        """Classify a question that has already been translated to English."""
        question = (question or "").strip()
        if not question:
            # Keep even the empty-input fallback inside the enabled label set.
            route = self._validate(RouteDecision(label=FALLBACK_LABELS[0]), question)
            route.repairs.insert(0, "empty question")
            return route
        if self.strategy == "keyword":
            return self._validate(self._keyword_decision(question), question)

        try:
            decision = self.model.invoke(self._prompt(question))
            # A json_mode model returns a mapping that may not satisfy the schema.
            if isinstance(decision, dict):
                decision = RouteDecision(**decision)
            if not isinstance(decision, RouteDecision):
                raise TypeError(f"router returned {type(decision).__name__}")
        except Exception as exc:  # noqa: BLE001 - keep an evaluation run alive
            route = self._validate(self._keyword_decision(question), question)
            route.repairs.insert(0, f"router model failed ({type(exc).__name__}); used keywords")
            return route
        return self._validate(decision, question)
