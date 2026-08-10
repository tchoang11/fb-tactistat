"""Expose stats as a LangChain tool with text and structured output."""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from tactistat.config import Config
from tactistat.stats_tool.aggregate import available_metrics
from tactistat.stats_tool.query import StatsAnswer, StatsQueryEngine


class FootballStatsInput(BaseModel):
    """Structured arguments filled by the router."""

    metric: str = Field(
        description="Statistic to compute. One of: " + ", ".join(sorted(available_metrics()))
    )
    players: list[str] = Field(
        default_factory=list,
        description=(
            "Player names from the question. Leave empty for a tournament-wide "
            "ranking; give one name to look that player up; give two or more to "
            "compare them."
        ),
    )
    per90: bool = Field(
        default=False,
        description=(
            "Normalise by minutes played. Use when the question compares players "
            "with unequal playing time, or asks about a rate, efficiency, or "
            "productivity rather than a raw total."
        ),
    )
    top_n: int = Field(
        default=5, ge=1, le=50, description="How many players to return in a ranking."
    )
    team: str | None = Field(default=None, description="Restrict a ranking to one national team.")
    match_ids: list[int] = Field(
        default_factory=list, description="Optional StatsBomb match IDs to include."
    )
    stage: str | None = Field(
        default=None, description="Optional exact tournament stage, such as Group Stage."
    )
    date_from: str | None = Field(default=None, description="Optional start date in YYYY-MM-DD.")
    date_to: str | None = Field(default=None, description="Optional end date in YYYY-MM-DD.")


def make_stats_tool(config: Config, engine: StatsQueryEngine | None = None) -> BaseTool:
    """Build a tool bound to one reusable query engine."""
    engine = StatsQueryEngine(config) if engine is None else engine

    class ConfiguredFootballStatsInput(FootballStatsInput):
        metric: str = Field(
            description="Statistic to compute. One of: " + ", ".join(sorted(engine.allowed_metrics))
        )

    @tool(
        "football_stats",
        args_schema=ConfiguredFootballStatsInput,
        response_format="content_and_artifact",
    )
    def football_stats(
        metric: str,
        players: list[str] | None = None,
        per90: bool = False,
        top_n: int = 5,
        team: str | None = None,
        match_ids: list[int] | None = None,
        stage: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> tuple[str, StatsAnswer]:
        """Compute a World Cup statistic with supporting match evidence."""
        players = players or []
        scope = {
            "match_ids": match_ids,
            "stage": stage,
            "date_from": date_from,
            "date_to": date_to,
        }
        if len(players) == 0:
            answer = engine.leaderboard(metric, top_n=top_n, per90=per90, team=team, **scope)
        elif len(players) == 1:
            answer = engine.player_metric(players[0], metric, per90=per90, **scope)
        else:
            answer = engine.compare(players, metric, per90=per90, **scope)
        return answer.to_context(), answer

    return football_stats
