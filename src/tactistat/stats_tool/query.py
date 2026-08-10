"""Query player statistics with evidence; never guess unresolved names."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

import pandas as pd

from tactistat.config import Config
from tactistat.data.statsbomb import DataError, load_lineups, load_matches
from tactistat.stats_tool.aggregate import (
    DERIVED_METRICS,
    METRICS_BY_NAME,
    build_player_matches,
    build_player_totals,
)

PLAYER_MATCHES_FILE = "player_matches.parquet"
PLAYER_TOTALS_FILE = "player_totals.parquet"


# Result models


@dataclass
class MatchRef:
    """One match a number was accumulated over."""

    match_id: int
    date: str
    fixture: str
    stage: str
    value: float
    minutes: float

    def describe(self) -> str:
        return (
            f"{self.date}  {self.fixture} ({self.stage}): {self.value:g} in {self.minutes:.0f} min"
        )


@dataclass
class PlayerRow:
    player_name: str
    team: str
    value: float
    minutes: float
    appearances: int


@dataclass
class StatsAnswer:
    """A computed result with supporting match evidence."""

    metric: str
    per90: bool
    rows: list[PlayerRow]
    evidence: list[MatchRef] = field(default_factory=list)
    note: str | None = None
    ok: bool = True

    def to_context(self, max_evidence: int = 8) -> str:
        """Render the structured result as plain text for synthesis."""
        if not self.ok:
            return f"STATS TOOL: no result. {self.note}"

        unit = f"{self.metric} per 90" if self.per90 else self.metric
        lines = [f"STATS TOOL — {unit} (2022 FIFA World Cup, computed from event data)"]
        for row in self.rows:
            lines.append(
                f"  {row.player_name} ({row.team}): {row.value:.2f} "
                f"[{row.minutes:.0f} min, {row.appearances} apps]"
            )
        if self.note:
            lines.append(f"  note: {self.note}")
        if self.evidence:
            lines.append("  matches:")
            lines.extend(f"    {ref.describe()}" for ref in self.evidence[:max_evidence])
            if len(self.evidence) > max_evidence:
                lines.append(f"    ... and {len(self.evidence) - max_evidence} more")
        return "\n".join(lines)


@dataclass
class PlayerResolution:
    """Outcome of mapping a name in a question onto a name in the data."""

    query: str
    status: str  # ok | not_found | ambiguous
    player_name: str | None = None
    candidates: list[str] = field(default_factory=list)


# Build and load tables


def processed_dir(config: Config):
    path = config.path("stats_tool.processed_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_stats_tables(config: Config) -> dict[str, object]:
    """Compute both tables and write them to ``data/processed/``."""
    out = processed_dir(config)
    player_matches = build_player_matches(config)
    player_totals = build_player_totals(config, player_matches)
    player_matches.to_parquet(out / PLAYER_MATCHES_FILE, index=False)
    player_totals.to_parquet(out / PLAYER_TOTALS_FILE, index=False)
    return {
        "player_matches": out / PLAYER_MATCHES_FILE,
        "player_totals": out / PLAYER_TOTALS_FILE,
        "frames": (player_matches, player_totals),
    }


def _strip_accents(text: str) -> str:
    """Normalize case and accents so "Alvarez" matches "Álvarez"."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower().strip()


class StatsQueryEngine:
    """Loaded tables plus the handful of questions the router can ask of them."""

    def __init__(self, config: Config):
        self.config = config
        out = processed_dir(config)
        matches_path, totals_path = out / PLAYER_MATCHES_FILE, out / PLAYER_TOTALS_FILE
        if not matches_path.exists() or not totals_path.exists():
            raise DataError(
                f"{totals_path.name} not found in {out}. Build the stats tables first:\n"
                "    python scripts/03_build_stats.py"
            )
        self.player_matches = pd.read_parquet(matches_path)
        self.player_totals = pd.read_parquet(totals_path)
        self.matches = load_matches(config).set_index("match_id")
        self.min_minutes = config["dataset.min_minutes_for_per90"]
        self._name_index = self._build_name_index()

    # Name resolution

    def _build_name_index(self) -> dict[str, set[str]]:
        """Index legal names, nicknames, and surnames after normalization."""
        index: dict[str, set[str]] = {}

        def add(alias: str | None, canonical: str) -> None:
            if not alias or not isinstance(alias, str):
                return
            index.setdefault(_strip_accents(alias), set()).add(canonical)

        lineups = load_lineups(self.config)
        nicknames = lineups.groupby("player_name")["player_nickname"].first()

        for canonical in self.player_totals["player_name"]:
            add(canonical, canonical)
            nickname = nicknames.get(canonical)
            add(nickname, canonical)
            for source in (canonical, nickname):
                if isinstance(source, str) and source.split():
                    add(source.split()[-1], canonical)  # surname
        return index

    def resolve_player(self, name: str) -> PlayerResolution:
        """Map a name from a question onto a player in the data."""
        key = _strip_accents(name)

        exact = self._name_index.get(key)
        if exact and len(exact) == 1:
            return PlayerResolution(name, "ok", next(iter(exact)))
        if exact:
            return PlayerResolution(name, "ambiguous", candidates=sorted(exact))

        partial = {
            canonical
            for alias, canonicals in self._name_index.items()
            if key in alias
            for canonical in canonicals
        }
        if len(partial) == 1:
            return PlayerResolution(name, "ok", next(iter(partial)))
        if partial:
            return PlayerResolution(name, "ambiguous", candidates=sorted(partial)[:10])
        return PlayerResolution(name, "not_found")

    # Query helpers

    def _column(self, metric: str, per90: bool) -> str | None:
        if metric not in METRICS_BY_NAME and metric not in DERIVED_METRICS:
            return None
        if not per90:
            return metric if metric in self.player_totals.columns else None
        column = f"{metric}_per90"
        return column if column in self.player_totals.columns else None

    def _evidence(self, player_name: str, metric: str) -> list[MatchRef]:
        """The matches behind a player's total, most productive first."""
        if metric not in self.player_matches.columns:
            return []
        rows = self.player_matches[
            (self.player_matches["player_name"] == player_name)
            & (self.player_matches["minutes"] > 0)
        ]
        refs = []
        for row in rows.itertuples():
            match = self.matches.loc[row.match_id]
            refs.append(
                MatchRef(
                    match_id=int(row.match_id),
                    date=str(match["match_date"]),
                    fixture=f"{match['home_team']} {match['home_score']}-"
                    f"{match['away_score']} {match['away_team']}",
                    stage=str(match["competition_stage"]),
                    value=float(getattr(row, metric)),
                    minutes=float(row.minutes),
                )
            )
        return sorted(refs, key=lambda r: (-r.value, r.date))

    def _row(self, player_name: str, column: str) -> PlayerRow | None:
        matched = self.player_totals[self.player_totals["player_name"] == player_name]
        if matched.empty:
            return None
        record = matched.iloc[0]
        return PlayerRow(
            player_name=player_name,
            team=str(record["team"]),
            value=float(record[column]),
            minutes=float(record["minutes"]),
            appearances=int(record["appearances"]),
        )

    # Query operations

    def player_metric(self, name: str, metric: str, per90: bool = False) -> StatsAnswer:
        """Return one metric for one player."""
        column = self._column(metric, per90)
        if column is None:
            return StatsAnswer(metric, per90, [], ok=False, note=f"unknown metric {metric!r}")

        resolution = self.resolve_player(name)
        if resolution.status != "ok":
            note = (
                f"no player matching {name!r} at this tournament"
                if resolution.status == "not_found"
                else f"{name!r} is ambiguous: {', '.join(resolution.candidates)}"
            )
            return StatsAnswer(metric, per90, [], ok=False, note=note)

        player_name = resolution.player_name
        row = self._row(player_name, column)
        if row is None:
            return StatsAnswer(metric, per90, [], ok=False, note=f"no rows for {player_name!r}")

        note = None
        if per90 and pd.isna(row.value):
            return StatsAnswer(
                metric,
                per90,
                [],
                ok=False,
                note=f"{player_name} played {row.minutes:g} min, below the "
                f"{self.min_minutes}-minute threshold for per-90 rates",
            )
        return StatsAnswer(
            metric, per90, [row], evidence=self._evidence(player_name, metric), note=note
        )

    def leaderboard(
        self, metric: str, top_n: int = 5, per90: bool = False, team: str | None = None
    ) -> StatsAnswer:
        """Rank players by one metric, optionally within a team."""
        column = self._column(metric, per90)
        if column is None:
            return StatsAnswer(metric, per90, [], ok=False, note=f"unknown metric {metric!r}")

        table = self.player_totals
        if team is not None:
            key = _strip_accents(team)
            table = table[table["team"].map(lambda t: _strip_accents(str(t)) == key)]
            if table.empty:
                return StatsAnswer(metric, per90, [], ok=False, note=f"no team matching {team!r}")

        ranked = table.dropna(subset=[column]).nlargest(top_n, column)
        rows = [self._row(name, column) for name in ranked["player_name"]]
        note = f"restricted to players with at least {self.min_minutes} minutes" if per90 else None
        evidence = self._evidence(rows[0].player_name, metric) if rows else []
        return StatsAnswer(metric, per90, [r for r in rows if r], evidence=evidence, note=note)

    def compare(self, names: list[str], metric: str, per90: bool = False) -> StatsAnswer:
        """Compare resolved players by one metric."""
        column = self._column(metric, per90)
        if column is None:
            return StatsAnswer(metric, per90, [], ok=False, note=f"unknown metric {metric!r}")

        rows, unresolved, evidence = [], [], []
        for name in names:
            resolution = self.resolve_player(name)
            if resolution.status != "ok":
                unresolved.append(f"{name} ({resolution.status})")
                continue
            row = self._row(resolution.player_name, column)
            if row is not None:
                rows.append(row)
                evidence.extend(self._evidence(resolution.player_name, metric))

        if not rows:
            return StatsAnswer(
                metric, per90, [], ok=False, note=f"none resolved: {', '.join(unresolved)}"
            )
        note = f"unresolved: {', '.join(unresolved)}" if unresolved else None
        rows.sort(key=lambda r: r.value if not pd.isna(r.value) else -1, reverse=True)
        return StatsAnswer(metric, per90, rows, evidence=evidence, note=note)
