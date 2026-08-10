"""Query player statistics with evidence; never guess unresolved names."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tactistat.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    manifest_matches,
    read_manifest,
    write_json_atomic,
    write_parquet_atomic,
)
from tactistat.config import Config
from tactistat.data.statsbomb import (
    DataError,
    build_player_aliases,
    dataset_identity,
    load_lineups,
    load_matches,
)
from tactistat.stats_tool.aggregate import (
    DERIVED_METRICS,
    METRICS_BY_NAME,
    build_player_matches,
    build_player_totals,
    check_configured_metrics,
)

PLAYER_MATCHES_FILE = "player_matches.parquet"
PLAYER_TOTALS_FILE = "player_totals.parquet"
PLAYER_ALIASES_FILE = "player_aliases.parquet"
STATS_MATCHES_FILE = "stats_matches.parquet"
STATS_MANIFEST_FILE = "stats_manifest.json"


# Result models


@dataclass
class MatchRef:
    """One match a number was accumulated over."""

    player_id: int
    player_name: str
    match_id: int
    date: str
    fixture: str
    stage: str
    value: float
    minutes: float
    numerator: float | None = None
    denominator: float | None = None

    def describe(self) -> str:
        value = f"{self.value:g}"
        if self.numerator is not None and self.denominator is not None:
            ratio = f"{self.value:.1%}" if self.denominator else "undefined"
            value = f"{ratio} ({self.numerator:g}/{self.denominator:g})"
        return f"{self.date}  {self.fixture} ({self.stage}): {value} in {self.minutes:.0f} min"


@dataclass
class PlayerRow:
    player_id: int
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
        lines = [f"STATS TOOL — {unit} (computed from configured event data)"]
        for row in self.rows:
            value = f"{row.value:.2%}" if self.metric == "pass_accuracy" else f"{row.value:.2f}"
            lines.append(
                f"  {row.player_name} ({row.team}): {value} "
                f"[{row.minutes:.0f} min, {row.appearances} apps]"
            )
        if self.note:
            lines.append(f"  note: {self.note}")
        if self.evidence:
            per_player = max(1, max_evidence // max(1, len(self.rows)))
            for row in self.rows:
                refs = [ref for ref in self.evidence if ref.player_id == row.player_id]
                if not refs:
                    continue
                lines.append(f"  matches for {row.player_name}:")
                lines.extend(f"    {ref.describe()}" for ref in refs[:per_player])
                if len(refs) > per_player:
                    lines.append(f"    ... and {len(refs) - per_player} more")
        return "\n".join(lines)


@dataclass
class PlayerResolution:
    """Outcome of mapping a name in a question onto a name in the data."""

    query: str
    status: str  # ok | not_found | ambiguous
    player_id: int | None = None
    player_name: str | None = None
    candidates: list[str] = field(default_factory=list)


# Build and load tables


def processed_dir(config: Config) -> Path:
    path = config.path("stats_tool.processed_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def stats_manifest(config: Config) -> dict[str, object]:
    """Describe the dataset and schema used by processed stats."""
    return {
        "artifact": "player_stats",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "dataset": dataset_identity(config),
    }


def build_stats_tables(config: Config) -> dict[str, object]:
    """Build query-ready tables and write the manifest last."""
    out = processed_dir(config)
    player_matches = build_player_matches(config)
    player_totals = build_player_totals(config, player_matches)

    matches = load_matches(config)[
        [
            "match_id",
            "match_date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "competition_stage",
        ]
    ].copy()
    aliases = build_player_aliases(load_lineups(config))
    aliases = aliases[aliases["player_id"].isin(player_totals["player_id"])].reset_index(drop=True)

    targets = {
        "player_matches": out / PLAYER_MATCHES_FILE,
        "player_totals": out / PLAYER_TOTALS_FILE,
        "player_aliases": out / PLAYER_ALIASES_FILE,
        "stats_matches": out / STATS_MATCHES_FILE,
    }
    write_parquet_atomic(player_matches, targets["player_matches"])
    write_parquet_atomic(player_totals, targets["player_totals"])
    write_parquet_atomic(aliases, targets["player_aliases"])
    write_parquet_atomic(matches, targets["stats_matches"])
    write_json_atomic(out / STATS_MANIFEST_FILE, stats_manifest(config))
    return {
        **targets,
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
        check_configured_metrics(config)
        out = processed_dir(config)
        paths = {
            "matches": out / PLAYER_MATCHES_FILE,
            "totals": out / PLAYER_TOTALS_FILE,
            "aliases": out / PLAYER_ALIASES_FILE,
            "metadata": out / STATS_MATCHES_FILE,
        }
        missing = [path.name for path in paths.values() if not path.exists()]
        if missing:
            raise DataError(
                f"Missing processed stats file(s) in {out}: {', '.join(missing)}. "
                "Build the stats tables first:\n"
                "    python scripts/03_build_stats.py"
            )

        manifest = read_manifest(out / STATS_MANIFEST_FILE)
        if not manifest_matches(manifest, stats_manifest(config)):
            raise DataError(
                "Processed stats do not match the configured dataset or schema. "
                "Rebuild them with python scripts/03_build_stats.py."
            )

        self.player_matches = pd.read_parquet(paths["matches"])
        self.player_totals = pd.read_parquet(paths["totals"])
        self.aliases = pd.read_parquet(paths["aliases"])
        self.matches = pd.read_parquet(paths["metadata"]).set_index("match_id")
        self.min_minutes = int(config["dataset.min_minutes_for_per90"])
        self.allowed_metrics = set(config["stats_tool.metrics"])
        self.per90_metrics = set(config["stats_tool.per90_metrics"])
        self._refresh_per90(self.player_totals)
        self._name_index = self._build_name_index()

    # Name resolution

    def _refresh_per90(self, table: pd.DataFrame) -> None:
        """Apply the active threshold instead of trusting cached rates."""
        eligible = table["minutes"] >= self.min_minutes
        for metric in self.per90_metrics:
            table[f"{metric}_per90"] = (table[metric] / table["minutes"] * 90).where(eligible)
        table["per90_eligible"] = eligible

    def _build_name_index(self) -> dict[str, set[int]]:
        """Index legal names, nicknames, and surnames after normalization."""
        index: dict[str, set[int]] = {}
        self._canonical_names = {
            int(row.player_id): str(row.player_name) for row in self.player_totals.itertuples()
        }

        def add(alias: str | None, player_id: int) -> None:
            if not alias or not isinstance(alias, str):
                return
            index.setdefault(_strip_accents(alias), set()).add(player_id)

        for row in self.aliases.itertuples():
            player_id = int(row.player_id)
            add(row.alias, player_id)
            if isinstance(row.alias, str) and row.alias.split():
                add(row.alias.split()[-1], player_id)
        for player_id, canonical in self._canonical_names.items():
            add(canonical, player_id)
        return index

    def resolve_player(self, name: str) -> PlayerResolution:
        """Map a name from a question onto a player in the data."""
        key = _strip_accents(name)

        exact = self._name_index.get(key)
        if exact and len(exact) == 1:
            player_id = next(iter(exact))
            return PlayerResolution(name, "ok", player_id, self._canonical_names[player_id])
        if exact:
            candidates = sorted(self._canonical_names[player_id] for player_id in exact)
            return PlayerResolution(name, "ambiguous", candidates=candidates)

        partial = {
            player_id
            for alias, player_ids in self._name_index.items()
            if key in alias
            for player_id in player_ids
        }
        if len(partial) == 1:
            player_id = next(iter(partial))
            return PlayerResolution(name, "ok", player_id, self._canonical_names[player_id])
        if partial:
            candidates = sorted(self._canonical_names[player_id] for player_id in partial)[:10]
            return PlayerResolution(name, "ambiguous", candidates=candidates)
        return PlayerResolution(name, "not_found")

    # Query helpers

    def _column(self, metric: str, per90: bool) -> tuple[str | None, str | None]:
        if metric not in METRICS_BY_NAME and metric not in DERIVED_METRICS:
            return None, f"unknown metric {metric!r}"
        if metric not in self.allowed_metrics:
            return None, f"metric {metric!r} is disabled by stats_tool.metrics"
        if not per90:
            return metric, None
        if metric not in self.per90_metrics:
            return None, f"per-90 is not available for metric {metric!r}"
        column = f"{metric}_per90"
        return column, None

    def _scope_match_ids(
        self,
        match_ids: list[int] | None = None,
        stage: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> tuple[set[int] | None, str | None]:
        """Resolve optional match, stage, and date filters."""
        if not any((match_ids, stage, date_from, date_to)):
            return None, None

        table = self.matches.reset_index()
        if match_ids:
            requested = {int(match_id) for match_id in match_ids}
            table = table[table["match_id"].isin(requested)]
        if stage:
            key = _strip_accents(stage)
            table = table[
                table["competition_stage"].map(lambda value: _strip_accents(str(value)) == key)
            ]
        dates = pd.to_datetime(table["match_date"])
        try:
            if date_from:
                table = table[dates >= pd.Timestamp(date_from)]
                dates = pd.to_datetime(table["match_date"])
            if date_to:
                table = table[dates <= pd.Timestamp(date_to)]
        except (TypeError, ValueError) as exc:
            return set(), f"invalid date filter: {exc}"
        if table.empty:
            return set(), "no matches satisfy the requested filters"
        return set(table["match_id"].astype(int)), None

    def _table_for_scope(self, scoped_ids: set[int] | None) -> pd.DataFrame:
        if scoped_ids is None:
            return self.player_totals
        matches = self.player_matches[self.player_matches["match_id"].isin(scoped_ids)]
        if matches.empty:
            return self.player_totals.iloc[0:0].copy()
        table = build_player_totals(self.config, matches)
        self._refresh_per90(table)
        return table

    def _evidence(
        self, player_id: int, metric: str, scoped_ids: set[int] | None = None
    ) -> list[MatchRef]:
        """The matches behind a player's total, most productive first."""
        if metric not in self.player_matches.columns:
            return []
        rows = self.player_matches[self.player_matches["player_id"] == player_id]
        if scoped_ids is not None:
            rows = rows[rows["match_id"].isin(scoped_ids)]
        refs = []
        for row in rows.itertuples():
            match = self.matches.loc[row.match_id]
            numerator = denominator = None
            value = float(getattr(row, metric))
            if metric == "pass_accuracy":
                numerator = float(row.passes_completed)
                denominator = float(row.passes_attempted)
            refs.append(
                MatchRef(
                    player_id=player_id,
                    player_name=str(row.player_name),
                    match_id=int(row.match_id),
                    date=str(match["match_date"]),
                    fixture=f"{match['home_team']} {match['home_score']}-"
                    f"{match['away_score']} {match['away_team']}",
                    stage=str(match["competition_stage"]),
                    value=value,
                    minutes=float(row.minutes),
                    numerator=numerator,
                    denominator=denominator,
                )
            )
        return sorted(refs, key=lambda ref: (pd.isna(ref.value), -ref.value, ref.date))

    def _row(self, table: pd.DataFrame, player_id: int, column: str) -> PlayerRow | None:
        matched = table[table["player_id"] == player_id]
        if matched.empty:
            return None
        record = matched.iloc[0]
        return PlayerRow(
            player_id=player_id,
            player_name=str(record["player_name"]),
            team=str(record["team"]),
            value=float(record[column]),
            minutes=float(record["minutes"]),
            appearances=int(record["appearances"]),
        )

    @staticmethod
    def _join_notes(*notes: str | None) -> str | None:
        return "; ".join(note for note in notes if note) or None

    # Query operations

    def player_metric(
        self,
        name: str,
        metric: str,
        per90: bool = False,
        *,
        match_ids: list[int] | None = None,
        stage: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> StatsAnswer:
        """Return one metric for one player."""
        column, error = self._column(metric, per90)
        if column is None:
            return StatsAnswer(metric, per90, [], ok=False, note=error)

        scoped_ids, scope_error = self._scope_match_ids(match_ids, stage, date_from, date_to)
        if scope_error:
            return StatsAnswer(metric, per90, [], ok=False, note=scope_error)
        table = self._table_for_scope(scoped_ids)

        resolution = self.resolve_player(name)
        if resolution.status != "ok":
            note = (
                f"no player matching {name!r} at this tournament"
                if resolution.status == "not_found"
                else f"{name!r} is ambiguous: {', '.join(resolution.candidates)}"
            )
            return StatsAnswer(metric, per90, [], ok=False, note=note)

        player_id = int(resolution.player_id)
        player_name = str(resolution.player_name)
        row = self._row(table, player_id, column)
        if row is None:
            return StatsAnswer(
                metric, per90, [], ok=False, note=f"no rows for {player_name!r} in this scope"
            )

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
            metric,
            per90,
            [row],
            evidence=self._evidence(player_id, metric, scoped_ids),
        )

    def leaderboard(
        self,
        metric: str,
        top_n: int = 5,
        per90: bool = False,
        team: str | None = None,
        *,
        match_ids: list[int] | None = None,
        stage: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> StatsAnswer:
        """Rank players by one metric, optionally within a team."""
        column, error = self._column(metric, per90)
        if column is None:
            return StatsAnswer(metric, per90, [], ok=False, note=error)
        if not 1 <= top_n <= 50:
            return StatsAnswer(metric, per90, [], ok=False, note="top_n must be between 1 and 50")

        scoped_ids, scope_error = self._scope_match_ids(match_ids, stage, date_from, date_to)
        if scope_error:
            return StatsAnswer(metric, per90, [], ok=False, note=scope_error)
        table = self._table_for_scope(scoped_ids)
        if team is not None:
            key = _strip_accents(team)
            table = table[table["team"].map(lambda t: _strip_accents(str(t)) == key)]
            if table.empty:
                return StatsAnswer(metric, per90, [], ok=False, note=f"no team matching {team!r}")

        ranked = table.dropna(subset=[column]).nlargest(top_n, column)
        rows = [
            row
            for player_id in ranked["player_id"]
            if (row := self._row(table, int(player_id), column)) is not None
        ]
        note = f"restricted to players with at least {self.min_minutes} minutes" if per90 else None
        evidence = [
            ref for row in rows for ref in self._evidence(row.player_id, metric, scoped_ids)
        ]
        return StatsAnswer(metric, per90, rows, evidence=evidence, note=note)

    def compare(
        self,
        names: list[str],
        metric: str,
        per90: bool = False,
        *,
        match_ids: list[int] | None = None,
        stage: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> StatsAnswer:
        """Compare resolved players by one metric."""
        column, error = self._column(metric, per90)
        if column is None:
            return StatsAnswer(metric, per90, [], ok=False, note=error)

        scoped_ids, scope_error = self._scope_match_ids(match_ids, stage, date_from, date_to)
        if scope_error:
            return StatsAnswer(metric, per90, [], ok=False, note=scope_error)
        table = self._table_for_scope(scoped_ids)

        rows, unresolved, ineligible, evidence = [], [], [], []
        for name in names:
            resolution = self.resolve_player(name)
            if resolution.status != "ok":
                unresolved.append(f"{name} ({resolution.status})")
                continue
            player_id = int(resolution.player_id)
            row = self._row(table, player_id, column)
            if row is not None:
                if per90 and pd.isna(row.value):
                    ineligible.append(f"{row.player_name} ({row.minutes:g} min)")
                    continue
                rows.append(row)
                evidence.extend(self._evidence(player_id, metric, scoped_ids))
            else:
                unresolved.append(f"{resolution.player_name} (no rows in this scope)")

        if not rows:
            note = self._join_notes(
                f"unresolved: {', '.join(unresolved)}" if unresolved else None,
                (
                    f"below {self.min_minutes}-minute per-90 threshold: {', '.join(ineligible)}"
                    if ineligible
                    else None
                ),
            )
            return StatsAnswer(metric, per90, [], ok=False, note=note or "no comparable rows")
        note = self._join_notes(
            f"unresolved: {', '.join(unresolved)}" if unresolved else None,
            (
                f"below {self.min_minutes}-minute per-90 threshold: {', '.join(ineligible)}"
                if ineligible
                else None
            ),
        )
        rows.sort(key=lambda row: row.value, reverse=True)
        return StatsAnswer(metric, per90, rows, evidence=evidence, note=note)
