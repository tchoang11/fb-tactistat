"""Fetch StatsBomb matches, events, and lineups into local Parquet files."""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from tactistat.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    manifest_matches,
    read_manifest,
    write_json_atomic,
    write_parquet_atomic,
)
from tactistat.config import Config

# Open data intentionally uses no StatsBomb credentials.
warnings.filterwarnings("ignore", message=".*credentials were not supplied.*")

MATCHES_FILE = "matches.parquet"
EVENTS_FILE = "events.parquet"
LINEUPS_FILE = "lineups.parquet"
RAW_MANIFEST_FILE = "manifest.json"

# Modest concurrency for the free data endpoint.
MAX_WORKERS = 8


class DataError(Exception):
    """Raised when the requested competition data cannot be fetched or is empty."""


def dataset_identity(config: Config) -> dict[str, Any]:
    """Return the configured competition identity stored in artifact manifests."""
    return {
        "competition_id": int(config["dataset.competition_id"]),
        "season_id": int(config["dataset.season_id"]),
        "competition_name": str(config["dataset.competition_name"]),
        "season_name": str(config["dataset.season_name"]),
    }


def raw_manifest(config: Config) -> dict[str, Any]:
    return {
        "artifact": "statsbomb_raw",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "dataset": dataset_identity(config),
    }


def raw_dir(config: Config) -> Path:
    """``data/raw/`` for the configured competition, created if absent."""
    path = config.path("stats_tool.processed_dir").parent / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


# Fetching


def fetch_matches(config: Config) -> pd.DataFrame:
    """Download the match list for the configured competition and season."""
    from statsbombpy import sb

    competition_id = config["dataset.competition_id"]
    season_id = config["dataset.season_id"]

    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    if matches is None or matches.empty:
        raise DataError(
            f"StatsBomb returned no matches for competition_id={competition_id}, "
            f"season_id={season_id}. Check the pair against "
            "https://github.com/statsbomb/open-data/blob/master/data/competitions.json"
        )
    return matches.sort_values("match_date").reset_index(drop=True)


def _fetch_one(match_id: int, kind: str) -> pd.DataFrame:
    """Fetch and tag events or lineups for one match."""
    from statsbombpy import sb

    if kind == "events":
        frame = sb.events(match_id=match_id)
        if frame is None or frame.empty:
            return pd.DataFrame()
        # Ensure a stable join key across responses.
        frame["match_id"] = match_id
        return frame

    per_team = sb.lineups(match_id=match_id)
    if not per_team:
        return pd.DataFrame()
    frames = []
    for team_name, team_frame in per_team.items():
        if team_frame is None or team_frame.empty:
            continue
        tagged = team_frame.copy()
        tagged["team"] = team_name
        tagged["match_id"] = match_id
        frames.append(tagged)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_per_match(match_ids: list[int], kind: str, desc: str) -> pd.DataFrame:
    """Fetch every match concurrently; fail if any result is missing."""
    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, mid, kind): mid for mid in match_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="match"):
            match_id = futures[future]
            try:
                frame = future.result()
            except Exception as exc:  # noqa: BLE001 - surfaced below with context
                errors.append(f"  match {match_id}: {type(exc).__name__}: {exc}")
                continue
            if frame.empty:
                errors.append(f"  match {match_id}: returned no {kind}")
            else:
                frames.append(frame)

    if errors:
        raise DataError(
            f"Failed to fetch {kind} for {len(errors)} match(es):\n" + "\n".join(errors)
        )
    return pd.concat(frames, ignore_index=True)


# Build and load


def build_raw_dataset(config: Config, force: bool = False) -> dict[str, Path]:
    """Build the raw Parquet cache; reuse it unless ``force`` is set."""
    out = raw_dir(config)
    targets = {
        "matches": out / MATCHES_FILE,
        "events": out / EVENTS_FILE,
        "lineups": out / LINEUPS_FILE,
    }
    manifest_path = out / RAW_MANIFEST_FILE
    expected_manifest = raw_manifest(config)

    if not force and all(path.exists() for path in targets.values()):
        manifest = read_manifest(manifest_path)
        if manifest_matches(manifest, expected_manifest):
            print(f"Raw dataset already present in {out} (use --force to refetch).")
            return targets

        matches = pd.read_parquet(targets["matches"], columns=["competition_id", "season_id"])
        cached_dataset = {
            "competition_id": int(matches["competition_id"].iloc[0]),
            "season_id": int(matches["season_id"].iloc[0]),
        }
        expected_dataset = expected_manifest["dataset"]
        if all(cached_dataset[key] == expected_dataset[key] for key in cached_dataset):
            write_json_atomic(manifest_path, expected_manifest)
            print(f"Raw dataset already present in {out} (manifest added).")
            return targets
        raise DataError(
            f"Cached data in {out} belongs to competition_id="
            f"{cached_dataset['competition_id']}, season_id={cached_dataset['season_id']}; "
            "use --force to replace it."
        )

    competition = f"{config['dataset.competition_name']} {config['dataset.season_name']}"
    print(f"Fetching {competition} from StatsBomb open data...")

    matches = fetch_matches(config)
    match_ids = [int(m) for m in matches["match_id"].tolist()]
    print(f"  {len(match_ids)} matches")

    events = fetch_per_match(match_ids, "events", "  events ")
    lineups = fetch_per_match(match_ids, "lineups", "  lineups")

    # Serialize nested objects that Parquet cannot store as one typed column.
    events = _stringify_nested(events)
    lineups = _stringify_nested(lineups)

    write_parquet_atomic(matches, targets["matches"])
    write_parquet_atomic(events, targets["events"])
    write_parquet_atomic(lineups, targets["lineups"])
    write_json_atomic(manifest_path, expected_manifest)

    for name, path in targets.items():
        size_mb = path.stat().st_size / 1e6
        rows = {"matches": len(matches), "events": len(events), "lineups": len(lineups)}[name]
        print(f"  wrote {path.name:16} {rows:>7,} rows  {size_mb:6.1f} MB")

    return targets


def _stringify_nested(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert list/dict-valued cells to their ``repr`` so parquet accepts them."""
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype != object:
            continue
        sample = result[column].dropna()
        if sample.empty:
            continue
        if isinstance(sample.iloc[0], (list, dict)):
            result[column] = result[column].apply(lambda v: None if v is None else str(v))
    return result


def _preferred_text(values: pd.Series) -> str | None:
    """Choose the most frequent non-blank value, breaking ties alphabetically."""
    clean = values[values.map(lambda value: isinstance(value, str) and bool(value.strip()))]
    if clean.empty:
        return None
    counts = clean.value_counts()
    return sorted(counts[counts == counts.max()].index)[0]


def build_player_index(lineups: pd.DataFrame) -> pd.DataFrame:
    """Return one canonical name, nickname, and team per StatsBomb player ID."""
    frame = lineups.dropna(subset=["player_id"]).copy()
    frame["player_id"] = frame["player_id"].astype(int)
    records = []
    for player_id, group in frame.groupby("player_id", sort=True):
        records.append(
            {
                "player_id": int(player_id),
                "player_name": _preferred_text(group["player_name"]),
                "player_nickname": _preferred_text(group["player_nickname"]),
                "team": _preferred_text(group["team"]),
            }
        )
    return pd.DataFrame.from_records(records)


def build_player_aliases(lineups: pd.DataFrame) -> pd.DataFrame:
    """Return every observed legal name and nickname for each player ID."""
    index = build_player_index(lineups).set_index("player_id")
    records: set[tuple[int, str]] = set()
    for row in lineups.dropna(subset=["player_id"]).itertuples():
        player_id = int(row.player_id)
        for value in (row.player_name, row.player_nickname, index.loc[player_id, "player_name"]):
            if isinstance(value, str) and value.strip():
                records.add((player_id, value.strip()))
    aliases = pd.DataFrame(sorted(records), columns=["player_id", "alias"])
    aliases["player_name"] = aliases["player_id"].map(index["player_name"])
    return aliases


def _load(config: Config, filename: str) -> pd.DataFrame:
    path = raw_dir(config) / filename
    if not path.exists():
        raise DataError(
            f"{path} not found. Build the raw dataset first:\n"
            "    python scripts/01_build_dataset.py"
        )
    return pd.read_parquet(path)


def _validate_raw_manifest(config: Config) -> None:
    manifest = read_manifest(raw_dir(config) / RAW_MANIFEST_FILE)
    if not manifest_matches(manifest, raw_manifest(config)):
        raise DataError(
            "Raw StatsBomb cache has no compatible manifest; "
            "run scripts/01_build_dataset.py to validate it."
        )


def load_matches(config: Config) -> pd.DataFrame:
    """Match list, one row per match."""
    matches = _load(config, MATCHES_FILE)
    expected = dataset_identity(config)
    cached = {
        "competition_id": int(matches["competition_id"].iloc[0]),
        "season_id": int(matches["season_id"].iloc[0]),
    }
    if any(cached[key] != expected[key] for key in cached):
        raise DataError(
            "Cached StatsBomb matches do not match the configured competition and season; "
            "run scripts/01_build_dataset.py --force."
        )
    return matches


def load_events(config: Config) -> pd.DataFrame:
    """Event table, one row per on-ball action across the whole competition."""
    _validate_raw_manifest(config)
    return _load(config, EVENTS_FILE)


def load_lineups(config: Config) -> pd.DataFrame:
    """Lineup table, one row per player per match."""
    _validate_raw_manifest(config)
    return _load(config, LINEUPS_FILE)
