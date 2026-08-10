"""Fetch StatsBomb matches, events, and lineups into local Parquet files."""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tactistat.config import Config

# Open data intentionally uses no StatsBomb credentials.
warnings.filterwarnings("ignore", message=".*credentials were not supplied.*")

MATCHES_FILE = "matches.parquet"
EVENTS_FILE = "events.parquet"
LINEUPS_FILE = "lineups.parquet"

# Modest concurrency for the free data endpoint.
MAX_WORKERS = 8


class DataError(Exception):
    """Raised when the requested competition data cannot be fetched or is empty."""


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

    if not force and all(path.exists() for path in targets.values()):
        print(f"Raw dataset already present in {out} (use --force to refetch).")
        return targets

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

    matches.to_parquet(targets["matches"], index=False)
    events.to_parquet(targets["events"], index=False)
    lineups.to_parquet(targets["lineups"], index=False)

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


def _load(config: Config, filename: str) -> pd.DataFrame:
    path = raw_dir(config) / filename
    if not path.exists():
        raise DataError(
            f"{path} not found. Build the raw dataset first:\n"
            "    python scripts/01_build_dataset.py"
        )
    return pd.read_parquet(path)


def load_matches(config: Config) -> pd.DataFrame:
    """Match list, one row per match."""
    return _load(config, MATCHES_FILE)


def load_events(config: Config) -> pd.DataFrame:
    """Event table, one row per on-ball action across the whole competition."""
    return _load(config, EVENTS_FILE)


def load_lineups(config: Config) -> pd.DataFrame:
    """Lineup table, one row per player per match."""
    return _load(config, LINEUPS_FILE)
