"""Build a competition-scoped Wikipedia corpus and log entity resolutions."""

from __future__ import annotations

import ast
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from tactistat.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    manifest_matches,
    read_manifest,
    stable_hash,
    write_json_atomic,
    write_text_atomic,
)
from tactistat.config import Config
from tactistat.data.statsbomb import (
    DataError,
    build_player_index,
    dataset_identity,
    load_events,
    load_lineups,
    load_matches,
)

CORPUS_FILE = "corpus.jsonl"
RESOLUTION_LOG_FILE = "resolution_log.json"
CORPUS_MANIFEST_FILE = "manifest.json"

# Stored on each page for attribution.
LICENSE = "CC BY-SA 4.0"

# Real Wikipedia titles for tactical retrieval; duplicate pages are removed later.
TACTICAL_CONCEPTS = [
    # Shape and system
    "Formation (association football)",
    "Association football tactics",
    "Association football positions",
    "Total Football",
    "Tiki-taka",
    "Catenaccio",
    "Gegenpressing",
    "Counter-attack",
    "Long ball",
    # Roles
    "Goalkeeper (association football)",
    "Defender (association football)",
    "Midfielder",
    "Forward (association football)",
    "Winger (association football)",
    "Captain (association football)",
    # "Wingback" points to an American-football article.
    # On-ball actions
    "Dribbling",
    "Cross (association football)",
    "Tackle (football move)",
    "Passing (association football)",
    # Defending
    "Marking (association football)",
    "Offside (association football)",
    # Set pieces and restarts
    "Corner kick",
    "Free kick (association football)",
    "Penalty kick (association football)",
    "Penalty shoot-out (association football)",
    "Throw-in",
    # Match rules and structure
    "Substitute (association football)",
    "Overtime (sports)",
    "Video assistant referee",
    "Association football referee",
    # Measurement and vocabulary
    "Expected goals",
    "Glossary of association football terms",
    "Clean sheet",
]


@dataclass
class PageSpec:
    """One thing to look up, and why it is in scope."""

    query: str
    entity_type: str  # team | player | tournament | concept
    entity_key: str | None = None  # exact StatsBomb name, for joining back to stats


@dataclass
class Section:
    heading: str
    level: int
    text: str


@dataclass
class WikiPage:
    page_id: int
    title: str
    url: str
    revision_id: int
    fetched_at: str
    license: str
    entity_type: str
    entity_key: str | None
    query: str
    sections: list[Section] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# Select pages


def _played_positions(cell: Any) -> list[dict]:
    """Parse the ``positions`` column, which parquet stores as a repr string."""
    if isinstance(cell, list):
        return cell
    if not isinstance(cell, str) or not cell.strip() or cell.strip() == "[]":
        return []
    try:
        parsed = ast.literal_eval(cell)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def select_players(config: Config) -> pd.DataFrame:
    """Select notable players and their Wikipedia search names."""
    lineups = load_lineups(config)
    events = load_events(config)

    lineups = lineups.copy()
    lineups["played"] = lineups["positions"].apply(lambda c: len(_played_positions(c)) > 0)
    appeared = lineups[lineups["played"]].copy()
    appeared["player_id"] = appeared["player_id"].astype(int)

    per_player = appeared.groupby("player_id").agg(apps=("match_id", "nunique")).reset_index()
    player_index = build_player_index(lineups).rename(columns={"player_nickname": "nickname"})
    per_player = per_player.merge(player_index, on="player_id", validate="one_to_one")

    # Period 5 contains shootout kicks.
    goals = (
        events[
            (events["type"] == "Shot")
            & (events["shot_outcome"] == "Goal")
            & (events["period"] <= 4)
        ]
        .dropna(subset=["player_id"])
        .assign(player_id=lambda frame: frame["player_id"].astype(int))
        .groupby("player_id")
        .size()
        .rename("goals")
    )
    per_player = per_player.merge(goals, how="left", left_on="player_id", right_index=True)
    per_player["goals"] = per_player["goals"].fillna(0).astype(int)

    keep = per_player["apps"] >= config["wikipedia.min_appearances"]
    if config["wikipedia.include_all_scorers"]:
        keep |= per_player["goals"] > 0

    selected = per_player[keep].copy()
    # Prefer the common nickname for Wikipedia search.
    selected["search_name"] = selected["nickname"].fillna(selected["player_name"])
    return selected.sort_values(["goals", "apps"], ascending=False).reset_index(drop=True)


def build_page_specs(config: Config) -> list[PageSpec]:
    """Everything to fetch: teams, notable players, tournament pages, concepts."""
    matches = load_matches(config)
    specs: list[PageSpec] = []

    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    for team in teams:
        specs.append(PageSpec(f"{team} national football team", "team", team))

    for _, row in select_players(config).iterrows():
        specs.append(PageSpec(row["search_name"], "player", row["player_name"]))

    competition = f"{config['dataset.season_name']} {config['dataset.competition_name']}"
    specs.append(PageSpec(competition, "tournament", competition))
    specs.append(PageSpec(f"{competition} final", "tournament", None))
    specs.append(PageSpec(f"{competition} knockout stage", "tournament", None))
    specs.append(PageSpec(f"{competition} squads", "tournament", None))
    for group in "ABCDEFGH":
        specs.append(PageSpec(f"{competition} Group {group}", "tournament", None))

    for concept in TACTICAL_CONCEPTS:
        specs.append(PageSpec(concept, "concept", None))

    return specs


# Fetching

HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")


def _split_sections(extract: str, config: Config) -> list[Section]:
    """Split plaintext on wiki headings and drop navigational sections."""
    drop = {name.lower() for name in config["wikipedia.drop_sections"]}
    min_chars = config["wikipedia.min_section_chars"]

    sections: list[Section] = []
    heading, level, buffer = "Introduction", 0, []
    # Dropping a level-2 section also drops its children.
    dropping = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if not dropping and len(body) >= min_chars:
            sections.append(Section(heading=heading, level=level, text=body))

    for line in extract.splitlines():
        match = HEADING_RE.match(line)
        if not match:
            buffer.append(line)
            continue
        flush()
        level = len(match.group(1))
        heading = match.group(2)
        buffer = []
        if level == 2:
            dropping = heading.lower() in drop
        elif dropping:
            # Remain inside the dropped parent section.
            pass
    flush()
    return sections


def _is_disambiguation(extract: str) -> bool:
    return "may refer to" in extract[:300].lower()


# Other football codes are invalid resolutions.
OTHER_FOOTBALL_CODES = (
    "gridiron",
    "american football",
    "australian rules",
    "rugby",
    "gaelic football",
)

# Domain-generic words do not help compare a query with a title.
_GENERIC_TOKENS = {
    "football",
    "footballer",
    "association",
    "soccer",
    "fifa",
    "world",
    "cup",
    "national",
    "team",
    "mens",
    "men",
    "womens",
    "women",
    "the",
    "of",
    "a",
    "an",
    "and",
    "in",
    "born",
    "club",
}
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _content_tokens(text: str) -> set[str]:
    """Distinctive lowercase tokens, with domain-generic words removed."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _GENERIC_TOKENS}


def _is_other_football_code(page: dict[str, Any]) -> bool:
    head = (page["title"] + " " + page["extract"][:300]).lower()
    return any(code in head for code in OTHER_FOOTBALL_CODES)


def _years_conflict(query: str, title: str) -> bool:
    """Return whether query and title contain conflicting years."""
    query_years = set(_YEAR_RE.findall(query))
    title_years = set(_YEAR_RE.findall(title))
    return bool(query_years and title_years and not (query_years & title_years))


def _is_footballer_bio(page: dict[str, Any]) -> bool:
    """Detect a footballer biography from its normalized opening text."""
    opening = re.sub(r"\s+", " ", page["extract"][:400]).lower()
    return any(phrase in opening for phrase in ("footballer", "football player", "soccer player"))


class _RateLimiter:
    """Enforce one request rate across all worker threads."""

    def __init__(self, requests_per_second: float):
        self._min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + self._min_interval
        if wait:
            time.sleep(wait)


class WikipediaClient:
    """Thin wrapper over the MediaWiki Action API, rate-limited and retrying."""

    # Transient, worth retrying. Anything else is a real error and is raised.
    RETRY_STATUS = {429, 502, 503, 504}

    def __init__(self, config: Config):
        self.endpoint = f"https://{config['wikipedia.language']}.wikipedia.org/w/api.php"
        self.timeout = config["wikipedia.timeout_seconds"]
        self.maxlag = config["wikipedia.maxlag_seconds"]
        self.max_retries = config["wikipedia.max_retries"]
        self.limiter = _RateLimiter(config["wikipedia.requests_per_second"])
        self.user_agent = config["wikipedia.user_agent"]
        self._sessions = threading.local()

    def _session(self) -> requests.Session:
        """Give each worker its own requests session."""
        session = getattr(self._sessions, "value", None)
        if session is None:
            session = requests.Session()
            session.headers["User-Agent"] = self.user_agent
            self._sessions.value = session
        return session

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        params = {"format": "json", "formatversion": 2, "maxlag": self.maxlag, **params}

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self.limiter.acquire()
            try:
                response = self._session().get(self.endpoint, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(2**attempt)
                continue

            if response.status_code in self.RETRY_STATUS:
                # Prefer the server-provided delay.
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                last_error = requests.HTTPError(f"HTTP {response.status_code}", response=response)
                time.sleep(delay)
                continue

            response.raise_for_status()
            payload = response.json()

            # MediaWiki reports maxlag inside an HTTP 200 body.
            error = payload.get("error", {})
            if error.get("code") == "maxlag":
                last_error = requests.HTTPError(f"maxlag: {error.get('info', '')}")
                time.sleep(2**attempt)
                continue

            return payload

        raise last_error or requests.HTTPError("request failed after retries")

    def get_page(self, title: str) -> dict[str, Any] | None:
        """Fetch one article as plain text, following redirects."""
        data = self._get(
            {
                "action": "query",
                "prop": "extracts|info",
                "explaintext": 1,
                "exsectionformat": "wiki",
                "inprop": "url",
                "redirects": 1,
                "titles": title,
            }
        )
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return None
        page = pages[0]
        if page.get("missing") or not page.get("extract"):
            return None
        return page

    def search(self, query: str, limit: int = 5) -> list[str]:
        data = self._get(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "srnamespace": 0,
            }
        )
        return [hit["title"] for hit in data.get("query", {}).get("search", [])]

    # Search hints disambiguate short names and broad concepts.
    SEARCH_HINT = {"player": "footballer", "team": "national football team", "concept": "football"}

    def _accept(self, spec: PageSpec, page: dict[str, Any], via_search: bool) -> tuple[bool, str]:
        """Validate a direct or search-based page resolution."""
        title, extract = page["title"], page["extract"]

        if _is_disambiguation(extract):
            return False, "disambiguation page"
        if _is_other_football_code(page):
            return False, f"different football code: {title}"
        if _years_conflict(spec.query, title):
            return False, f"year mismatch: {title}"

        # Player pages must look like footballer biographies.
        if spec.entity_type == "player" and not _is_footballer_bio(page):
            return False, f"not a footballer biography: {title}"

        if not via_search:
            return True, "direct"

        query_tokens = _content_tokens(spec.query)
        title_tokens = _content_tokens(title)

        if spec.entity_type == "concept":
            # Concept titles must contain every distinctive query token.
            if not query_tokens <= title_tokens:
                return False, f"title does not name the concept: {title}"
        elif not (query_tokens & title_tokens):
            # People and team names allow partial overlap for transliteration.
            return False, f"no token overlap: {title}"

        return True, f"search -> {title}"

    def resolve(self, spec: PageSpec) -> tuple[dict[str, Any] | None, str]:
        """Find the article for ``spec``. Returns ``(page, how_it_resolved)``."""
        rejections: list[str] = []

        page = self.get_page(spec.query)
        if page is not None:
            ok, reason = self._accept(spec, page, via_search=False)
            if ok:
                return page, reason
            rejections.append(f"direct({reason})")

        # Do not search-fallback tournament pages across editions.
        if spec.entity_type == "tournament":
            return None, "; ".join(rejections) or "no such page"

        # Avoid appending a search hint already present in the query.
        hint = self.SEARCH_HINT.get(spec.entity_type, "")
        query = spec.query if hint.lower() in spec.query.lower() else f"{spec.query} {hint}".strip()
        for candidate in self.search(query):
            candidate_page = self.get_page(candidate)
            if candidate_page is None:
                continue
            ok, reason = self._accept(spec, candidate_page, via_search=True)
            if ok:
                return candidate_page, reason
            rejections.append(f"{candidate}({reason})")

        return None, "; ".join(rejections[:4]) or "no candidates"


# Build corpus


def corpus_manifest(config: Config) -> dict[str, Any]:
    """Describe inputs that affect corpus scope and text."""
    settings = {
        "language": config["wikipedia.language"],
        "min_appearances": config["wikipedia.min_appearances"],
        "include_all_scorers": config["wikipedia.include_all_scorers"],
        "drop_sections": config["wikipedia.drop_sections"],
        "min_section_chars": config["wikipedia.min_section_chars"],
    }
    return {
        "artifact": "wikipedia_corpus",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "dataset": dataset_identity(config),
        "settings_hash": stable_hash(settings),
    }


def _fetch_one(
    client: WikipediaClient, spec: PageSpec, config: Config
) -> tuple[WikiPage | None, dict]:
    try:
        page, how = client.resolve(spec)
    except requests.RequestException as exc:
        return None, {
            "query": spec.query,
            "entity_type": spec.entity_type,
            "status": "error",
            "detail": f"{type(exc).__name__}: {exc}",
        }

    if page is None:
        return None, {
            "query": spec.query,
            "entity_type": spec.entity_type,
            "status": "unresolved",
            "detail": how,
        }

    sections = _split_sections(page["extract"], config)
    if not sections:
        return None, {
            "query": spec.query,
            "entity_type": spec.entity_type,
            "status": "empty",
            "detail": f"{page['title']}: no usable sections",
        }

    wiki_page = WikiPage(
        page_id=page["pageid"],
        title=page["title"],
        url=page["fullurl"],
        revision_id=page["lastrevid"],
        fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
        license=LICENSE,
        entity_type=spec.entity_type,
        entity_key=spec.entity_key,
        query=spec.query,
        sections=sections,
    )
    return wiki_page, {
        "query": spec.query,
        "entity_type": spec.entity_type,
        "status": "ok",
        "title": page["title"],
        "detail": how,
        "sections": len(sections),
    }


def build_corpus(config: Config, force: bool = False) -> Path:
    """Resolve and download every in-scope page, writing JSONL + a resolution log."""
    out_dir = config.path("rag.corpus_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / CORPUS_FILE
    manifest_path = out_dir / CORPUS_MANIFEST_FILE
    expected_manifest = corpus_manifest(config)

    if corpus_path.exists() and not force:
        if manifest_matches(read_manifest(manifest_path), expected_manifest):
            print(f"Corpus already present at {corpus_path} (use --force to refetch).")
            return corpus_path
        raise DataError(
            "Cached Wikipedia corpus does not match the configured dataset or settings; "
            "use --force to rebuild it."
        )

    specs = build_page_specs(config)
    by_type: dict[str, int] = {}
    for spec in specs:
        by_type[spec.entity_type] = by_type.get(spec.entity_type, 0) + 1
    print(
        f"Resolving {len(specs)} pages: "
        + ", ".join(f"{n} {t}" for t, n in sorted(by_type.items()))
    )

    client = WikipediaClient(config)
    pages: list[WikiPage] = []
    log: list[dict] = []

    with ThreadPoolExecutor(max_workers=config["wikipedia.max_workers"]) as pool:
        futures = {pool.submit(_fetch_one, client, spec, config): spec for spec in specs}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="  wikipedia", unit="page"
        ):
            page, entry = future.result()
            log.append(entry)
            if page is not None:
                pages.append(page)

    # Keep one copy of each Wikipedia page ID.
    seen: set[int] = set()
    unique: list[WikiPage] = []
    duplicates = 0
    for page in sorted(pages, key=lambda p: (p.entity_type, p.title)):
        if page.page_id in seen:
            duplicates += 1
            continue
        seen.add(page.page_id)
        unique.append(page)

    write_text_atomic(corpus_path, "".join(page.to_json() + "\n" for page in unique))

    log_path = out_dir / RESOLUTION_LOG_FILE
    write_json_atomic(
        log_path,
        sorted(log, key=lambda entry: (entry["status"], entry["query"])),
    )
    write_json_atomic(manifest_path, expected_manifest)

    total_sections = sum(len(p.sections) for p in unique)
    total_chars = sum(len(s.text) for p in unique for s in p.sections)
    failed = [e for e in log if e["status"] != "ok"]

    print()
    print(f"  pages written    {len(unique)}  ({duplicates} duplicate resolutions dropped)")
    print(f"  sections         {total_sections:,}")
    print(f"  text             {total_chars / 1e6:.2f} M chars")
    print(f"  unresolved       {len(failed)}  (see {log_path.name})")
    print(f"  corpus           {corpus_path}")
    return corpus_path


def load_corpus(config: Config) -> list[WikiPage]:
    """Read the corpus back as ``WikiPage`` objects."""
    path = config.path("rag.corpus_dir") / CORPUS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the corpus first:\n    python scripts/02_build_corpus.py"
        )
    if not manifest_matches(
        read_manifest(path.parent / CORPUS_MANIFEST_FILE), corpus_manifest(config)
    ):
        raise DataError(
            "Wikipedia corpus does not match the configured dataset or settings; rebuild it."
        )
    pages: list[WikiPage] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            record["sections"] = [Section(**s) for s in record["sections"]]
            pages.append(WikiPage(**record))
    return pages
