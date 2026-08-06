"""Collect the Wikipedia half of the corpus, scoped to the loaded competition.

Two problems this module exists to solve.

**Scope.** Section 4.2 of the spec says to stay inside the chosen competition.
Rather than trust a hand-maintained page list to stay in sync, the entity list
is *derived from the event data*: teams that played, players who appeared often
enough to matter, plus the tournament's own pages and a curated set of tactical
concepts. Change the competition in the config and the corpus follows.

**Names.** StatsBomb records legal names; Wikipedia uses common names.

    Lionel Andrés Messi Cuccittini   ->  Lionel Messi
    Kylian Mbappé Lottin             ->  Kylian Mbappé
    Abdul Rahman Baba                ->  Baba Rahman

There is no rule that maps one to the other, so each name is resolved against
the live API and every resolution is logged. A wrong resolution is worse than a
missing page: silently indexing the wrong Wikipedia article gives the retriever
plausible text about the wrong person, and the synthesis step will cite it
confidently. Hence the football check in ``_looks_like_football_page`` and the
audit trail in ``resolution_log.json``.

Output (``data/processed/wikipedia/``, committed to git -- it is small and it
makes the repo runnable without a crawl):

    corpus.jsonl          one JSON record per page, sections preserved
    resolution_log.json   every query, what it resolved to, and what failed
"""

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

from tactistat.config import Config
from tactistat.data.statsbomb import load_events, load_lineups, load_matches

CORPUS_FILE = "corpus.jsonl"
RESOLUTION_LOG_FILE = "resolution_log.json"

# Wikipedia text is CC BY-SA 4.0. Recorded on every page so the synthesis step
# can attribute correctly and the repo stays license-clean.
LICENSE = "CC BY-SA 4.0"

# Football concepts a TACTICAL question is likely to lean on.
#
# These are real article titles, not invented phrases. The first version of this
# list contained descriptive queries such as "Park the bus football" and
# "Overlapping run football", which have no article; every one of them fell
# through to search and came back with something unrelated -- "Through ball"
# resolved to "2026 FIFA World Cup Group C". Wikipedia's redirect graph handles
# the aliases we actually need (Gegenpressing -> Association football tactics),
# so a title that exists beats a phrase that reads well.
#
# Several entries deliberately collapse onto the same article: "Midfielder"
# covers both attacking and defensive roles. The corpus builder de-duplicates by
# page ID and reports how many collapsed.
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
    # "Wingback" is deliberately absent: that article is the American-football
    # position, and the association-football wing-back redirects to
    # "Defender (association football)", which is already listed above.
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


# --------------------------------------------------------------------------- #
# Choosing what to fetch                                                       #
# --------------------------------------------------------------------------- #


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
    """Players worth a Wikipedia page, with their common name where known.

    Returns columns ``player_name`` (StatsBomb legal name), ``search_name``
    (what to look up), ``team``, ``apps``, ``goals``.
    """
    lineups = load_lineups(config)
    events = load_events(config)

    lineups = lineups.copy()
    lineups["played"] = lineups["positions"].apply(lambda c: len(_played_positions(c)) > 0)
    appeared = lineups[lineups["played"]]

    per_player = (
        appeared.groupby("player_name")
        .agg(
            apps=("match_id", "nunique"),
            team=("team", "first"),
            nickname=("player_nickname", "first"),
        )
        .reset_index()
    )

    # Shootout kicks are period 5 and are not goals; see tests/test_data_integrity.py.
    goals = (
        events[
            (events["type"] == "Shot")
            & (events["shot_outcome"] == "Goal")
            & (events["period"] <= 4)
        ]
        .groupby("player")
        .size()
        .rename("goals")
    )
    per_player = per_player.merge(goals, how="left", left_on="player_name", right_index=True)
    per_player["goals"] = per_player["goals"].fillna(0).astype(int)

    keep = per_player["apps"] >= config["wikipedia.min_appearances"]
    if config["wikipedia.include_all_scorers"]:
        keep |= per_player["goals"] > 0

    selected = per_player[keep].copy()
    # StatsBomb's nickname is the common name when it has one; otherwise the
    # legal name usually already is the common name ("Aaron Ramsey").
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


# --------------------------------------------------------------------------- #
# Fetching                                                                     #
# --------------------------------------------------------------------------- #

HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")


def _split_sections(extract: str, config: Config) -> list[Section]:
    """Turn a plaintext extract into sections, dropping navigational ones.

    The API returns headings as wiki markup (``== Club career ==``), so the
    structure survives the conversion to plain text. Splitting on it keeps the
    baseline chunking strategy (Section 5.3) honest: a chunk is a real section
    of the article, not an arbitrary token window.
    """
    drop = {name.lower() for name in config["wikipedia.drop_sections"]}
    min_chars = config["wikipedia.min_section_chars"]

    sections: list[Section] = []
    heading, level, buffer = "Introduction", 0, []
    # A dropped top-level section takes its subsections with it: "References"
    # owns any "=== Cited works ===" beneath it.
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
            # Still inside a dropped top-level section.
            pass
    flush()
    return sections


def _is_disambiguation(extract: str) -> bool:
    return "may refer to" in extract[:300].lower()


# Codes of football that are not association football. A page about one of them
# is never a valid resolution here: "Route One football" resolved to
# "Route (gridiron football)" on the first crawl.
OTHER_FOOTBALL_CODES = (
    "gridiron",
    "american football",
    "australian rules",
    "rugby",
    "gaelic football",
)

# Words too common in this domain to carry signal when comparing a query
# against a resolved title. "football" appears in nearly every page here, so
# matching on it would let any football page satisfy any football query.
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
    """True when query and title both name years and none of them agree.

    This is the check that catches the worst class of failure seen on the first
    crawl: "2022 FIFA World Cup Group H" resolving to "2026 FIFA World Cup
    Group A". Both pages are about football, both are World Cup group pages,
    and every topical heuristic accepts the swap -- only the year exposes it.
    """
    query_years = set(_YEAR_RE.findall(query))
    title_years = set(_YEAR_RE.findall(title))
    return bool(query_years and title_years and not (query_years & title_years))


def _is_footballer_bio(page: dict[str, Any]) -> bool:
    """Wikipedia opens player biographies with a standard phrase.

    "... is an Argentine professional footballer who plays as ..." -- checking
    the opening sentence separates a player from a politician or musician who
    happens to share the name.

    Whitespace is collapsed first. Stripped wikilinks and templates leave double
    spaces in the plaintext extract, and the US players' articles happen to read
    "American professional soccer  player" -- a substring match on the
    single-spaced phrase rejected both Tim Ream and Tyler Adams.
    """
    opening = re.sub(r"\s+", " ", page["extract"][:400]).lower()
    return any(phrase in opening for phrase in ("footballer", "football player", "soccer player"))


class _RateLimiter:
    """Enforce a global request rate across every worker thread.

    Sleeping inside each thread bounds that thread's rate, not the process's:
    N workers each pausing 0.1s still burst at 10N requests/second. The limiter
    serialises the decision of *when* the next request may leave, so the cap
    holds no matter how many workers are running.
    """

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
        self.session = requests.Session()
        self.session.headers["User-Agent"] = config["wikipedia.user_agent"]

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        params = {"format": "json", "formatversion": 2, "maxlag": self.maxlag, **params}

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self.limiter.acquire()
            try:
                response = self.session.get(self.endpoint, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(2**attempt)
                continue

            if response.status_code in self.RETRY_STATUS:
                # The server knows better than any local guess how long to wait.
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                last_error = requests.HTTPError(f"HTTP {response.status_code}", response=response)
                time.sleep(delay)
                continue

            response.raise_for_status()
            payload = response.json()

            # maxlag rejections arrive as HTTP 200 with an error body, so a
            # status-code-only check would treat them as valid empty results.
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

    # Search fallback needs a disambiguating hint. Searching the bare string
    # "Fred" returns Fred Trump and Fred Astaire; "Fred footballer" returns
    # "Fred (footballer, born 1993)" as the first hit.
    SEARCH_HINT = {"player": "footballer", "team": "national football team", "concept": "football"}

    def _accept(self, spec: PageSpec, page: dict[str, Any], via_search: bool) -> tuple[bool, str]:
        """Decide whether ``page`` is a valid resolution of ``spec``.

        Direct hits and search hits are held to different standards on purpose.
        A direct hit means Wikipedia's own title resolution -- including its
        redirect graph -- pointed here, which is far better evidence than any
        local heuristic; "Gegenpressing" redirecting to "Association football
        tactics" is a correct answer that a strict title comparison would
        reject. A search hit is a guess by a relevance ranker that knows
        nothing about this competition, so it has to earn acceptance.
        """
        title, extract = page["title"], page["extract"]

        if _is_disambiguation(extract):
            return False, "disambiguation page"
        if _is_other_football_code(page):
            return False, f"different football code: {title}"
        if _years_conflict(spec.query, title):
            return False, f"year mismatch: {title}"

        # Player identity is the one thing a topical check can verify cheaply,
        # and getting it wrong means confidently citing the wrong person.
        if spec.entity_type == "player" and not _is_footballer_bio(page):
            return False, f"not a footballer biography: {title}"

        if not via_search:
            return True, "direct"

        query_tokens = _content_tokens(spec.query)
        title_tokens = _content_tokens(title)

        if spec.entity_type == "concept":
            # A concept article is named after the concept. Requiring every
            # distinctive query token in the title rejects "Through ball" ->
            # "2026 FIFA World Cup Group C" and "Counter-pressing" ->
            # "Jürgen Klopp", while still accepting "Set piece (association
            # football)" -> "Set piece (football)".
            if not query_tokens <= title_tokens:
                return False, f"title does not name the concept: {title}"
        elif not (query_tokens & title_tokens):
            # People and teams tolerate looser matching: transliteration varies
            # ("Cho Kyu-Sung" -> "Cho Gue-sung"), so demanding every token
            # would reject correct resolutions.
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

        # Tournament titles are exact and predictable -- "2022 FIFA World Cup
        # Group H" either exists under that name or does not. Searching for a
        # near-miss is how the first crawl ended up indexing the 2026 and 2030
        # tournaments, so there is no fallback here: an unresolved page is a
        # visible gap, a wrong page is a silent one.
        if spec.entity_type == "tournament":
            return None, "; ".join(rejections) or "no such page"

        # Append the hint only when it is not already in the query. Team specs
        # are built as "<team> national football team", and blindly appending
        # the team hint produced "Canada national football team national
        # football team", whose top hits were Cape Verde and Ivory Coast. The
        # un-doubled query returns "Canada men's national soccer team" first.
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


# --------------------------------------------------------------------------- #
# Building the corpus                                                          #
# --------------------------------------------------------------------------- #


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

    if corpus_path.exists() and not force:
        print(f"Corpus already present at {corpus_path} (use --force to refetch).")
        return corpus_path

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

    # Two different queries can resolve to the same article -- a player's search
    # name and a redirect, say. Indexing it twice would let one page occupy
    # several slots in a top-k result and crowd out genuine alternatives.
    seen: set[int] = set()
    unique: list[WikiPage] = []
    duplicates = 0
    for page in sorted(pages, key=lambda p: (p.entity_type, p.title)):
        if page.page_id in seen:
            duplicates += 1
            continue
        seen.add(page.page_id)
        unique.append(page)

    with corpus_path.open("w", encoding="utf-8") as handle:
        for page in unique:
            handle.write(page.to_json() + "\n")

    log_path = out_dir / RESOLUTION_LOG_FILE
    log_path.write_text(
        json.dumps(
            sorted(log, key=lambda e: (e["status"], e["query"])), ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

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
    pages: list[WikiPage] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            record["sections"] = [Section(**s) for s in record["sections"]]
            pages.append(WikiPage(**record))
    return pages
