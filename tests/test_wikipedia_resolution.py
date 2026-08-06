"""Tests for Wikipedia entity resolution and the resulting corpus.

Every case in the first half is a bug that actually shipped into a corpus
during development. Resolution failures are quiet by nature -- a wrong article
is well written, on-topic, and indistinguishable from a right one once it is
sitting in the index -- so each one is pinned here rather than left to a code
comment.

The first crawl produced, among 327 pages:

  "2022 FIFA World Cup Group H"  ->  "2026 FIFA World Cup Group A"
  "2022 FIFA World Cup Group E"  ->  "2030 FIFA World Cup"
  "Through ball"                 ->  "2026 FIFA World Cup Group C"
  "Counter-pressing"             ->  "Jürgen Klopp"
  "Route One football"           ->  "Route (gridiron football)"

Nothing downstream would have flagged any of them. A question about Group H
would have been answered, with a citation, from an article about a tournament
that had not happened when the event data was recorded.

The guard functions are pure, so these tests need no network and no fixtures.
The corpus tests at the bottom skip when the corpus has not been built.
"""

from __future__ import annotations

import re

import pytest

from tactistat.config import load_config
from tactistat.data.wikipedia import (
    _content_tokens,
    _is_footballer_bio,
    _is_other_football_code,
    _split_sections,
    _years_conflict,
    load_corpus,
)

# --------------------------------------------------------------------------- #
# Guards                                                                       #
# --------------------------------------------------------------------------- #


class TestYearConflict:
    """The check that separates the 2022 World Cup from the 2026 one.

    No topical heuristic can catch this substitution: both articles are about
    football, both are World Cup group pages, both are well written. Only the
    year distinguishes them.
    """

    def test_rejects_a_different_tournament_edition(self):
        assert _years_conflict("2022 FIFA World Cup Group H", "2026 FIFA World Cup Group A")
        assert _years_conflict("2022 FIFA World Cup Group E", "2030 FIFA World Cup")

    def test_accepts_the_matching_edition(self):
        assert not _years_conflict("2022 FIFA World Cup Group H", "2022 FIFA World Cup Group H")

    def test_ignores_a_year_only_the_title_has(self):
        """A player's article carries their birth year; the query does not.

        Treating that as a conflict would reject "Fred" ->
        "Fred (footballer, born 1993)", which is the correct resolution.
        """
        assert not _years_conflict("Fred", "Fred (footballer, born 1993)")
        assert not _years_conflict("Antony", "Antony (footballer, born 2000)")


class TestOtherFootballCodes:
    def test_rejects_gridiron(self):
        assert _is_other_football_code(
            {
                "title": "Route (gridiron football)",
                "extract": "A route is a pattern run by a receiver.",
            }
        )

    def test_rejects_american_football_in_the_body(self):
        assert _is_other_football_code(
            {"title": "Wingback", "extract": "In American football, a wingback is a position ..."}
        )

    def test_accepts_association_football(self):
        assert not _is_other_football_code(
            {
                "title": "Tiki-taka",
                "extract": "Tiki-taka is a style of play in association football ...",
            }
        )


class TestFootballerBio:
    def test_accepts_the_standard_opening(self):
        opening = "Lionel Messi is an Argentine professional footballer who plays as a forward."
        assert _is_footballer_bio({"extract": opening})

    def test_accepts_a_double_space_before_player(self):
        """Regression: this rejected two real USA internationals.

        Stripped wikilinks leave double spaces in the plaintext extract, and
        both Tim Ream's and Tyler Adams's articles read "American professional
        soccer  player". A substring match on the single-spaced phrase failed,
        so the resolver fell through to search and then gave up.
        """
        assert _is_footballer_bio(
            {
                "extract": "Timothy Michael Ream (born October 5, 1987) is an American "
                "professional soccer  player who plays as a center-back."
            }
        )

    def test_rejects_a_non_footballer(self):
        assert not _is_footballer_bio(
            {"extract": "Fred Astaire was an American dancer, actor, singer and choreographer."}
        )


class TestContentTokens:
    def test_strips_domain_generic_words(self):
        """ "football" appears in nearly every page here, so it carries no signal.

        Leaving it in would let any football article satisfy any football query.
        """
        assert _content_tokens("Canada national football team") == {"canada"}

    def test_concept_queries_keep_their_distinctive_words(self):
        assert _content_tokens("Through ball") == {"through", "ball"}

    def test_a_wrong_concept_resolution_shares_no_tokens(self):
        """ "Through ball" -> "2026 FIFA World Cup Group C" from the first crawl."""
        query = _content_tokens("Through ball")
        title = _content_tokens("2026 FIFA World Cup Group C")
        assert not (query <= title)

    def test_a_correct_concept_resolution_is_a_subset(self):
        query = _content_tokens("Set piece (association football)")
        title = _content_tokens("Set piece (football)")
        assert query <= title


# --------------------------------------------------------------------------- #
# Section splitting                                                            #
# --------------------------------------------------------------------------- #


class TestSectionSplitting:
    @pytest.fixture
    def config(self):
        return load_config(load_env=False)

    def test_intro_becomes_its_own_section(self, config):
        text = "Intro paragraph that is long enough to survive the minimum length filter. " * 3
        sections = _split_sections(text, config)
        assert sections[0].heading == "Introduction"
        assert sections[0].level == 0

    def test_headings_become_sections(self, config):
        body = "Body text long enough to clear the minimum section length filter. " * 3
        text = f"Intro. {body}\n== Club career ==\n{body}\n=== Barcelona ===\n{body}"
        headings = [(s.heading, s.level) for s in _split_sections(text, config)]
        assert ("Club career", 2) in headings
        assert ("Barcelona", 3) in headings

    def test_dropped_sections_take_their_subsections_with_them(self, config):
        """ "References" owning a "=== Cited works ===" must remove both.

        Reference lists match any query containing a player's name, so a single
        leaked one can outrank the prose that could actually answer a question.
        """
        body = "Body text long enough to clear the minimum section length filter. " * 3
        text = (
            f"Intro. {body}\n== Career ==\n{body}\n"
            f"== References ==\n{body}\n=== Cited works ===\n{body}\n"
            f"== Honours ==\n{body}"
        )
        headings = [s.heading for s in _split_sections(text, config)]
        assert "Career" in headings
        assert "Honours" in headings, "a section after a dropped one must come back"
        assert "References" not in headings
        assert "Cited works" not in headings

    def test_trivially_short_sections_are_discarded(self, config):
        text = "Intro long enough to be kept as a real section of prose. " * 4 + "\n== Stub ==\nx"
        assert [s.heading for s in _split_sections(text, config)] == ["Introduction"]


# --------------------------------------------------------------------------- #
# The built corpus                                                             #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def corpus():
    config = load_config(load_env=False)
    try:
        return load_corpus(config)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def test_corpus_covers_every_entity_type(corpus):
    types = {page.entity_type for page in corpus}
    assert types == {"team", "player", "tournament", "concept"}


def test_no_out_of_scope_tournament_pages(corpus):
    """The corpus must not contain a World Cup other than the configured one.

    This is the single most damaging contamination available: a page about a
    different edition answers a question about this one fluently and wrongly.
    """
    config = load_config(load_env=False)
    season = str(config["dataset.season_name"])
    intruders = [
        page.title
        for page in corpus
        if re.search(r"\b(19|20)\d{2}\s+FIFA World Cup", page.title) and season not in page.title
    ]
    assert not intruders, f"out-of-scope tournament pages: {intruders}"


def test_all_thirty_two_teams_resolved(corpus):
    assert len({page.entity_key for page in corpus if page.entity_type == "team"}) == 32


def test_group_stage_pages_are_all_present(corpus):
    titles = {page.title for page in corpus}
    for group in "ABCDEFGH":
        assert any(f"Group {group}" in title for title in titles), f"Group {group} missing"


def test_every_page_carries_attribution(corpus):
    """CC BY-SA requires attribution, and citation accuracy is a scored metric."""
    for page in corpus:
        assert page.url.startswith("https://")
        assert page.revision_id > 0
        assert page.license
        assert page.sections


def test_navigational_sections_never_reach_the_corpus(corpus):
    config = load_config(load_env=False)
    dropped = {name.lower() for name in config["wikipedia.drop_sections"]}
    leaked = [
        (page.title, section.heading)
        for page in corpus
        for section in page.sections
        if section.heading.lower() in dropped
    ]
    assert not leaked, f"navigational sections leaked: {leaked[:5]}"


def test_known_players_resolved_to_the_right_article(corpus):
    """Spot-check the resolutions that the name mapping makes non-obvious."""
    by_key = {page.entity_key: page.title for page in corpus if page.entity_type == "player"}
    expected = {
        "Lionel Andrés Messi Cuccittini": "Lionel Messi",
        "Kylian Mbappé Lottin": "Kylian Mbappé",
        "Richarlison de Andrade": "Richarlison",
    }
    for statsbomb_name, wikipedia_title in expected.items():
        assert by_key.get(statsbomb_name) == wikipedia_title


def test_north_american_teams_use_their_soccer_titles(corpus):
    """USA and Canada file under "soccer", not "football".

    "United States national football team" is a disambiguation page pointing at
    gridiron. Resolving these needs the search fallback, which is what the
    duplicated-hint bug broke.
    """
    titles = {page.title for page in corpus if page.entity_type == "team"}
    assert "United States men's national soccer team" in titles
    assert "Canada men's national soccer team" in titles
