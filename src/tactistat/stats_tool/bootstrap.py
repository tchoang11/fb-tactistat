"""Confidence intervals for per-90 rates, by resampling matches.

A World Cup gives a player three to seven matches. "Messi 0.91 goals per 90 vs
Mbappe 1.21" reads like a settled comparison, and on that many matches it is
mostly noise: one more goal in one more game moves a rate by a tenth. The point
estimate is the same number the stats tool already reports; what this adds is
how far it would move if the tournament were replayed.

The resampling unit is the **match**, not the minute or the shot. A match is
what the tournament actually draws — a player either played Croatia or did not
— and it keeps the minutes and the events of a game together, which resampling
minutes would break.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MINUTES_PER_90 = 90.0


@dataclass(frozen=True)
class Interval:
    """A point estimate and the percentile interval around it."""

    point: float
    low: float
    high: float
    confidence: float
    n_matches: int
    n_resamples: int
    # For a difference: how many of those matches both players appeared in.
    # Those fixtures are drawn once and counted once, so the field also says
    # how far the paired estimand is from two independent samples.
    n_shared: int = 0

    @property
    def width(self) -> float:
        return self.high - self.low

    def excludes_zero(self) -> bool:
        """Whether the interval lies entirely on one side of zero."""
        return self.low > 0 or self.high < 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": self.point,
            "low": self.low,
            "high": self.high,
            "confidence": self.confidence,
            "n_matches": self.n_matches,
            "n_resamples": self.n_resamples,
            "n_shared": self.n_shared,
        }

    @property
    def label(self) -> str:
        """The confidence as a percentage, without inventing or losing precision.

        `{:.0%}` would print a 95.5% interval as "96% CI", which is a different
        interval. Trailing zeros are dropped so the ordinary case stays "95%".
        """
        percent = self.confidence * 100
        text = f"{percent:.2f}".rstrip("0").rstrip(".")
        # Rounding 99.999 up to "100%" names a confidence no interval can have.
        return (f"{percent:.6f}".rstrip("0").rstrip(".") if float(text) >= 100 else text) + "%"

    def render(self, digits: int = 2) -> str:
        return f"{self.point:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"


def bootstrap_settings(config: Any) -> tuple[int, float, int]:
    """Validate the bootstrap block before any resampling is done."""
    n_resamples = config.get("eval.bootstrap.n_resamples", 10_000)
    confidence = config.get("eval.bootstrap.confidence_level", 0.95)
    seed = config.get("eval.bootstrap.seed", 42)
    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool) or n_resamples < 100:
        raise ValueError("eval.bootstrap.n_resamples must be an integer of at least 100")
    if not isinstance(confidence, (int, float)) or not 0 < float(confidence) < 1:
        raise ValueError("eval.bootstrap.confidence_level must lie strictly between 0 and 1")
    # NumPy refuses a negative seed, and the stratified draw derives seed + 2,
    # so the check belongs here rather than three frames deeper.
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("eval.bootstrap.seed must be a non-negative integer")
    return n_resamples, float(confidence), seed


def _picks(n_units: int, n_resamples: int, seed: int) -> np.ndarray:
    """Row indices for `n_resamples` draws with replacement of `n_units` units."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_units, size=(n_resamples, n_units))


def _rate_from(drawn_totals: np.ndarray, drawn_minutes: np.ndarray) -> np.ndarray:
    """Per-90 rate from already-summed resample totals."""
    # A resample can draw only matches this player sat out; it carries no rate.
    played = drawn_minutes > 0
    return np.where(
        played, drawn_totals * MINUTES_PER_90 / np.where(played, drawn_minutes, 1), np.nan
    )


def _sums(
    totals: np.ndarray, minutes: np.ndarray, picks: np.ndarray | None, n_resamples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Totals and minutes each resample draws; zeros when the stratum is empty."""
    if picks is None:
        zeros = np.zeros(n_resamples)
        return zeros, zeros
    return totals[picks].sum(axis=1), minutes[picks].sum(axis=1)


def _rate(totals: np.ndarray, minutes: np.ndarray, picks: np.ndarray) -> np.ndarray:
    """Per-90 rate under each resample described by `picks`."""
    return _rate_from(*_sums(totals, minutes, picks, len(picks)))


def _draw(totals: np.ndarray, minutes: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    """Per-90 rate for `n_resamples` resamples of the same matches."""
    return _rate(totals, minutes, _picks(len(totals), n_resamples, seed))


def _percentiles(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    usable = samples[~np.isnan(samples)]
    if usable.size == 0:
        return math.nan, math.nan
    tail = (1.0 - confidence) / 2.0
    low, high = np.percentile(usable, [tail * 100, (1.0 - tail) * 100])
    return float(low), float(high)


def _check_metric(matches: pd.DataFrame, metric: str) -> None:
    if metric not in matches.columns:
        raise ValueError(f"{metric!r} is not a match-level column")


def _by_match(matches: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per fixture. The resampling unit is the match, so make it one."""
    _check_metric(matches, metric)
    if "match_id" not in matches.columns:
        return matches[[metric, "minutes"]]
    return matches.groupby("match_id", sort=True)[[metric, "minutes"]].sum()


def _columns(matches: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    per_match = _by_match(matches, metric)
    return (
        per_match[metric].to_numpy(dtype=float),
        per_match["minutes"].to_numpy(dtype=float),
    )


def _aligned(
    matches: pd.DataFrame, metric: str, fixtures: list[Any]
) -> tuple[np.ndarray, np.ndarray]:
    """The player's totals over `fixtures`, zero-filled where they did not play."""
    per_match = _by_match(matches, metric).reindex(fixtures).fillna(0.0)
    return (
        per_match[metric].to_numpy(dtype=float),
        per_match["minutes"].to_numpy(dtype=float),
    )


def per90_interval(
    matches: pd.DataFrame,
    metric: str,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Interval:
    """Resample one player's matches and bound their per-90 rate."""
    totals, minutes = _columns(matches, metric)
    played = minutes.sum()
    point = float(totals.sum() * MINUTES_PER_90 / played) if played > 0 else math.nan
    if len(totals) == 0:
        return Interval(math.nan, math.nan, math.nan, confidence, 0, n_resamples)
    low, high = _percentiles(_draw(totals, minutes, n_resamples, seed), confidence)
    return Interval(point, low, high, confidence, len(totals), n_resamples)


def difference_interval(
    left: pd.DataFrame,
    right: pd.DataFrame,
    metric: str,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
    paired: bool = True,
) -> Interval:
    """Bound the gap between two players' per-90 rates, paired on shared fixtures.

    The two schedules overlap. Messi and Mbappe both played the final, and that
    match moves both rates at once — the same 120 minutes, the same scoreline.
    Resampling the two players independently throws that correlation away and
    counts the shared fixture twice.

    This is a **stratified, partially paired** bootstrap, conditional on how the
    two schedules actually overlapped. Fixtures are split into three strata —
    both played, only the left player played, only the right player played — and
    each stratum is resampled with replacement at its own observed size. The
    shared draw is made once and handed to both players, so a shared fixture
    contributes to both rates in the same resample or to neither.

    Two consequences worth stating, because they are not what "paired" suggests:

    * The stratum sizes are held fixed. A pair with one shared fixture draws
      that stratum 1-of-1, so the shared match appears in *every* resample and
      contributes no variance at all — some of the narrowing for such pairs is
      that fixed contribution, not covariance.
    * Where the overlap is large the covariance dominates, and it need not be
      positive: two forwards in the same XI tend to split their team's goals, so
      pairing can leave the interval *wider* than treating them as independent.

    Holding stratum sizes fixed is what keeps each player's resample the size of
    their real schedule; drawing from the pooled union instead would let a
    player's minutes denominator swing between three matches and eleven, which
    widens the interval with noise the tournament never had.

    An interval that spans zero is the useful answer: it says these matches
    cannot order the two players, which the point estimate alone never admits.
    It bounds the rates the observed matches support, not the ordering of the
    per-90 numbers themselves, which are exact.

    `paired=False` resamples the two schedules independently, discarding the
    covariance. It is kept because a claim that pairing changes the answer has
    to be checkable against the estimator it replaced, not asserted.
    """
    _check_metric(left, metric)
    _check_metric(right, metric)
    for frame in (left, right):
        # Pairing by row position instead would silently compare a player's
        # third match to the other player's third match. Refuse rather than
        # invent an alignment.
        if not frame.empty and "match_id" not in frame.columns:
            raise ValueError("a difference interval needs a 'match_id' column to pair fixtures")
    if left.empty or right.empty:
        return Interval(math.nan, math.nan, math.nan, confidence, 0, n_resamples)

    left_ids = set(_by_match(left, metric).index)
    right_ids = set(_by_match(right, metric).index)
    shared = sorted(left_ids & right_ids) if paired else []
    left_only = sorted(left_ids - set(shared))
    right_only = sorted(right_ids - set(shared))

    left_totals, left_minutes = _columns(left, metric)
    right_totals, right_minutes = _columns(right, metric)
    left_played, right_played = left_minutes.sum(), right_minutes.sum()
    point = (
        float(
            left_totals.sum() * MINUTES_PER_90 / left_played
            - right_totals.sum() * MINUTES_PER_90 / right_played
        )
        if left_played > 0 and right_played > 0
        else math.nan
    )

    # One stream per stratum, so that with no shared fixtures the two players
    # are drawn from independent streams rather than coupled through indices.
    strata = (
        (shared, _picks(len(shared), n_resamples, seed) if shared else None),
        (left_only, _picks(len(left_only), n_resamples, seed + 1) if left_only else None),
        (right_only, _picks(len(right_only), n_resamples, seed + 2) if right_only else None),
    )
    (shared_ids, shared_picks), (left_ids_only, left_picks), (right_ids_only, right_picks) = strata

    def drawn(frame: pd.DataFrame, own: list[Any], own_picks: np.ndarray | None) -> np.ndarray:
        totals_shared, minutes_shared = _sums(
            *_aligned(frame, metric, shared_ids), shared_picks, n_resamples
        )
        totals_own, minutes_own = _sums(*_aligned(frame, metric, own), own_picks, n_resamples)
        return _rate_from(totals_shared + totals_own, minutes_shared + minutes_own)

    samples = drawn(left, left_ids_only, left_picks) - drawn(right, right_ids_only, right_picks)
    low, high = _percentiles(samples, confidence)
    # Unpaired, a fixture both played is drawn twice and counted twice, which
    # is exactly the estimand `paired=False` describes.
    counted = len(shared) + len(left_only) + len(right_only)
    return Interval(
        point,
        low,
        high,
        confidence,
        counted,
        n_resamples,
        n_shared=len(shared),
    )
