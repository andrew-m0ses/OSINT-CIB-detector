"""
Temporal coordination detection (v2).

1. Co-posting with statistical significance (Poisson test)
2. Shift correlation: matching active/inactive hourly cycles
3. Burst synchronization: correlated daily activity spikes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import stats as sp_stats

from .ingest import Dataset, Post

logger = logging.getLogger(__name__)


@dataclass
class TemporalPairScore:
    account_a: str
    account_b: str
    copost_score: float = 0.0
    copost_count: int = 0
    copost_pvalue: float = 1.0       # Poisson p-value (lower = more significant)
    shift_correlation: float = 0.0
    burst_correlation: float = 0.0
    combined: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}


def _hourly_histogram(posts: list[Post]) -> np.ndarray:
    h = np.zeros(24, dtype=np.float64)
    for p in posts:
        h[p.timestamp.hour] += 1
    s = h.sum()
    return h / s if s > 0 else h


def _daily_series(posts: list[Post], min_ts: float, max_ts: float) -> np.ndarray:
    n = max(int((max_ts - min_ts) / 86400) + 1, 1)
    s = np.zeros(n, dtype=np.float64)
    for p in posts:
        idx = min(int((p.timestamp.timestamp() - min_ts) / 86400), n - 1)
        s[idx] += 1
    return s


def _copost_with_significance(
    posts_a: list[Post], posts_b: list[Post], window: float
) -> tuple[float, int, float]:
    """
    Count co-posts within window, compute expected count under uniform
    null hypothesis, and return Poisson survival p-value.

    Returns: (score, count, p_value)
    """
    ts_a = sorted(p.timestamp.timestamp() for p in posts_a)
    ts_b = sorted(p.timestamp.timestamp() for p in posts_b)
    if not ts_a or not ts_b:
        return 0.0, 0, 1.0

    # Count co-occurrences
    count = 0
    j = 0
    for ta in ts_a:
        while j < len(ts_b) and ts_b[j] < ta - window:
            j += 1
        k = j
        while k < len(ts_b) and ts_b[k] <= ta + window:
            count += 1
            k += 1

    span = max(ts_a[-1], ts_b[-1]) - min(ts_a[0], ts_b[0])
    if span <= 0:
        return 0.0, count, 1.0

    # Expected co-occurrences under uniform-random null
    expected = len(ts_a) * len(ts_b) * (2 * window) / span

    if expected <= 0:
        return 0.0, count, 1.0

    # Poisson survival function: P(X >= count) where X ~ Poisson(expected)
    # This gives us a proper p-value
    p_value = float(sp_stats.poisson.sf(count - 1, expected))  # sf(k-1) = P(X >= k)

    # Score: convert to 0-1 based on significance
    # -log10(p_value) maps p=0.05 -> 1.3, p=0.001 -> 3, p=1e-10 -> 10
    if p_value <= 0:
        sig_score = 1.0
    elif p_value >= 0.05:
        sig_score = 0.0  # not significant
    else:
        sig_score = min(-np.log10(p_value) / 5.0, 1.0)  # scale: p=1e-5 -> 1.0

    return sig_score, count, p_value


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    r, _ = sp_stats.pearsonr(a, b)
    return max(0.0, float(r))


def analyze_temporal(
    dataset: Dataset,
    window_seconds: float = 300,
    min_posts: int = 5,
    weights: dict | None = None,
) -> list[TemporalPairScore]:
    w = weights or {"copost": 0.5, "shift": 0.25, "burst": 0.25}
    eligible = {aid: a for aid, a in dataset.accounts.items() if a.post_count >= min_posts}
    ids = sorted(eligible)
    if len(ids) < 2:
        return []

    all_posts = dataset.get_all_posts()
    min_ts = min(p.timestamp.timestamp() for p in all_posts)
    max_ts = max(p.timestamp.timestamp() for p in all_posts)

    logger.info("Temporal: %d accounts, %d pairs", len(ids), len(ids) * (len(ids) - 1) // 2)
    results = []
    for a_id, b_id in combinations(ids, 2):
        pa, pb = eligible[a_id].posts, eligible[b_id].posts
        cp_score, cp_count, cp_pval = _copost_with_significance(pa, pb, window_seconds)
        shift = _safe_pearson(_hourly_histogram(pa), _hourly_histogram(pb))
        burst = _safe_pearson(_daily_series(pa, min_ts, max_ts), _daily_series(pb, min_ts, max_ts))
        combined = w["copost"] * cp_score + w["shift"] * shift + w["burst"] * burst
        results.append(TemporalPairScore(a_id, b_id, cp_score, cp_count, cp_pval, shift, burst, combined))

    results.sort(key=lambda x: x.combined, reverse=True)
    return results
