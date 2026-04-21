"""
Per-account behavioral anomaly detection.

Scores each account for non-human behavioral patterns:
1. Posting regularity: unnaturally even intervals between posts
2. Sleep gap: absence of natural circadian rest periods
3. Burst ratio: fraction of activity in concentrated bursts
4. Entropy: low posting-hour entropy suggests automated scheduling
5. Activity rate: posts per active day (abnormally high = suspicious)
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .ingest import Account, Dataset

logger = logging.getLogger(__name__)


@dataclass
class AnomalyScore:
    account_id: str
    regularity: float = 0.0     # how regular the inter-post intervals are (0=random, 1=clockwork)
    sleep_gap: float = 0.0      # 0=normal sleep, 1=no sleep pattern (posts 24/7)
    burst_ratio: float = 0.0    # fraction of posts in top 10% busiest hours
    entropy_score: float = 0.0  # 0=diverse hours, 1=concentrated in few hours (inverted)
    activity_rate: float = 0.0  # posts per active day, normalized
    combined: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}


def _interval_regularity(timestamps: list[float]) -> float:
    """
    Measure how regular the intervals between posts are.
    Perfectly regular = coefficient of variation near 0 = score near 1.
    Random = high CV = score near 0.
    """
    if len(timestamps) < 10:
        return 0.0

    ts = sorted(timestamps)
    intervals = np.diff(ts)
    # Remove very short intervals (< 10s, likely API artifacts)
    intervals = intervals[intervals > 10]
    if len(intervals) < 5:
        return 0.0

    mean_int = intervals.mean()
    std_int = intervals.std()
    if mean_int <= 0:
        return 0.0

    cv = std_int / mean_int  # coefficient of variation
    # Normal users: CV ~ 1.5-3.0. Bots: CV ~ 0.1-0.5
    # Map: CV < 0.3 -> 1.0, CV > 2.0 -> 0.0
    score = max(0.0, min(1.0, (2.0 - cv) / 1.7))
    return float(score)


def _sleep_gap_score(hours: list[int]) -> float:
    """
    Check if there's a natural sleep gap (consecutive hours with no/few posts).
    Normal humans have 5-8 consecutive quiet hours. Bots/shift-workers don't.
    """
    if len(hours) < 20:
        return 0.0

    hist = np.zeros(24, dtype=np.float64)
    for h in hours:
        hist[h] += 1
    total = hist.sum()
    if total == 0:
        return 0.0

    hist /= total

    # Find longest consecutive run of "quiet" hours (< 2% of posts)
    quiet_threshold = 0.02
    doubled = list(hist) + list(hist)  # wrap around midnight
    max_gap = 0
    current_gap = 0
    for val in doubled:
        if val < quiet_threshold:
            current_gap += 1
            max_gap = max(max_gap, current_gap)
        else:
            current_gap = 0

    max_gap = min(max_gap, 24)  # cap at 24

    # Normal: gap of 5-8 hours. No gap: suspicious.
    # gap >= 5 -> score 0 (normal), gap == 0 -> score 1 (no sleep)
    if max_gap >= 5:
        return 0.0
    return float(1.0 - max_gap / 5.0)


def _burst_ratio(timestamps: list[float]) -> float:
    """Fraction of posts in the busiest 10% of hours."""
    if len(timestamps) < 20:
        return 0.0

    # Bin into 1-hour buckets
    ts = np.array(timestamps)
    min_ts = ts.min()
    hours = ((ts - min_ts) / 3600).astype(int)
    counts = Counter(hours)

    if not counts:
        return 0.0

    values = sorted(counts.values(), reverse=True)
    n_bins = len(values)
    top_10_pct = max(1, n_bins // 10)
    top_sum = sum(values[:top_10_pct])
    total = sum(values)

    ratio = top_sum / total if total > 0 else 0.0
    # Normal: ~0.2-0.4 in top 10%. Bursty: >0.6
    return float(max(0.0, min(1.0, (ratio - 0.3) / 0.4)))


def _hour_entropy(hours: list[int]) -> float:
    """
    Shannon entropy of posting hours. Low entropy = concentrated = suspicious.
    Returns inverted score: 0 = diverse (normal), 1 = concentrated (suspicious).
    """
    if len(hours) < 10:
        return 0.0

    hist = np.zeros(24, dtype=np.float64)
    for h in hours:
        hist[h] += 1
    total = hist.sum()
    if total == 0:
        return 0.0

    hist /= total
    eps = 1e-10
    entropy = -np.sum(hist * np.log2(hist + eps))
    max_entropy = np.log2(24)  # uniform distribution

    # Normalize: 0 = max entropy (diverse), 1 = low entropy (concentrated)
    return float(max(0.0, 1.0 - entropy / max_entropy))


def _activity_rate(timestamps: list[float]) -> float:
    """Posts per active day, normalized to 0-1 suspicious score."""
    if len(timestamps) < 5:
        return 0.0

    ts = sorted(timestamps)
    span_days = (ts[-1] - ts[0]) / 86400
    if span_days < 1:
        return 0.0

    # Count active days
    day_set = set(int((t - ts[0]) / 86400) for t in ts)
    active_days = len(day_set)
    rate = len(timestamps) / max(active_days, 1)

    # Normal: 5-20 posts/day. Suspicious: >50/day
    return float(max(0.0, min(1.0, (rate - 20) / 40)))


def score_account(account: Account) -> AnomalyScore:
    """Compute anomaly score for a single account."""
    timestamps = [p.timestamp.timestamp() for p in account.posts]
    hours = [p.timestamp.hour for p in account.posts]

    reg = _interval_regularity(timestamps)
    sleep = _sleep_gap_score(hours)
    burst = _burst_ratio(timestamps)
    entropy = _hour_entropy(hours)
    rate = _activity_rate(timestamps)

    # Combined: weighted average
    combined = 0.25 * reg + 0.25 * sleep + 0.15 * burst + 0.20 * entropy + 0.15 * rate

    return AnomalyScore(
        account_id=account.account_id,
        regularity=reg,
        sleep_gap=sleep,
        burst_ratio=burst,
        entropy_score=entropy,
        activity_rate=rate,
        combined=combined,
    )


def analyze_anomalies(
    dataset: Dataset, min_posts: int = 5,
) -> dict[str, AnomalyScore]:
    """Score all accounts for behavioral anomalies."""
    scores = {}
    for aid, acct in dataset.accounts.items():
        if acct.post_count >= min_posts:
            scores[aid] = score_account(acct)

    n_suspicious = sum(1 for s in scores.values() if s.combined > 0.5)
    logger.info("Anomaly: %d accounts scored, %d suspicious (>0.5)", len(scores), n_suspicious)
    return scores
