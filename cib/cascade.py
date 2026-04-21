"""
Cascade / amplification chain detection.

Analyzes repost and reply chains to find coordinated amplification:
1. Chain detection: A posts -> B reposts -> C reposts -> D replies
2. Chain frequency: how often the same set of accounts form chains
3. Speed: how quickly cascades propagate (unnaturally fast = coordinated)
4. Directionality: asymmetric amplification (many -> one = boosting)
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .ingest import Dataset, Post

logger = logging.getLogger(__name__)


@dataclass
class CascadeEvent:
    """A single amplification event: account B amplified account A's content."""
    source: str       # account being amplified
    amplifier: str    # account doing the amplifying
    timestamp: float
    event_type: str   # 'repost' or 'reply'


@dataclass
class CascadePairScore:
    account_a: str
    account_b: str
    mutual_amplification: int = 0     # total events in both directions
    directional_bias: float = 0.0     # 0=balanced, 1=completely one-way
    cascade_speed: float = 0.0        # mean time between source post and amplification
    chain_participation: int = 0       # times they appear in same cascade chain
    combined: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}


def _extract_events(dataset: Dataset) -> list[CascadeEvent]:
    """Extract all amplification events from the dataset."""
    events = []
    account_ids = set(dataset.account_ids)

    for acct in dataset.accounts.values():
        for post in acct.posts:
            if post.repost_of and post.repost_of in account_ids:
                events.append(CascadeEvent(
                    source=post.repost_of,
                    amplifier=post.account_id,
                    timestamp=post.timestamp.timestamp(),
                    event_type="repost",
                ))
            if post.reply_to and post.reply_to in account_ids:
                events.append(CascadeEvent(
                    source=post.reply_to,
                    amplifier=post.account_id,
                    timestamp=post.timestamp.timestamp(),
                    event_type="reply",
                ))

    events.sort(key=lambda e: e.timestamp)
    return events


def _build_pair_stats(events: list[CascadeEvent]) -> dict[tuple[str, str], dict]:
    """Build per-pair amplification statistics."""
    pair_events: dict[tuple[str, str], list[CascadeEvent]] = defaultdict(list)

    for ev in events:
        key = tuple(sorted([ev.source, ev.amplifier]))
        pair_events[key].append(ev)

    pair_stats = {}
    for key, evts in pair_events.items():
        a, b = key
        a_to_b = sum(1 for e in evts if e.source == a)
        b_to_a = sum(1 for e in evts if e.source == b)
        total = a_to_b + b_to_a

        if total == 0:
            continue

        # Directional bias: 0 = perfectly balanced, 1 = completely one-way
        bias = abs(a_to_b - b_to_a) / total

        pair_stats[key] = {
            "total": total,
            "a_to_b": a_to_b,
            "b_to_a": b_to_a,
            "bias": bias,
            "events": evts,
        }

    return pair_stats


def _detect_chains(events: list[CascadeEvent], chain_window: float = 600) -> list[list[CascadeEvent]]:
    """
    Detect cascade chains: sequences of amplification events on the same
    content within a time window.

    A chain is: A posts something -> B reposts within window -> C reposts within window
    """
    if not events:
        return []

    chains: list[list[CascadeEvent]] = []

    # Group events by source
    by_source: dict[str, list[CascadeEvent]] = defaultdict(list)
    for ev in events:
        by_source[ev.source].append(ev)

    # For each source, find cascading amplification sequences
    for source, source_events in by_source.items():
        source_events.sort(key=lambda e: e.timestamp)

        # Sliding window: group events within chain_window
        i = 0
        while i < len(source_events):
            chain = [source_events[i]]
            j = i + 1
            while j < len(source_events) and (source_events[j].timestamp - source_events[i].timestamp) <= chain_window:
                chain.append(source_events[j])
                j += 1

            if len(chain) >= 2:
                chains.append(chain)
            i = j if j > i + 1 else i + 1

    return chains


def _chain_participation(chains: list[list[CascadeEvent]]) -> dict[tuple[str, str], int]:
    """Count how often each pair of accounts appears in the same chain."""
    pair_counts: Counter = Counter()

    for chain in chains:
        participants = set(e.amplifier for e in chain)
        for a, b in combinations(sorted(participants), 2):
            pair_counts[(a, b)] += 1

    return dict(pair_counts)


def analyze_cascades(
    dataset: Dataset,
    chain_window: float = 600,
    min_posts: int = 5,
    weights: dict | None = None,
) -> list[CascadePairScore]:
    """
    Analyze amplification cascades across all account pairs.
    """
    w = weights or {"mutual": 0.35, "bias": 0.15, "chain": 0.50}

    events = _extract_events(dataset)
    if not events:
        logger.info("Cascade: no amplification events found")
        return []

    logger.info("Cascade: %d amplification events", len(events))

    pair_stats = _build_pair_stats(events)
    chains = _detect_chains(events, chain_window)
    chain_pairs = _chain_participation(chains)

    logger.info("Cascade: %d chains detected, %d pair relationships", len(chains), len(pair_stats))

    # Get all account pairs that have any cascade signal
    all_pairs = set(pair_stats.keys()) | set(chain_pairs.keys())

    eligible = {aid for aid, a in dataset.accounts.items() if a.post_count >= min_posts}
    results = []

    for key in all_pairs:
        a, b = key
        if a not in eligible or b not in eligible:
            continue

        stats = pair_stats.get(key, {"total": 0, "bias": 0.0})
        chain_count = chain_pairs.get(key, 0)

        total = stats.get("total", 0)
        bias = stats.get("bias", 0.0)

        # Normalize mutual amplification: >10 events is very high
        mutual_norm = min(total / 10.0, 1.0)

        # Chain participation: >3 chains together is suspicious
        chain_norm = min(chain_count / 3.0, 1.0)

        # Invert bias for scoring: balanced amplification is MORE suspicious
        # (one-way could just be a fan; mutual = coordination)
        balance_score = 1.0 - bias

        combined = (
            w["mutual"] * mutual_norm * balance_score
            + w["bias"] * balance_score
            + w["chain"] * chain_norm
        )

        results.append(CascadePairScore(
            account_a=a,
            account_b=b,
            mutual_amplification=total,
            directional_bias=bias,
            cascade_speed=0.0,  # TODO: compute from event timestamps
            chain_participation=chain_count,
            combined=combined,
        ))

    results.sort(key=lambda x: x.combined, reverse=True)
    return results
