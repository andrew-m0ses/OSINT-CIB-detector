"""
Main CIB detector (v2) — orchestrates all signal layers.

Pipeline:
  1. Per-account anomaly scoring (behavioral red flags)
  2. Temporal pair analysis (co-posting with significance, shift, burst)
  3. Content pair analysis (semantic similarity, narrative convergence, URL, hashtag)
  4. Cascade chain analysis (amplification chains)
  5. Network construction + community detection (weighted combination of all signals)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .anomaly import AnomalyScore, analyze_anomalies
from .cascade import CascadePairScore, analyze_cascades
from .content import ContentPairScore, analyze_content
from .ingest import Dataset, load_file
from .network import NetworkResult, analyze_network
from .temporal import TemporalPairScore, analyze_temporal

logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    # Temporal
    copost_window_seconds: float = 300
    min_posts: int = 5
    temporal_weights: dict = field(default_factory=lambda: {"copost": 0.5, "shift": 0.25, "burst": 0.25})
    # Content
    url_window_seconds: float = 3600
    content_weights: dict = field(default_factory=lambda: {
        "url": 0.25, "semantic": 0.25, "narrative": 0.20,
        "hashtag": 0.15, "amplification": 0.15,
    })
    # Cascade
    cascade_chain_window: float = 600
    # Network
    edge_threshold: float = 0.15
    temporal_graph_weight: float = 0.35
    content_graph_weight: float = 0.35
    cascade_graph_weight: float = 0.30
    cluster_resolution: float = 1.0

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class DetectionResult:
    dataset_name: str
    account_count: int
    post_count: int
    temporal_pairs: list[TemporalPairScore]
    content_pairs: list[ContentPairScore]
    cascade_pairs: list[CascadePairScore]
    anomaly_scores: dict[str, AnomalyScore]
    network: NetworkResult
    config: DetectionConfig
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "accounts": self.account_count,
            "posts": self.post_count,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "config": self.config.to_dict(),
            "clusters": [c.to_dict() for c in self.network.clusters],
            "top_temporal_pairs": [p.to_dict() for p in self.temporal_pairs[:50]],
            "top_content_pairs": [p.to_dict() for p in self.content_pairs[:50]],
            "top_cascade_pairs": [p.to_dict() for p in self.cascade_pairs[:50]],
            "anomaly_scores": {aid: s.to_dict() for aid, s in self.anomaly_scores.items()},
            "graph": self.network.get_graph_data(),
        }

    def summary(self) -> str:
        lines = [
            f"=== CIB Detection Report: {self.dataset_name} ===",
            f"Accounts: {self.account_count}  |  Posts: {self.post_count}  |  Time: {self.elapsed_seconds:.1f}s",
            "",
        ]

        # Anomaly summary
        suspicious = [(aid, s) for aid, s in self.anomaly_scores.items() if s.combined > 0.5]
        if suspicious:
            suspicious.sort(key=lambda x: x[1].combined, reverse=True)
            lines.append(f"Behavioral anomalies: {len(suspicious)} suspicious account(s)")
            for aid, s in suspicious[:5]:
                flags = []
                if s.regularity > 0.5: flags.append("regular-intervals")
                if s.sleep_gap > 0.5: flags.append("no-sleep-gap")
                if s.entropy_score > 0.5: flags.append("low-hour-entropy")
                if s.activity_rate > 0.5: flags.append("high-rate")
                lines.append(f"  {aid}: {s.combined:.2f} [{', '.join(flags) or 'mixed'}]")
            lines.append("")

        # Clusters
        if not self.network.clusters:
            lines.append("No coordinated clusters detected.")
        else:
            lines.append(f"Detected {len(self.network.clusters)} cluster(s):\n")
            for c in self.network.clusters:
                flag = "HIGH" if c.coordination_score > 1.0 else "MEDIUM" if c.coordination_score > 0.5 else "LOW"
                lines.append(f"  Cluster {c.cluster_id}  [{flag}]  score={c.coordination_score:.3f}")
                lines.append(f"    Accounts ({c.size}): {', '.join(c.accounts)}")
                lines.append(f"    Density: {c.density:.2f}  |  Mean edge weight: {c.mean_edge_weight:.3f}")
                lines.append(f"    Temporal: {c.evidence.get('mean_temporal', 0):.3f}  |  Content: {c.evidence.get('mean_content', 0):.3f}  |  Cascade: {c.evidence.get('mean_cascade', 0):.3f}")

                # Mean anomaly score for cluster members
                member_anomaly = [self.anomaly_scores[a].combined for a in c.accounts if a in self.anomaly_scores]
                if member_anomaly:
                    lines.append(f"    Mean anomaly: {sum(member_anomaly)/len(member_anomaly):.3f}")
                lines.append("")

        # Top pairs
        if self.temporal_pairs:
            lines.append("Top 10 temporal pairs:")
            for p in self.temporal_pairs[:10]:
                lines.append(f"  {p.account_a} <-> {p.account_b}  score={p.combined:.3f}  (copost={p.copost_score:.2f} p={p.copost_pvalue:.1e}, shift={p.shift_correlation:.2f}, burst={p.burst_correlation:.2f})")
            lines.append("")

        if self.content_pairs:
            lines.append("Top 10 content pairs:")
            for p in self.content_pairs[:10]:
                lines.append(f"  {p.account_a} <-> {p.account_b}  score={p.combined:.3f}  (semantic={p.semantic_similarity:.2f}, narrative={p.narrative_convergence:.2f}, url={p.url_score:.2f}, hashtag={p.hashtag_overlap:.2f})")
            lines.append("")

        if self.cascade_pairs:
            lines.append("Top 10 cascade pairs:")
            for p in self.cascade_pairs[:10]:
                lines.append(f"  {p.account_a} <-> {p.account_b}  score={p.combined:.3f}  (mutual={p.mutual_amplification}, chains={p.chain_participation})")

        return "\n".join(lines)


def detect(dataset: Dataset, config: DetectionConfig | None = None) -> DetectionResult:
    if config is None:
        config = DetectionConfig()

    t0 = time.time()
    logger.info("Starting CIB detection on '%s': %d accounts, %d posts",
                dataset.name, len(dataset.accounts), dataset.total_posts)

    # 1. Per-account anomaly scoring
    anomalies = analyze_anomalies(dataset, min_posts=config.min_posts)

    # 2. Temporal analysis
    temporal = analyze_temporal(
        dataset,
        window_seconds=config.copost_window_seconds,
        min_posts=config.min_posts,
        weights=config.temporal_weights,
    )

    # 3. Content analysis (semantic + narrative)
    content = analyze_content(
        dataset,
        url_window=config.url_window_seconds,
        min_posts=config.min_posts,
        weights=config.content_weights,
    )

    # 4. Cascade chain analysis
    cascades = analyze_cascades(
        dataset,
        chain_window=config.cascade_chain_window,
        min_posts=config.min_posts,
    )

    # 5. Network analysis (combines all pairwise signals + anomaly weights)
    net = analyze_network(
        temporal, content, cascades, anomalies,
        edge_threshold=config.edge_threshold,
        temporal_weight=config.temporal_graph_weight,
        content_weight=config.content_graph_weight,
        cascade_weight=config.cascade_graph_weight,
        resolution=config.cluster_resolution,
    )

    elapsed = time.time() - t0
    return DetectionResult(
        dataset_name=dataset.name,
        account_count=len(dataset.accounts),
        post_count=dataset.total_posts,
        temporal_pairs=temporal,
        content_pairs=content,
        cascade_pairs=cascades,
        anomaly_scores=anomalies,
        network=net,
        config=config,
        elapsed_seconds=elapsed,
    )


def detect_file(path: str, config: DetectionConfig | None = None) -> DetectionResult:
    dataset = load_file(path)
    return detect(dataset, config)
