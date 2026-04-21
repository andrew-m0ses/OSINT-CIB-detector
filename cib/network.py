"""
Network topology analysis (v2).

Builds weighted coordination graph from temporal + content + cascade pair scores,
uses per-account anomaly scores to boost edges between suspicious accounts,
then detects clusters via community detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Cluster:
    cluster_id: int
    accounts: list[str]
    mean_edge_weight: float = 0.0
    density: float = 0.0
    coordination_score: float = 0.0
    evidence: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.accounts)

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "accounts": self.accounts,
            "size": self.size,
            "mean_edge_weight": round(self.mean_edge_weight, 4),
            "density": round(self.density, 4),
            "coordination_score": round(self.coordination_score, 4),
            "evidence": self.evidence,
        }


@dataclass
class NetworkResult:
    graph: nx.Graph
    clusters: list[Cluster]
    edges: list[dict]

    def to_dict(self) -> dict:
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "clusters": [c.to_dict() for c in self.clusters],
        }

    def get_graph_data(self) -> dict:
        nodes = []
        acct_cluster = {}
        for c in self.clusters:
            for a in c.accounts:
                acct_cluster[a] = c.cluster_id

        for n in self.graph.nodes(data=True):
            nodes.append({
                "id": n[0],
                "cluster": acct_cluster.get(n[0], -1),
                "degree": self.graph.degree(n[0]),
                "anomaly": round(n[1].get("anomaly", 0), 3),
            })
        edges = []
        for u, v, d in self.graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "weight": round(d.get("weight", 0), 4),
                "temporal": round(d.get("temporal", 0), 4),
                "content": round(d.get("content", 0), 4),
                "cascade": round(d.get("cascade", 0), 4),
            })
        return {"nodes": nodes, "edges": edges}


def build_graph(
    temporal_pairs: list,
    content_pairs: list,
    cascade_pairs: list | None = None,
    anomaly_scores: dict | None = None,
    edge_threshold: float = 0.15,
    temporal_weight: float = 0.35,
    content_weight: float = 0.35,
    cascade_weight: float = 0.30,
) -> nx.Graph:
    t_idx = {}
    for p in temporal_pairs:
        key = tuple(sorted([p.account_a, p.account_b]))
        t_idx[key] = p.combined

    c_idx = {}
    for p in content_pairs:
        key = tuple(sorted([p.account_a, p.account_b]))
        c_idx[key] = p.combined

    cas_idx = {}
    if cascade_pairs:
        for p in cascade_pairs:
            key = tuple(sorted([p.account_a, p.account_b]))
            cas_idx[key] = p.combined

    all_keys = set(t_idx) | set(c_idx) | set(cas_idx)
    G = nx.Graph()

    for key in all_keys:
        t_score = t_idx.get(key, 0.0)
        c_score = c_idx.get(key, 0.0)
        cas_score = cas_idx.get(key, 0.0)

        # Base edge weight
        if cascade_pairs:
            weight = temporal_weight * t_score + content_weight * c_score + cascade_weight * cas_score
        else:
            # No cascade data: reweight temporal + content
            weight = (temporal_weight + cascade_weight / 2) * t_score + (content_weight + cascade_weight / 2) * c_score

        # Anomaly boost: if both endpoints are behaviorally suspicious, boost the edge
        if anomaly_scores:
            a_anom = anomaly_scores.get(key[0])
            b_anom = anomaly_scores.get(key[1])
            if a_anom and b_anom:
                # Geometric mean of anomaly scores as multiplier (1.0 to 1.5)
                boost = 1.0 + 0.5 * np.sqrt(a_anom.combined * b_anom.combined)
                weight *= boost

        if weight >= edge_threshold:
            G.add_edge(key[0], key[1],
                       weight=weight, temporal=t_score, content=c_score, cascade=cas_score)

    # Add anomaly scores as node attributes
    if anomaly_scores:
        for aid, score in anomaly_scores.items():
            if aid in G.nodes:
                G.nodes[aid]["anomaly"] = score.combined

    logger.info("Graph: %d nodes, %d edges (threshold=%.2f)", G.number_of_nodes(), G.number_of_edges(), edge_threshold)
    return G


def detect_clusters(G: nx.Graph, resolution: float = 1.0) -> list[set[str]]:
    if G.number_of_nodes() == 0:
        return []
    try:
        communities = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=42)
    except Exception:
        communities = list(nx.connected_components(G))
    return [c for c in communities if len(c) >= 2]


def score_cluster(G: nx.Graph, members: set[str]) -> Cluster:
    sub = G.subgraph(members)
    n = len(members)
    m = sub.number_of_edges()
    max_edges = n * (n - 1) / 2 if n > 1 else 1

    weights = [d["weight"] for _, _, d in sub.edges(data=True)]
    mean_w = float(np.mean(weights)) if weights else 0.0
    density = m / max_edges if max_edges > 0 else 0.0

    coord = density * mean_w * np.log2(max(n, 2))

    t_scores = [d.get("temporal", 0) for _, _, d in sub.edges(data=True)]
    c_scores = [d.get("content", 0) for _, _, d in sub.edges(data=True)]
    cas_scores = [d.get("cascade", 0) for _, _, d in sub.edges(data=True)]
    anom_scores = [G.nodes[n].get("anomaly", 0) for n in members if "anomaly" in G.nodes.get(n, {})]

    evidence = {
        "mean_temporal": round(float(np.mean(t_scores)), 4) if t_scores else 0.0,
        "mean_content": round(float(np.mean(c_scores)), 4) if c_scores else 0.0,
        "mean_cascade": round(float(np.mean(cas_scores)), 4) if cas_scores else 0.0,
        "mean_anomaly": round(float(np.mean(anom_scores)), 4) if anom_scores else 0.0,
        "internal_edges": m,
        "max_possible_edges": int(max_edges),
    }

    return Cluster(
        cluster_id=0,
        accounts=sorted(members),
        mean_edge_weight=mean_w,
        density=density,
        coordination_score=float(coord),
        evidence=evidence,
    )


def analyze_network(
    temporal_pairs: list,
    content_pairs: list,
    cascade_pairs: list | None = None,
    anomaly_scores: dict | None = None,
    edge_threshold: float = 0.15,
    temporal_weight: float = 0.35,
    content_weight: float = 0.35,
    cascade_weight: float = 0.30,
    resolution: float = 1.0,
) -> NetworkResult:
    G = build_graph(
        temporal_pairs, content_pairs, cascade_pairs, anomaly_scores,
        edge_threshold, temporal_weight, content_weight, cascade_weight,
    )
    raw_clusters = detect_clusters(G, resolution)

    clusters = []
    for i, members in enumerate(raw_clusters):
        c = score_cluster(G, members)
        c.cluster_id = i
        clusters.append(c)

    clusters.sort(key=lambda c: c.coordination_score, reverse=True)

    edges = []
    for u, v, d in G.edges(data=True):
        edges.append({"source": u, "target": v, **{k: round(v_, 4) if isinstance(v_, float) else v_ for k, v_ in d.items()}})

    return NetworkResult(graph=G, clusters=clusters, edges=edges)
