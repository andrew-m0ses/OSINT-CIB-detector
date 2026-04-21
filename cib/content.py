"""
Content coordination detection (v3 — D3lta integration).

Uses VIGINUM's D3lta library for content duplication detection when available:
- Copypasta: near-exact duplicates
- Rewording: same meaning, different phrasing
- Translation: same content across languages

Falls back to sentence-transformers or TF-IDF if D3lta is not installed.

Additional signals (always available):
- URL co-sharing within time windows
- Hashtag co-usage (Jaccard overlap)
- Amplification (mutual repost/reply)
- Narrative convergence (topic distribution similarity)
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .ingest import Dataset, Post

logger = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://[^\s)<>\"]+")
HASHTAG_RE = re.compile(r"#(\w+)")

# ── D3lta availability check ────────────────────────────────────────────────

_d3lta_available = None


def _check_d3lta() -> bool:
    global _d3lta_available
    if _d3lta_available is None:
        try:
            from d3lta.faissd3lta import semantic_faiss
            _d3lta_available = True
            logger.info("D3lta library detected — using for content analysis")
        except Exception:
            _d3lta_available = False
            logger.info("D3lta not available — using built-in content analysis")
    return _d3lta_available


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ContentPairScore:
    account_a: str
    account_b: str
    url_score: float = 0.0
    url_cocount: int = 0
    semantic_similarity: float = 0.0
    near_dupes: int = 0
    copypasta_count: int = 0
    rewording_count: int = 0
    translation_count: int = 0
    narrative_convergence: float = 0.0
    hashtag_overlap: float = 0.0
    amplification: float = 0.0
    combined: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}


# ── D3lta-based content analysis ─────────────────────────────────────────────

@dataclass
class D3ltaResults:
    """Pre-computed D3lta results for all posts."""
    pair_copypasta: dict = field(default_factory=dict)   # (a,b) -> count
    pair_rewording: dict = field(default_factory=dict)
    pair_translation: dict = field(default_factory=dict)
    pair_total_dupes: dict = field(default_factory=dict)
    pair_mean_score: dict = field(default_factory=dict)


def _run_d3lta(dataset: Dataset, min_posts: int = 5) -> D3ltaResults:
    """
    Run D3lta on all posts, then aggregate duplicates per account pair.
    """
    import pandas as pd
    from d3lta.faissd3lta import semantic_faiss

    # Build DataFrame with all posts, tracking account ownership
    rows = []
    post_to_account = {}
    idx = 0
    for aid, acct in dataset.accounts.items():
        if acct.post_count < min_posts:
            continue
        posts = [p for p in acct.posts if len(p.text) > 15]
        # Sample if too many
        if len(posts) > 250:
            sampled_idx = np.random.choice(len(posts), 250, replace=False)
            posts = [posts[i] for i in sampled_idx]
        for p in posts:
            post_to_account[str(idx)] = aid
            rows.append({"original": p.text})
            idx += 1

    if len(rows) < 10:
        return D3ltaResults()

    df = pd.DataFrame(rows)
    df.index = df.index.astype(str)

    logger.info("D3lta: analyzing %d posts for duplicates...", len(df))

    try:
        matches, df_clusters = semantic_faiss(
            df=df,
            min_size_txt=15,
            threshold_grapheme=0.693,
            threshold_language=0.715,
            threshold_semantic=0.80,
        )
    except Exception as e:
        logger.warning("D3lta analysis failed: %s", e)
        return D3ltaResults()

    if matches is None or len(matches) == 0:
        logger.info("D3lta: no duplicates found")
        return D3ltaResults()

    logger.info("D3lta: found %d duplicate pairs", len(matches))

    # Aggregate by account pair
    result = D3ltaResults()

    for _, row in matches.iterrows():
        src = str(row.get("source", ""))
        tgt = str(row.get("target", ""))
        acct_a = post_to_account.get(src)
        acct_b = post_to_account.get(tgt)

        if not acct_a or not acct_b or acct_a == acct_b:
            continue  # skip self-matches

        key = tuple(sorted([acct_a, acct_b]))
        dup_type = row.get("dup_type", "unknown")
        score = row.get("score", 0.0)

        result.pair_total_dupes[key] = result.pair_total_dupes.get(key, 0) + 1

        if dup_type == "copy-pasta":
            result.pair_copypasta[key] = result.pair_copypasta.get(key, 0) + 1
        elif dup_type == "rewording":
            result.pair_rewording[key] = result.pair_rewording.get(key, 0) + 1
        elif dup_type == "translation":
            result.pair_translation[key] = result.pair_translation.get(key, 0) + 1

        # Running mean score
        prev = result.pair_mean_score.get(key, (0.0, 0))
        result.pair_mean_score[key] = (prev[0] + score, prev[1] + 1)

    # Finalize mean scores
    result.pair_mean_score = {
        k: v[0] / v[1] if v[1] > 0 else 0.0
        for k, v in result.pair_mean_score.items()
    }

    logger.info("D3lta: %d account pairs with shared content", len(result.pair_total_dupes))
    return result


# ── Built-in fallback (sentence-transformers / TF-IDF) ───────────────────────

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformer model...")
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Model loaded.")
        except Exception as e:
            logger.warning("Could not load sentence-transformer (%s), falling back to TF-IDF", e)
            _model = "tfidf_fallback"
    return _model


def _compute_account_embeddings(dataset: Dataset, min_posts: int = 5) -> dict[str, np.ndarray]:
    model = _get_model()

    all_texts = []
    text_map = []
    for aid, acct in dataset.accounts.items():
        if acct.post_count < min_posts:
            continue
        texts = [p.text for p in acct.posts if len(p.text) > 20]
        if len(texts) < 3:
            continue
        if len(texts) > 300:
            idx = np.random.choice(len(texts), 300, replace=False)
            texts = [texts[i] for i in idx]
        for t in texts:
            text_map.append(aid)
            all_texts.append(t)

    if not all_texts:
        return {}

    if model == "tfidf_fallback":
        from sklearn.feature_extraction.text import TfidfVectorizer
        try:
            vec = TfidfVectorizer(max_features=3000, stop_words="english", min_df=2)
            all_embeds = vec.fit_transform(all_texts).toarray()
        except ValueError:
            return {}
    else:
        all_embeds = model.encode(all_texts, show_progress_bar=False, batch_size=64)

    account_embeds = {}
    for i, aid in enumerate(text_map):
        if aid not in account_embeds:
            account_embeds[aid] = []
        account_embeds[aid].append(all_embeds[i])

    return {aid: np.array(vecs) for aid, vecs in account_embeds.items()}


def _semantic_similarity_fallback(emb_a: np.ndarray, emb_b: np.ndarray, threshold: float = 0.75) -> tuple[float, int]:
    from sklearn.metrics.pairwise import cosine_similarity

    centroid_a = emb_a.mean(axis=0)
    centroid_b = emb_b.mean(axis=0)
    norm_a = np.linalg.norm(centroid_a)
    norm_b = np.linalg.norm(centroid_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0, 0

    centroid_sim = float(np.dot(centroid_a, centroid_b) / (norm_a * norm_b))

    sample_a = emb_a[:100] if len(emb_a) > 100 else emb_a
    sample_b = emb_b[:100] if len(emb_b) > 100 else emb_b
    sim_matrix = cosine_similarity(sample_a, sample_b)
    near_dupes = int((sim_matrix >= threshold).sum())

    score = max(0.0, centroid_sim)
    if near_dupes > 0:
        dupe_bonus = min(near_dupes / (len(sample_a) * 0.05), 0.3)
        score = min(score + dupe_bonus, 1.0)

    return score, near_dupes


def _narrative_convergence(emb_a: np.ndarray, emb_b: np.ndarray, n_topics: int = 8) -> float:
    from sklearn.cluster import MiniBatchKMeans

    combined = np.vstack([emb_a, emb_b])
    if len(combined) < n_topics * 2:
        n_topics = max(2, len(combined) // 4)

    kmeans = MiniBatchKMeans(n_clusters=n_topics, random_state=42, n_init=3, batch_size=256)
    labels = kmeans.fit_predict(combined)

    labels_a = labels[:len(emb_a)]
    labels_b = labels[len(emb_a):]

    dist_a = np.zeros(n_topics, dtype=np.float64)
    dist_b = np.zeros(n_topics, dtype=np.float64)
    for l in labels_a:
        dist_a[l] += 1
    for l in labels_b:
        dist_b[l] += 1

    sa, sb = dist_a.sum(), dist_b.sum()
    if sa > 0:
        dist_a /= sa
    if sb > 0:
        dist_b /= sb

    m = 0.5 * (dist_a + dist_b)
    eps = 1e-10
    kl_a = np.sum(dist_a * np.log((dist_a + eps) / (m + eps)))
    kl_b = np.sum(dist_b * np.log((dist_b + eps) / (m + eps)))
    jsd = 0.5 * (kl_a + kl_b)
    similarity = max(0.0, 1.0 - np.sqrt(max(0.0, jsd)))

    entropy_a = -np.sum(dist_a * np.log(dist_a + eps))
    entropy_b = -np.sum(dist_b * np.log(dist_b + eps))
    max_entropy = np.log(n_topics)
    avg_entropy = (entropy_a + entropy_b) / 2

    focus_factor = 1.0 - (avg_entropy / max_entropy) * 0.5
    return float(similarity * focus_factor)


# ── Shared helpers ───────────────────────────────────────────────────────────

def _urls_from(post: Post) -> list[str]:
    found = URL_RE.findall(post.text)
    if post.url:
        found.append(post.url)
    return [u.lower().rstrip(".,;:!?)").split("?")[0] for u in found]


def _tags_from(post: Post) -> set[str]:
    found = set(h.lower() for h in HASHTAG_RE.findall(post.text))
    found.update(h.lower().lstrip("#") for h in post.hashtags)
    return found


def _url_coscore(posts_a: list[Post], posts_b: list[Post], window: float = 3600) -> tuple[float, int]:
    idx_a: dict[str, list[float]] = defaultdict(list)
    idx_b: dict[str, list[float]] = defaultdict(list)
    for p in posts_a:
        for u in _urls_from(p):
            idx_a[u].append(p.timestamp.timestamp())
    for p in posts_b:
        for u in _urls_from(p):
            idx_b[u].append(p.timestamp.timestamp())

    shared = set(idx_a) & set(idx_b)
    if not shared:
        return 0.0, 0

    co = 0
    for url in shared:
        for ta in idx_a[url]:
            for tb in idx_b[url]:
                if abs(ta - tb) <= window:
                    co += 1

    total = len(set(idx_a) | set(idx_b))
    score = len(shared) / max(total, 1)
    if co > 0:
        score = min(score * (1 + np.log1p(co)), 1.0)
    return float(score), co


def _hashtag_overlap(posts_a: list[Post], posts_b: list[Post]) -> float:
    sa: set[str] = set()
    sb: set[str] = set()
    for p in posts_a:
        sa.update(_tags_from(p))
    for p in posts_b:
        sb.update(_tags_from(p))
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def _amplification(posts_a: list[Post], posts_b: list[Post]) -> float:
    a_id = posts_a[0].account_id if posts_a else ""
    b_id = posts_b[0].account_id if posts_b else ""
    fwd = sum(1 for p in posts_a if p.repost_of == b_id or p.reply_to == b_id)
    rev = sum(1 for p in posts_b if p.repost_of == a_id or p.reply_to == a_id)
    base = min(len(posts_a), len(posts_b))
    if base == 0:
        return 0.0
    return min((fwd + rev) / max(base * 0.1, 1), 1.0)


# ── Main analysis ────────────────────────────────────────────────────────────

def analyze_content(
    dataset: Dataset,
    url_window: float = 3600,
    min_posts: int = 5,
    weights: dict | None = None,
) -> list[ContentPairScore]:
    use_d3lta = _check_d3lta()

    if use_d3lta:
        w = {
            "d3lta": 0.35, "url": 0.20, "narrative": 0.15,
            "hashtag": 0.15, "amplification": 0.15,
        }
    else:
        w = weights or {
            "semantic": 0.25, "url": 0.25, "narrative": 0.20,
            "hashtag": 0.15, "amplification": 0.15,
        }

    eligible = {aid: a for aid, a in dataset.accounts.items() if a.post_count >= min_posts}
    ids = sorted(eligible)
    if len(ids) < 2:
        return []

    # Run D3lta or fallback embeddings
    d3lta_results = None
    account_embeds = None

    if use_d3lta:
        try:
            d3lta_results = _run_d3lta(dataset, min_posts)
        except Exception as e:
            logger.warning("D3lta runtime error (%s), falling back to built-in analysis", e)
            use_d3lta = False
            w = {"semantic": 0.25, "url": 0.25, "narrative": 0.20, "hashtag": 0.15, "amplification": 0.15}

    logger.info("Content: computing embeddings...")
    account_embeds = _compute_account_embeddings(dataset, min_posts)
    logger.info("Embeddings computed for %d accounts", len(account_embeds))

    logger.info("Content: scoring %d pairs", len(ids) * (len(ids) - 1) // 2)
    results = []

    for a_id, b_id in combinations(ids, 2):
        pa, pb = eligible[a_id].posts, eligible[b_id].posts
        us, uc = _url_coscore(pa, pb, url_window)
        ho = _hashtag_overlap(pa, pb)
        amp = _amplification(pa, pb)
        key = tuple(sorted([a_id, b_id]))

        cp_count, rw_count, tr_count = 0, 0, 0
        sem, nd, narr = 0.0, 0, 0.0

        if use_d3lta and d3lta_results:
            # D3lta scores
            cp_count = d3lta_results.pair_copypasta.get(key, 0)
            rw_count = d3lta_results.pair_rewording.get(key, 0)
            tr_count = d3lta_results.pair_translation.get(key, 0)
            total_dupes = d3lta_results.pair_total_dupes.get(key, 0)
            mean_score = d3lta_results.pair_mean_score.get(key, 0.0)

            # Convert D3lta counts to a 0-1 score
            # Any cross-account duplication is suspicious; >5 is very suspicious
            sem = min(total_dupes / 5.0, 1.0) if total_dupes > 0 else 0.0
            # Weight by mean similarity score from D3lta
            sem = sem * max(0.5, mean_score)
            nd = total_dupes

            # Narrative convergence from embeddings
            if account_embeds and a_id in account_embeds and b_id in account_embeds:
                narr = _narrative_convergence(account_embeds[a_id], account_embeds[b_id])

            combined = (
                w["d3lta"] * sem + w["url"] * us + w["narrative"] * narr
                + w["hashtag"] * ho + w["amplification"] * amp
            )
        else:
            # Fallback: use embeddings directly
            if account_embeds and a_id in account_embeds and b_id in account_embeds:
                ea, eb = account_embeds[a_id], account_embeds[b_id]
                sem, nd = _semantic_similarity_fallback(ea, eb)
                narr = _narrative_convergence(ea, eb)

            combined = (
                w["semantic"] * sem + w["url"] * us + w["narrative"] * narr
                + w["hashtag"] * ho + w["amplification"] * amp
            )

        results.append(ContentPairScore(
            account_a=a_id, account_b=b_id,
            url_score=us, url_cocount=uc,
            semantic_similarity=sem, near_dupes=nd,
            copypasta_count=cp_count, rewording_count=rw_count,
            translation_count=tr_count,
            narrative_convergence=narr, hashtag_overlap=ho,
            amplification=amp, combined=combined,
        ))

    results.sort(key=lambda x: x.combined, reverse=True)
    return results
