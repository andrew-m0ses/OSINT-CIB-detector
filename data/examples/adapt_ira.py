"""
Adapter: FiveThirtyEight IRA dataset -> CIB detector format.

Takes the Clemson/538 Russian troll tweets CSV and:
1. Samples accounts across known operational categories (RightTroll, LeftTroll, etc.)
2. Converts to the CIB detector's expected schema
3. Runs detection
4. Evaluates: do detected clusters align with known IRA operational roles?

Usage:
    python adapt_ira.py IRAhandle_tweets_1.csv [--accounts 50] [--max-per-account 300]
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cib.ingest import Dataset, Post
from cib.detector import DetectionConfig, detect


HASHTAG_RE = re.compile(r"#(\w+)")
URL_RE = re.compile(r"https?://[^\s]+")


def load_ira_csv(path: str, max_per_account: int = 300, target_accounts: int = 50, seed: int = 42):
    """
    Load IRA CSV, sample accounts from each category, convert to Dataset.
    Returns (Dataset, ground_truth dict mapping account_id -> category).
    """
    rng = random.Random(seed)

    # First pass: index accounts by category
    account_cats: dict[str, str] = {}
    account_posts: dict[str, list[dict]] = defaultdict(list)

    print(f"Loading {path}...")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            lang = row.get("language", "")
            if lang != "English":
                continue

            author = row.get("author", "").strip()
            if not author:
                continue

            cat = row.get("account_category", "Unknown")
            account_cats[author] = cat
            account_posts[author].append(row)

    # Count per category
    cat_accounts: dict[str, list[str]] = defaultdict(list)
    for acct, cat in account_cats.items():
        if len(account_posts[acct]) >= 20:  # minimum tweets
            cat_accounts[cat].append(acct)

    print("\nAccounts with 20+ English tweets by category:")
    for cat, accts in sorted(cat_accounts.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(accts)} accounts")

    # Sample accounts: try to get representation from each category
    selected = []
    # Allocate proportionally but ensure at least 3 from each category with enough accounts
    cats_with_enough = {c: a for c, a in cat_accounts.items() if len(a) >= 3 and c != "Unknown"}
    total_available = sum(len(a) for a in cats_with_enough.values())

    for cat, accts in cats_with_enough.items():
        n = max(3, int(target_accounts * len(accts) / total_available))
        n = min(n, len(accts))
        selected.extend(rng.sample(accts, n))

    # Trim to target if overshot
    if len(selected) > target_accounts:
        selected = rng.sample(selected, target_accounts)

    print(f"\nSelected {len(selected)} accounts:")
    sel_cats = Counter(account_cats[a] for a in selected)
    for cat, n in sel_cats.most_common():
        print(f"  {cat}: {n}")

    # Build dataset
    ds = Dataset(name="ira_real")
    ground_truth = {}

    for author in selected:
        ground_truth[author] = account_cats[author]
        posts = account_posts[author]

        # Sample if too many
        if len(posts) > max_per_account:
            posts = rng.sample(posts, max_per_account)

        for row in posts:
            content = row.get("content", "")
            hashtags = HASHTAG_RE.findall(content)
            urls = URL_RE.findall(content)

            # Parse retweet info
            repost_of = None
            if row.get("retweet", "0") == "1" and content.startswith("RT @"):
                match = re.match(r"RT @(\w+)", content)
                if match:
                    rt_user = match.group(1).upper()
                    if rt_user in selected:
                        repost_of = rt_user

            ds.add_post(Post(
                account_id=author,
                timestamp=row.get("publish_date", "2017-01-01"),
                text=content,
                url=urls[0] if urls else None,
                hashtags=hashtags,
                repost_of=repost_of,
                platform="twitter",
            ))

    return ds, ground_truth


def evaluate(result, ground_truth: dict[str, str]):
    """Compare detected clusters against known IRA categories."""
    print("\n" + "=" * 60)
    print("EVALUATION: Detected clusters vs. IRA operational roles")
    print("=" * 60)

    for cluster in result.network.clusters:
        cat_counts = Counter(ground_truth.get(a, "?") for a in cluster.accounts)
        dominant_cat = cat_counts.most_common(1)[0] if cat_counts else ("?", 0)
        purity = dominant_cat[1] / cluster.size if cluster.size > 0 else 0

        level = "HIGH" if cluster.coordination_score > 1.0 else "MEDIUM" if cluster.coordination_score > 0.5 else "LOW"

        print(f"\n  Cluster {cluster.cluster_id} [{level}] score={cluster.coordination_score:.3f}")
        print(f"    Size: {cluster.size} | Density: {cluster.density:.2f}")
        print(f"    Temporal: {cluster.evidence.get('mean_temporal', 0):.3f} | Content: {cluster.evidence.get('mean_content', 0):.3f}")
        print(f"    Category breakdown:")
        for cat, n in cat_counts.most_common():
            marker = " <-- dominant" if cat == dominant_cat[0] else ""
            print(f"      {cat}: {n}{marker}")
        print(f"    Purity: {purity:.1%} (fraction from dominant category)")

    # Overall stats
    all_clustered = set()
    for c in result.network.clusters:
        all_clustered.update(c.accounts)

    print(f"\n  Summary:")
    print(f"    Total accounts: {result.account_count}")
    print(f"    Clustered: {len(all_clustered)}")
    print(f"    Clusters found: {len(result.network.clusters)}")

    # Check if categories got separated
    cat_cluster_map: dict[str, list[int]] = defaultdict(list)
    for c in result.network.clusters:
        for acct in c.accounts:
            cat = ground_truth.get(acct, "?")
            cat_cluster_map[cat].append(c.cluster_id)

    print(f"\n  Category spread across clusters:")
    for cat in sorted(cat_cluster_map):
        clusters = Counter(cat_cluster_map[cat])
        spread = ", ".join(f"C{cid}({n})" for cid, n in clusters.most_common())
        print(f"    {cat}: {spread}")


def main():
    parser = argparse.ArgumentParser(description="Run CIB detector on IRA dataset")
    parser.add_argument("csv_path", help="Path to IRAhandle_tweets_*.csv")
    parser.add_argument("--accounts", type=int, default=50, help="Number of accounts to sample")
    parser.add_argument("--max-per-account", type=int, default=300, help="Max tweets per account")
    parser.add_argument("--window", type=float, default=300, help="Co-post window (seconds)")
    parser.add_argument("--threshold", type=float, default=0.12, help="Edge threshold")
    parser.add_argument("--output", "-o", help="Save JSON report")
    args = parser.parse_args()

    ds, ground_truth = load_ira_csv(args.csv_path, args.max_per_account, args.accounts)
    print(f"\nDataset: {len(ds.accounts)} accounts, {ds.total_posts} posts")

    config = DetectionConfig(
        copost_window_seconds=args.window,
        min_posts=10,
        edge_threshold=args.threshold,
    )

    print("\nRunning CIB detection...")
    result = detect(ds, config)
    print(result.summary())
    evaluate(result, ground_truth)

    if args.output:
        report = result.to_dict()
        report["ground_truth"] = ground_truth
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
