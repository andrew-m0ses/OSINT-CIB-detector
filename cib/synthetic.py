"""
Synthetic dataset generator for testing CIB detection.

Creates datasets with:
- Organic accounts (random independent behavior)
- Coordinated groups (synchronized timing, shared URLs, similar text, mutual amplification)
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta, timezone

from .ingest import Dataset, Post

TOPICS = [
    "election", "vaccine", "economy", "immigration", "climate",
    "protest", "scandal", "military", "trade", "technology",
]

PHRASES = {
    "election": [
        "The upcoming election is being stolen by corrupt officials",
        "Massive voter fraud detected in multiple states",
        "Don't trust the mainstream polls they are rigged",
        "Our candidate is the only hope for real change",
        "Wake up people the election system is broken",
        "Share this before they delete it — proof of fraud",
    ],
    "vaccine": [
        "New study shows alarming side effects being covered up",
        "Why are they forcing this on our children",
        "Big pharma doesn't want you to see this data",
        "Exposed: the real numbers they won't publish",
        "My friend took it and now has serious health issues",
        "The truth about what's really in these shots",
    ],
    "economy": [
        "The economy is about to collapse and they know it",
        "Inflation is way worse than what they're reporting",
        "The elites are hoarding wealth while we suffer",
        "Stock market crash imminent according to insider sources",
        "Your savings are worth nothing wake up",
    ],
    "immigration": [
        "Open borders are destroying our communities",
        "Millions of illegals flooding in and nobody cares",
        "Crime rates skyrocketing in border states",
        "They want cheap labor and don't care about you",
    ],
    "default": [
        "This is really important please share widely",
        "The mainstream media won't cover this story",
        "They don't want you to know the truth",
        "Wake up and share this with everyone you know",
        "Another day another cover up by the elites",
        "Something big is about to happen stay alert",
        "Can't believe more people aren't talking about this",
    ],
}

URLS = [
    "https://truthnews247.com/article/{id}",
    "https://realfacts.blog/post/{id}",
    "https://uncensored-report.com/{id}",
    "https://patriotwatch.net/breaking/{id}",
    "https://freedomdaily.org/expose/{id}",
]

ORGANIC_PHRASES = [
    "Just had the best coffee at this new place downtown",
    "Anyone watching the game tonight?",
    "Happy birthday to my amazing friend!",
    "Beautiful sunset today",
    "Can't wait for the weekend",
    "This recipe turned out amazing",
    "Just finished reading a great book",
    "Traffic is terrible today",
    "Looking forward to vacation next month",
    "New album is fire honestly",
    "Monday mood: need more coffee",
    "Trying a new workout routine this week",
    "Congratulations to the team!",
    "Great discussion at today's meetup",
    "This weather is perfect for a hike",
]


def _random_ts(base: datetime, jitter_hours: float = 24 * 30) -> datetime:
    """Random timestamp within jitter_hours of base."""
    delta = timedelta(hours=random.uniform(-jitter_hours, jitter_hours))
    return base + delta


def _coordinated_ts(base: datetime, anchor: datetime, jitter_seconds: float = 180) -> datetime:
    """Timestamp near anchor with small jitter (simulates coordination)."""
    delta = timedelta(seconds=random.gauss(0, jitter_seconds))
    return anchor + delta


def generate_dataset(
    n_organic: int = 22,
    n_coordinated: int = 8,
    posts_per_account: int = 100,
    n_coordinated_groups: int = 1,
    base_time: datetime | None = None,
    span_days: int = 30,
) -> Dataset:
    if base_time is None:
        base_time = datetime(2026, 3, 1, tzinfo=timezone.utc)

    ds = Dataset(name="synthetic")
    rng = random.Random(42)

    # --- organic accounts ---
    for i in range(n_organic):
        acct_id = f"organic_{i:03d}"
        # Random active hours (simulate real timezone)
        active_start = rng.randint(6, 14)  # wake hour
        active_end = active_start + rng.randint(10, 16)  # sleep hour

        for _ in range(posts_per_account):
            hour = rng.gauss((active_start + active_end) / 2, 3) % 24
            day_offset = rng.uniform(0, span_days)
            ts = base_time + timedelta(days=day_offset, hours=hour)

            text = rng.choice(ORGANIC_PHRASES)
            # Occasionally add some noise hashtags
            hashtags = []
            if rng.random() < 0.15:
                hashtags = [rng.choice(["fun", "life", "mood", "vibes", "love", "goals"])]

            ds.add_post(Post(
                account_id=acct_id,
                timestamp=ts,
                text=text,
                hashtags=hashtags,
                platform="synthetic",
            ))

    # --- coordinated groups ---
    per_group = max(n_coordinated // n_coordinated_groups, 2)
    coord_idx = 0

    for g in range(n_coordinated_groups):
        group_size = per_group if g < n_coordinated_groups - 1 else n_coordinated - coord_idx
        group_ids = [f"coord_g{g}_{j:02d}" for j in range(group_size)]
        coord_idx += group_size

        # Pick 1-2 topics for this group
        group_topics = rng.sample(TOPICS, min(2, len(TOPICS)))

        # Shared URL pool
        url_pool = []
        for _ in range(posts_per_account // 3):
            template = rng.choice(URLS)
            url_pool.append(template.format(id=rng.randint(1000, 9999)))

        # Shared hashtags
        group_hashtags = [f"{t}truth" for t in group_topics] + [
            rng.choice(["wakeup", "resist", "sharetruth", "exposed", "breaking"])
            for _ in range(3)
        ]

        # Generate coordinated posting waves
        n_waves = posts_per_account // 2  # half of posts are coordinated waves
        wave_anchors = [
            base_time + timedelta(days=rng.uniform(0, span_days), hours=rng.uniform(0, 24))
            for _ in range(n_waves)
        ]

        # Shared active window (simulates shift-work)
        shift_start = rng.randint(9, 15)
        shift_end = shift_start + rng.randint(6, 10)

        for acct_id in group_ids:
            # Coordinated wave posts
            for anchor in wave_anchors:
                ts = _coordinated_ts(base_time, anchor, jitter_seconds=120)
                topic = rng.choice(group_topics)
                phrases = PHRASES.get(topic, PHRASES["default"])
                text = rng.choice(phrases)

                # Often share same URL
                url = None
                if rng.random() < 0.5:
                    url = rng.choice(url_pool)

                hashtags = []
                if rng.random() < 0.6:
                    hashtags = rng.sample(group_hashtags, min(rng.randint(1, 3), len(group_hashtags)))

                # Mutual amplification
                reply_to = None
                repost_of = None
                if rng.random() < 0.2:
                    other = rng.choice([x for x in group_ids if x != acct_id])
                    if rng.random() < 0.5:
                        reply_to = other
                    else:
                        repost_of = other

                ds.add_post(Post(
                    account_id=acct_id,
                    timestamp=ts,
                    text=text,
                    url=url,
                    hashtags=hashtags,
                    reply_to=reply_to,
                    repost_of=repost_of,
                    platform="synthetic",
                ))

            # Independent filler posts (during shift hours)
            for _ in range(posts_per_account - n_waves):
                hour = rng.gauss((shift_start + shift_end) / 2, 2) % 24
                day_offset = rng.uniform(0, span_days)
                ts = base_time + timedelta(days=day_offset, hours=hour)
                text = rng.choice(PHRASES.get(rng.choice(group_topics), PHRASES["default"]))
                hashtags = []
                if rng.random() < 0.4:
                    hashtags = rng.sample(group_hashtags, min(rng.randint(1, 2), len(group_hashtags)))

                ds.add_post(Post(
                    account_id=acct_id,
                    timestamp=ts,
                    text=text,
                    hashtags=hashtags,
                    platform="synthetic",
                ))

    return ds


def write_csv(ds: Dataset, path: str):
    posts = ds.get_all_posts()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "account_id", "timestamp", "text", "url", "hashtags",
            "reply_to", "repost_of", "platform",
        ])
        w.writeheader()
        for p in posts:
            w.writerow({
                "account_id": p.account_id,
                "timestamp": p.timestamp.isoformat(),
                "text": p.text,
                "url": p.url or "",
                "hashtags": ",".join(p.hashtags),
                "reply_to": p.reply_to or "",
                "repost_of": p.repost_of or "",
                "platform": p.platform,
            })
