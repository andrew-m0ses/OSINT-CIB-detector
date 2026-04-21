"""
Data models and ingestion for CIB detection.

Supports CSV, JSONL, JSON array. Columns/keys accepted:
  account_id | user_id | author
  timestamp  | created_at | date
  text       | content | body
  url        | link
  hashtags   | tags          (comma-separated string or list)
  reply_to   | in_reply_to   (account id)
  repost_of  | retweeted_user
  post_id    | id
  platform
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dateutil import parser as dateparser


@dataclass
class Post:
    account_id: str
    timestamp: datetime
    text: str = ""
    url: str | None = None
    hashtags: list[str] = field(default_factory=list)
    reply_to: str | None = None
    repost_of: str | None = None
    post_id: str | None = None
    platform: str = "unknown"
    raw: dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = dateparser.parse(self.timestamp)
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
        if isinstance(self.hashtags, str):
            self.hashtags = [
                h.strip().lower().lstrip("#") for h in self.hashtags.split(",") if h.strip()
            ]


@dataclass
class Account:
    account_id: str
    posts: list[Post] = field(default_factory=list)
    platform: str = "unknown"
    metadata: dict = field(default_factory=dict)

    @property
    def post_count(self) -> int:
        return len(self.posts)

    @property
    def first_seen(self) -> Optional[datetime]:
        return min((p.timestamp for p in self.posts), default=None)

    @property
    def last_seen(self) -> Optional[datetime]:
        return max((p.timestamp for p in self.posts), default=None)


@dataclass
class Dataset:
    accounts: dict[str, Account] = field(default_factory=dict)
    name: str = "untitled"

    @property
    def total_posts(self) -> int:
        return sum(a.post_count for a in self.accounts.values())

    @property
    def account_ids(self) -> list[str]:
        return list(self.accounts.keys())

    def add_post(self, post: Post):
        if post.account_id not in self.accounts:
            self.accounts[post.account_id] = Account(
                account_id=post.account_id, platform=post.platform,
            )
        self.accounts[post.account_id].posts.append(post)

    def get_all_posts(self) -> list[Post]:
        out = []
        for a in self.accounts.values():
            out.extend(a.posts)
        return sorted(out, key=lambda p: p.timestamp)


def _parse_row(row: dict) -> Post:
    return Post(
        account_id=str(row.get("account_id") or row.get("user_id") or row.get("author") or "unknown"),
        timestamp=row.get("timestamp") or row.get("created_at") or row.get("date") or "2000-01-01",
        text=str(row.get("text") or row.get("content") or row.get("body") or ""),
        url=row.get("url") or row.get("link") or None,
        hashtags=row.get("hashtags") or row.get("tags") or [],
        reply_to=row.get("reply_to") or row.get("in_reply_to") or None,
        repost_of=row.get("repost_of") or row.get("retweeted_user") or None,
        post_id=str(row.get("post_id") or row.get("id") or ""),
        platform=str(row.get("platform") or "unknown"),
        raw=row,
    )


def load_csv(path: str | Path) -> Dataset:
    path = Path(path)
    ds = Dataset(name=path.stem)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            ds.add_post(_parse_row(row))
    return ds


def load_jsonl(path: str | Path) -> Dataset:
    path = Path(path)
    ds = Dataset(name=path.stem)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                ds.add_post(_parse_row(json.loads(line)))
    return ds


def load_json(path: str | Path) -> Dataset:
    path = Path(path)
    ds = Dataset(name=path.stem)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    if isinstance(data, list):
        for row in data:
            ds.add_post(_parse_row(row))
    return ds


def load_file(path: str | Path) -> Dataset:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        return load_csv(path)
    elif ext == ".jsonl":
        return load_jsonl(path)
    elif ext == ".json":
        return load_json(path)
    else:
        try:
            return load_json(path)
        except (json.JSONDecodeError, KeyError):
            return load_csv(path)
