This OSINT tool detects coordinated inauthentic behavior (CIB) in social media datasets. Given a collection of posts from multiple accounts, it identifies clusters of accounts operating as a coordinated network.

Unlike persona-matching tools that ask "are these the same person?", CIB detection asks "are these accounts acting in concert--i.e., working from a shared playbook or troll farm?"

---
<img width="1718" height="480" alt="Screenshot 2026-04-18 at 6 08 18 PM" src="https://github.com/user-attachments/assets/d6e746c8-5e0b-4b6b-aca7-97f67b1dd5c5" />

---
<img width="1716" height="897" alt="Screenshot 2026-04-21 at 8 49 48 AM" src="https://github.com/user-attachments/assets/ed41719b-6d57-4274-98c5-394f2c10d250" />

---
<img width="1715" height="527" alt="Screenshot 2026-04-18 at 6 14 26 PM" src="https://github.com/user-attachments/assets/4b7baeb9-e13a-4da8-a7e6-ca4afb686d15" />

---

**how it works**

Five signal layers are combined into a weighted coordination graph:

temporal coordination
- co-posting with statistical significance (Poisson test with p-values)
- shift correlation: matching active/inactive hourly cycles (detects shift-work patterns)
- burst synchronization: correlated daily activity spikes (a fantastic paper for reference: https://www.nature.com/articles/s41467-025-64785-1)

content coordination (augmented by VIGINUM's D3lta and sentence-transformers)
- copypasta detection: near-exact duplicate posts across accounts
- rewording detection: same meaning, different phrasing
- cross-language translation: same content posted in different languages
- narrative convergence: topic distribution similarity via embedding clusters
- URL co-sharing within time windows
- hashtag co-usage (Jaccard overlap)

behavioral anomaly scoring (per-account)
- posting interval regularity
- sleep gap absence (no circadian rest pattern)
- hour entropy (concentrated posting schedule)
- burst ratio and activity rate

cascade / amplification chain detection
- retweet/reply chain mapping
- mutual amplification scoring
- chain co-participation frequency

network topology
- builds weighted graph from all pairwise signals
- anomaly-boosted edge weights (suspicious accounts amplify connections)
- Louvain community detection for cluster identification

D3lta integration: optionally uses [VIGINUM's D3lta library](https://github.com/VIGINUM-FR/D3lta) for content duplication detection. D3lta is an open-source tool built by France's foreign digital interference agency (VIGINUM) that classifies duplicate content as copypasta, rewording, or translation. When installed (`pip install git+https://github.com/VIGINUM-FR/D3lta.git`), D3lta replaces the built-in content similarity engine with Faiss-accelerated semantic search and cross-language matching. The tool works without D3lta using sentence-transformers as a fallback.

---

**quickstart**

```bash
pip install -r requirements.txt

# start the web UI
python -m cib serve
```

open `http://localhost:8000` — upload a file or click **Load IRA demo dataset** to run detection on real Internet Research Agency troll data.

for D3lta support (optional):
```bash
pip install git+https://github.com/VIGINUM-FR/D3lta.git
```

for sentence-transformer support (optional, ~80MB model download on first run):
```bash
pip install sentence-transformers
```

---

**CLI**

```bash
# detect coordination in a dataset
python -m cib detect data.csv
python -m cib detect data.csv --window 600 --threshold 0.2 --min-posts 10 -o report.json

# start web server
python -m cib serve --port 8000

# generate synthetic test data
python -m cib generate --accounts 40 --coordinated 12 --groups 2 --posts 150 -o synthetic.csv
```

---

**input format**

CSV, JSON array, or JSONL. Recognized columns/keys:

| field | aliases | required |
|-------|---------|----------|
| `account_id` | `user_id`, `author` | yes |
| `timestamp` | `created_at`, `date` | yes |
| `text` | `content`, `body` | yes |
| `url` | `link` | no |
| `hashtags` | `tags` | no (comma-separated or list) |
| `reply_to` | `in_reply_to` | no (account id) |
| `repost_of` | `retweeted_user` | no (account id) |
| `platform` | | no |

---

**demo dataset**

The bundled demo is 40 real accounts (5,417 tweets) from the Internet Research Agency--a Russian state-backed troll farm based in St. Petersburg, indicted by the Mueller investigation in 2018. The dataset comes from Clemson University researchers via FiveThirtyEight.

The accounts are organized into known operational roles--RightTroll, LeftTroll, HashtagGamer, Fearmonger, NonEnglish--which the detector recovers without being told about them.

---

**output**

coordination score interpretation:
-  1.0 — HIGH: strong coordination signal, likely inauthentic
-  0.5 – 1.0 — MEDIUM: possible coordination, warrants investigation
-  <0.5 — LOW: weak signal, may be organic overlap

Each cluster reports density, mean edge weight, and per-signal breakdowns (temporal, content, cascade, anomaly).

---

**project structure**

```
cib/
├── __init__.py
├── __main__.py
├── ingest.py       data models + CSV/JSON/JSONL loading
├── temporal.py     temporal coordination (Poisson co-posting, shift, burst)
├── content.py      content coordination (D3lta / sentence-transformers / TF-IDF)
├── anomaly.py      per-account behavioral anomaly scoring
├── cascade.py      amplification chain detection
├── network.py      graph construction + Louvain community detection
├── detector.py     main orchestrator (all 5 layers)
├── synthetic.py    test data generator
├── cli.py          command-line interface
├── server.py       FastAPI REST server
└── static/
    └── index.html  web dashboard

data/examples/
├── ira_demo.json           bundled IRA dataset (40 accounts)
├── ira_ground_truth.json   known operational roles
└── adapt_ira.py            adapter for full FiveThirtyEight IRA dataset
```

---

**API**

Start with `python -m cib serve`. Interactive docs at `http://localhost:8000/docs`.

| method | path | description |
|--------|------|-------------|
| POST | `/api/detect` | upload file, run detection |
| POST | `/api/generate` | load IRA demo dataset + detect |
| GET | `/api/result` | latest detection result |
| GET | `/api/graph` | graph data for visualization |

---

**license**

MIT
