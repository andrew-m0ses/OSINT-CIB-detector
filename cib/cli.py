"""
CLI for CIB detection.

Usage:
    python -m cib detect data.csv
    python -m cib detect data.csv --threshold 0.2 --window 600 --output report.json
    python -m cib serve --port 8000
    python -m cib generate --accounts 30 --coordinated 8 --posts 200 --output synthetic.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .detector import DetectionConfig, detect_file


def cmd_detect(args):
    config = DetectionConfig(
        copost_window_seconds=args.window,
        min_posts=args.min_posts,
        edge_threshold=args.threshold,
    )

    result = detect_file(args.file, config)
    print(result.summary())

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        print(f"\nFull report saved to {args.output}")


def cmd_serve(args):
    from .server import create_app
    import uvicorn

    app = create_app()
    print(f"Starting CIB detector server at http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_generate(args):
    from .synthetic import generate_dataset, write_csv

    ds = generate_dataset(
        n_organic=args.accounts - args.coordinated,
        n_coordinated=args.coordinated,
        posts_per_account=args.posts,
        n_coordinated_groups=args.groups,
    )
    out = args.output or "synthetic.csv"
    write_csv(ds, out)
    print(f"Generated {ds.total_posts} posts from {len(ds.accounts)} accounts -> {out}")
    print(f"  Organic: {args.accounts - args.coordinated}, Coordinated: {args.coordinated} in {args.groups} group(s)")


def main():
    parser = argparse.ArgumentParser(prog="cib", description="CIB Detector — Coordinated Inauthentic Behavior detection")
    sub = parser.add_subparsers(dest="command")

    # detect
    p_det = sub.add_parser("detect", help="Run CIB detection on a dataset")
    p_det.add_argument("file", help="Path to CSV, JSON, or JSONL file")
    p_det.add_argument("--window", type=float, default=300, help="Co-post window in seconds (default 300)")
    p_det.add_argument("--threshold", type=float, default=0.15, help="Edge threshold (default 0.15)")
    p_det.add_argument("--min-posts", type=int, default=5, help="Min posts per account (default 5)")
    p_det.add_argument("--output", "-o", help="Save JSON report to file")
    p_det.set_defaults(func=cmd_detect)

    # serve
    p_srv = sub.add_parser("serve", help="Start web UI server")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.add_argument("--host", default="0.0.0.0")
    p_srv.set_defaults(func=cmd_serve)

    # generate
    p_gen = sub.add_parser("generate", help="Generate synthetic test data")
    p_gen.add_argument("--accounts", type=int, default=30, help="Total accounts (default 30)")
    p_gen.add_argument("--coordinated", type=int, default=8, help="Coordinated accounts (default 8)")
    p_gen.add_argument("--groups", type=int, default=1, help="Number of coordinated groups (default 1)")
    p_gen.add_argument("--posts", type=int, default=100, help="Posts per account (default 100)")
    p_gen.add_argument("--output", "-o", help="Output file (default synthetic.csv)")
    p_gen.set_defaults(func=cmd_generate)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
