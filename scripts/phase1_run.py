"""phase1_run.py

Phase 1 pipeline runner.

What it does:
- Loads Takeout/Chrome history JSON -> normalized Events
- Sessionizes events (time-gap based)
- Builds a minimal heterogeneous graph (sessions/domains/queries)
- Detects topical communities (domain/query projection)
- Writes Phase-1 artifacts to disk + a small human-readable report

Outputs (in --out):
- graph_stats.json
- community_summaries.json
- node_to_comm.json
- session_trails.json
- REPORT.md

Usage:
  python scripts/phase1_run.py --json search_history.json --out artifacts_phase1

Notes:
- Defaults match the previous behavior (json_path=search_history.json, gap=30, min_size=8, top_k=10).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ingest.parse_takeout import load_events
from src.ingest.sessionize import assign_sessions
from src.graph.build_graph import build_history_graph, basic_graph_stats
from src.graph.communities import detect_topic_communities, summarize_topic_communities
from src.graph.trails import build_session_trails


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Phase 1: ingestion -> sessionize -> graph -> communities")
    ap.add_argument("--json", dest="json_path", type=str, default="search_history.json", help="Input history JSON")
    ap.add_argument("--out", dest="out_dir", type=str, default="artifacts_phase1", help="Output directory")
    ap.add_argument("--gap-min", dest="gap_minutes", type=int, default=30, help="Session gap in minutes")
    ap.add_argument("--min-community-size", dest="min_size", type=int, default=8, help="Min community size")
    ap.add_argument("--top-k", dest="top_k", type=int, default=10, help="Top-k domains/queries per community")
    args = ap.parse_args()

    json_path = args.json_path
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(json_path)
    events, _session_map = assign_sessions(events, gap_minutes=int(args.gap_minutes))

    G = build_history_graph(events)
    stats = basic_graph_stats(G)

    node_to_comm = detect_topic_communities(G, min_size=int(args.min_size))
    comm_summaries = summarize_topic_communities(G, node_to_comm, top_k=int(args.top_k))

    trails = build_session_trails(events)

    (out_dir / "graph_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir / "community_summaries.json").write_text(json.dumps(comm_summaries, indent=2), encoding="utf-8")
    (out_dir / "node_to_comm.json").write_text(json.dumps(node_to_comm, indent=2), encoding="utf-8")
    (out_dir / "session_trails.json").write_text(json.dumps(trails, indent=2), encoding="utf-8")

    # quick human-readable report
    lines: list[str] = []
    lines.append("# Phase 1 Report")
    lines.append("")
    lines.append("## Graph stats")
    lines.append(f"- nodes: {stats['nodes']}")
    lines.append(f"- edges: {stats['edges']}")
    lines.append(f"- sessions: {stats['sessions']}")
    lines.append(f"- domains: {stats['domains']}")
    lines.append(f"- queries: {stats['queries']}")
    lines.append("")
    lines.append("## Top communities (by size)")
    for c in comm_summaries[:8]:
        lines.append(f"### Community {c['community_id']} (size={c['size']})")
        lines.append(f"- top domains: {', '.join(c['top_domains'][:8])}")
        lines.append(f"- top queries: {', '.join(c['top_queries'][:8])}")
        lines.append("")

    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote artifacts to: {out_dir.resolve()}")
    print(f"Open {out_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()