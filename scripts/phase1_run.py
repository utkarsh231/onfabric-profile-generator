from __future__ import annotations

import json
from pathlib import Path

from src.ingest.parse_takeout import load_events
from src.ingest.sessionize import assign_sessions
from src.graph.build_graph import build_history_graph, basic_graph_stats
from src.graph.communities import detect_topic_communities, summarize_topic_communities
from src.graph.trails import build_session_trails


def main() -> None:
    json_path = "search_history.json"
    out_dir = Path("artifacts_phase1")
    out_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(json_path)
    events, session_map = assign_sessions(events, gap_minutes=30)

    G = build_history_graph(events)
    stats = basic_graph_stats(G)

    node_to_comm = detect_topic_communities(G, min_size=8)
    comm_summaries = summarize_topic_communities(G, node_to_comm, top_k=10)

    trails = build_session_trails(events)

    (out_dir / "graph_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir / "community_summaries.json").write_text(json.dumps(comm_summaries, indent=2), encoding="utf-8")
    (out_dir / "node_to_comm.json").write_text(json.dumps(node_to_comm, indent=2), encoding="utf-8")
    (out_dir / "session_trails.json").write_text(json.dumps(trails, indent=2), encoding="utf-8")

    # quick human-readable report
    lines = []
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