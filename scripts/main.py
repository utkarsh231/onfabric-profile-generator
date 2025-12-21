from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx

from src.ingest.parse_takeout import load_events
from src.ingest.sessionize import assign_sessions
from src.graph.build_graph import build_history_graph
from src.graph.trails import build_session_trails

from src.agent.config import SuitConfig
from src.agent.context import build_query_context
from src.agent.expand import expand_suit
from src.agent.llm_judge import llm_build_profile_snapshot, llm_refine_suit_card
from src.agent.snapshot import (
    _enrich_snapshot_with_evidence,
    _recompute_top_sessions_from_kept_queries,
    _simple_snapshot,
)
from src.agent.suits import discover_suits
from src.agent.io import write_profile_md, save_json

def _build_query_meta(G: nx.Graph) -> Dict[str, dict]:
    """Used only for trails formatting (same as phase2_agent.py)."""
    query_meta: Dict[str, dict] = {}
    for n, data in G.nodes(data=True):
        if isinstance(n, str) and n.startswith("q:"):
            q = n.split(":", 1)[1]
            query_meta[q] = {
                "qclass": data.get("qclass", "interest"),
                "psignal": float(data.get("psignal", 0.0)),
            }
    return query_meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2: two-pass suits profile builder (interpretable)")
    ap.add_argument("--json", dest="json_path", type=str, default="search_history.json", help="Input history JSON")
    ap.add_argument("--out", dest="out_dir", type=str, default="artifacts", help="Output directory")
    ap.add_argument("--gap-min", dest="gap_minutes", type=int, default=30, help="Session gap in minutes")

    # Config knobs (interpretable)
    ap.add_argument("--seed-psignal-min", type=float, default=0.35)
    ap.add_argument("--sim-threshold", type=float, default=0.27)
    ap.add_argument("--expand-sim-threshold", type=float, default=0.18)
    ap.add_argument("--max-suits", type=int, default=8)
    ap.add_argument("--session-gate-sim", type=float, default=0.24)
    ap.add_argument("--domain-max-session-frac", type=float, default=0.18)

    ap.add_argument(
        "--no-llm-judge",
        dest="use_llm_judge",
        action="store_false",
        default=True,
        help="Disable LLM judge refinement (enabled by default)",
    )
    ap.add_argument(
        "--llm-model",
        type=str,
        default=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        help="Claude model ID or alias (e.g., claude-sonnet-4-5).",
    )
    ap.add_argument("--llm-cache", type=str, default="", help="Optional path to cache JSON for LLM judge")
    ap.add_argument(
        "--llm-base-url",
        type=str,
        default=os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        help="Base URL for Anthropic Messages API (default: https://api.anthropic.com).",
    )

    args = ap.parse_args()

    cfg = SuitConfig(
        seed_psignal_min=float(args.seed_psignal_min),
        sim_threshold=float(args.sim_threshold),
        expand_sim_threshold=float(args.expand_sim_threshold),
        max_suits=int(args.max_suits),
        session_gate_sim=float(args.session_gate_sim),
        domain_max_session_frac=float(args.domain_max_session_frac),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(args.json_path)
    events, _ = assign_sessions(events, gap_minutes=int(args.gap_minutes))

    qctx = build_query_context(events)
    G = build_history_graph(events)

    trails = build_session_trails(events, query_meta=_build_query_meta(G))

    suits, vecs, info = discover_suits(G, cfg)

    expanded: List[dict] = []
    for s in suits:
        expanded.append(expand_suit(G, s, vecs, info, cfg))

    model = str(args.llm_model)
    base_url = str(args.llm_base_url)

    if args.use_llm_judge:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("--use-llm-judge set but ANTHROPIC_API_KEY is not set")

        cache: Dict[str, dict] = {}
        cache_path = Path(args.llm_cache) if args.llm_cache else None
        if cache_path and cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                cache = {}

        all_labels = [c.get("label", "") for c in expanded]
        refined_cards: List[dict] = []
        for card in expanded:
            time.sleep(0.15)
            refined_cards.append(
                llm_refine_suit_card(
                    card,
                    query_ctx=qctx,
                    all_suit_labels=all_labels,
                    model=model,
                    api_key=api_key,
                    cache=cache,
                    base_url=base_url,
                )
            )
        expanded = refined_cards

        # Align representative sessions with LLM-pruned queries
        for card in expanded:
            try:
                card["top_sessions"] = _recompute_top_sessions_from_kept_queries(card, G, k=cfg.top_sessions_per_suit)
            except Exception:
                pass

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")

    snapshot: dict = _simple_snapshot(expanded, G)
    snapshot = _enrich_snapshot_with_evidence(snapshot, expanded, G)

    if args.use_llm_judge:
        try:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            snap = llm_build_profile_snapshot(
                expanded=expanded,
                query_ctx=qctx,
                model=model,
                api_key=api_key,
                base_url=base_url,
            )
            if isinstance(snap, dict) and snap:
                snapshot = _enrich_snapshot_with_evidence(snap, expanded, G)
        except Exception:
            pass

    payload = {
        "config": asdict(cfg),
        "n_events": len(events),
        "n_suits": len(expanded),
        "suits": expanded,
        "llm_judge": bool(args.use_llm_judge),
        "snapshot": snapshot,
    }

    save_json(out_dir / "suits.json", payload)
    write_profile_md(out_dir / "PROFILE.md", expanded, trails, snapshot=snapshot)

    print(f"Wrote: {out_dir / 'suits.json'}")
    print(f"Wrote: {out_dir / 'PROFILE.md'}")


if __name__ == "__main__":
    main()