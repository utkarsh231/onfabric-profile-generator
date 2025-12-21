"""trails.py

What it does:
- Builds an explainable "micro-story" (trail) per session for human-readable output.
- Trails are computed from raw `Event`s so they remain faithful.

Main entrypoint:
- build_session_trails(events, max_events_per_session=8, query_meta=None) -> Dict[session_id, trail]

Notes:
- No keyword-based utility detection. If `query_meta` is supplied, we split queries using
  the qclass computed from data-driven `psignal`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional

from src.ingest.parse_takeout import Event


def build_session_trails(
    events: List[Event],
    *,
    max_events_per_session: int = 8,
    query_meta: Optional[Dict[str, dict]] = None,
) -> Dict[str, dict]:
    """Build an explainable "micro-story" per session."""

    by_session: Dict[str, List[Event]] = defaultdict(list)
    for e in events:
        sid = e.id.split(":", 1)[0]
        by_session[sid].append(e)

    trails: Dict[str, dict] = {}
    for sid, es in by_session.items():
        es_sorted = sorted(es, key=lambda x: x.time)
        t0 = es_sorted[0].time
        t1 = es_sorted[-1].time

        doms = [e.domain for e in es_sorted if e.domain]

        qs_interest: List[str] = []
        qs_utility: List[str] = []
        for ev in es_sorted:
            if not ev.query:
                continue
            if query_meta and isinstance(query_meta.get(ev.query), dict):
                qc = str(query_meta[ev.query].get("qclass", "interest"))
                (qs_utility if qc == "utility" else qs_interest).append(ev.query)
            else:
                qs_interest.append(ev.query)

        top_domains = [d for d, _ in Counter(doms).most_common(5)]
        top_queries = [q for q, _ in Counter(qs_interest).most_common(5)]
        top_queries_utility = [q for q, _ in Counter(qs_utility).most_common(5)]

        # representative titles: first/last + a few middles
        reps: List[str] = []
        if es_sorted:
            reps.append(es_sorted[0].title)
            if len(es_sorted) > 1:
                reps.append(es_sorted[-1].title)
        for e in es_sorted[1:-1]:
            if len(reps) >= max_events_per_session:
                break
            if e.title not in reps:
                reps.append(e.title)

        trails[sid] = {
            "session_id": sid,
            "start_time": t0.isoformat(),
            "end_time": t1.isoformat(),
            "n_events": len(es_sorted),
            "top_domains": top_domains,
            "top_queries": top_queries,
            "top_queries_utility": top_queries_utility,
            "representative_titles": reps[:max_events_per_session],
        }

    return trails