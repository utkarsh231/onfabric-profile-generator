"""trails.py

What it does:
- Builds an explainable "micro-story" (trail) per session for human-readable output.
- Trails are computed from raw `Event`s (not from the clustered graph) so they remain faithful.

Main entrypoint:
- build_session_trails(events, max_events_per_session=8) -> Dict[session_id, trail]

Notes:
- Uses the same utility-query detection as Phase-1 graph building.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List

import re

from src.ingest.parse_takeout import Event
from src.graph.build_graph import is_utility_query


def build_session_trails(events: List[Event], *, max_events_per_session: int = 8) -> Dict[str, dict]:
    """
    Build an explainable "micro-story" per session:
      - time span
      - top domains
      - top queries
      - representative event titles
    """
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
        qs_interest = [e.query for e in es_sorted if e.query and not is_utility_query(e.query)]
        qs_utility = [e.query for e in es_sorted if e.query and is_utility_query(e.query)]

        top_domains = [d for d, _ in Counter(doms).most_common(5)]
        # Keep trails focused on interest queries by default
        top_queries = [q for q, _ in Counter(qs_interest).most_common(5)]
        top_queries_utility = [q for q, _ in Counter(qs_utility).most_common(5)]

        # representative titles: first/last + a few middles
        reps: List[str] = []
        if es_sorted:
            reps.append(es_sorted[0].title)
            if len(es_sorted) > 1:
                reps.append(es_sorted[-1].title)
        # add some middle titles (unique)
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