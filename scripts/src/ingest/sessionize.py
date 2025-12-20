"""sessionize.py

What it does:
- Assigns session IDs to events based on time gaps (default: 30 minutes).
- Rewrites Event.id to include the session prefix (e.g., s0003:<original_id>) so later joins are easy.

Main entrypoint:
- assign_sessions(events, gap_minutes=30) -> (new_events, session_to_event_ids)

Notes:
- This is deterministic given the event ordering.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import Dict, List, Tuple

from .parse_takeout import Event


def assign_sessions(events: List[Event], *, gap_minutes: int = 30) -> Tuple[List[Event], Dict[str, List[str]]]:
    """
    Assign session_id by time gap. Returns:
      - new_events: list of Events with id rewritten to include session (stable unique)
      - session_to_event_ids: mapping session_id -> event ids
    """
    if not events:
        return [], {}

    gap = timedelta(minutes=gap_minutes)

    def _sid(i: int) -> str:
        return f"s{i:04d}"

    session_idx = 0
    session_id = _sid(session_idx)
    session_to_event_ids: Dict[str, List[str]] = {session_id: []}

    new_events: List[Event] = []
    prev_time = events[0].time

    for e in events:
        if (e.time - prev_time) > gap:
            session_idx += 1
            session_id = _sid(session_idx)
            session_to_event_ids[session_id] = []

        prev_time = e.time

        # make event.id unique + session-aware (helps later joins)
        new_id = f"{session_id}:{e.id}"
        new_events.append(replace(e, id=new_id))
        session_to_event_ids[session_id].append(new_id)

    return new_events, session_to_event_ids