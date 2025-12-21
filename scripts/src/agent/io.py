from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional

def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def write_profile_md(path: Path, suits: List[dict], trails: dict, snapshot: Optional[dict] = None) -> None:
    lines: List[str] = []
    lines.append("# User Profile")
    lines.append("")

    if snapshot:
        lines.append("## Snapshot")
        lines.append("")
        lines.append(f"- **Location:** {snapshot.get('location', 'Not enough evidence')}")
        lines.append(f"- **Lifestyle:** {snapshot.get('lifestyle', 'Not enough evidence')}")
        lines.append(f"- **Fashion:** {snapshot.get('fashion', 'Not enough evidence')}")
        lines.append(f"- **Travel:** {snapshot.get('travel', 'Not enough evidence')}")
        lines.append(f"- **Work:** {snapshot.get('work', 'Not enough evidence')}")
        if snapshot.get("other_places_searched"):
            lines.append(f"- **Other places searched:** {snapshot.get('other_places_searched')}")
        if snapshot.get("fashion_examples"):
            lines.append(f"- **Fashion evidence:** {', '.join(snapshot.get('fashion_examples')[:4])}")
        if snapshot.get("travel_examples"):
            lines.append(f"- **Travel evidence:** {', '.join(snapshot.get('travel_examples')[:4])}")
        other = snapshot.get("other_interests")
        if other:
            lines.append(f"- **Other interests:** {other}")
        notes = snapshot.get("confidence_notes") or snapshot.get("notes")
        if notes:
            lines.append(f"- **Notes:** {notes}")
        lines.append("")

    for s in suits:
        lines.append(f"## {s['label']} (mass={s['mass']:.2f})")
        lines.append("")
        lines.append(s.get("paragraph", ""))
        lines.append("")

        tq = s.get("top_queries", [])
        td = s.get("top_domains", [])
        ts = s.get("top_sessions", [])

        if td:
            lines.append("**Top domains**")
            for d in td[:8]:
                lines.append(f"- {d}")
            lines.append("")

        if tq:
            lines.append("**Top queries**")
            for q in tq[:10]:
                lines.append(f"- {q}")
            lines.append("")

        if ts:
            lines.append("**Representative sessions**")
            for sid in ts[:6]:
                t = trails.get(sid)
                if not t:
                    lines.append(f"- {sid}")
                    continue
                reps = t.get("representative_titles", [])
                rep_line = " | ".join(reps[:3]) if reps else ""
                lines.append(f"- {sid}: {rep_line}")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")