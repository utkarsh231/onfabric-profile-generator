from __future__ import annotations

import networkx as nx
import re
from collections import Counter
from typing import Iterable, List

from .expand import _sessions_for_item

# Helper to recompute top_sessions from LLM-pruned queries
def _recompute_top_sessions_from_kept_queries(card: dict, G: nx.Graph, *, k: int = 10) -> List[str]:
    """Recompute top_sessions based only on the card's current top_queries.

    This aligns representative sessions with LLM-pruned queries and reduces mixed sessions.
    """
    keep = set(card.get("top_queries") or [])
    if not keep:
        return card.get("top_sessions") or []

    sess_scores: Counter[str] = Counter()
    for ev in (card.get("evidence_primary") or []):
        if not isinstance(ev, dict):
            continue
        if ev.get("kind") != "query":
            continue
        if ev.get("text") not in keep:
            continue
        item_id = ev.get("item_id")
        if not isinstance(item_id, str) or not item_id.startswith("q:"):
            continue
        cos = float(ev.get("cosine", 0.0) or 0.0)
        for s in _sessions_for_item(G, item_id):
            sess_scores[s] += cos

    if not sess_scores:
        return card.get("top_sessions") or []

    top = [s for s, _ in sess_scores.most_common(int(k))]
    return [sid[2:] if sid.startswith("s:") else sid for sid in top]


# Evidence-grounded extraction helpers for snapshot
def _extract_place_mentions_from_queries(queries: Iterable[str], max_items: int = 8) -> List[str]:
    """Extract place mentions from queries in an evidence-grounded way."""
    ctr: Counter[str] = Counter()
    patterns = [
        re.compile(r"\bthings\s+to\s+do\s+in\s+([a-z][a-z\s]{2,40})", re.I),
        re.compile(r"\baround\s+([a-z][a-z\s]{2,40})\b", re.I),
        re.compile(r"\bnear\s+([a-z][a-z\s]{2,40})\b", re.I),
    ]

    for q in queries:
        s = (q or "").strip().lower()
        if not s:
            continue
        for pat in patterns:
            m = pat.search(s)
            if not m:
                continue
            place = m.group(1).strip()
            place = " ".join(place.split()[:3])
            if len(place) < 3:
                continue
            ctr[place] += 1

    return [p for p, _ in ctr.most_common(max_items)]


def _extract_fashion_signals(queries: Iterable[str]) -> List[str]:
    kws = ["dress", "gown", "heels", "shoes", "jewelry", "jewellery", "van cleef", "revolve", "soles", "bags"]
    out: List[str] = []
    seen: set[str] = set()
    for q in queries:
        s = (q or "").lower()
        if any(k in s for k in kws):
            if q not in seen:
                seen.add(q)
                out.append(q)
        if len(out) >= 6:
            break
    return out



def _extract_travel_signals(queries: Iterable[str]) -> List[str]:
    kws = ["skyscanner", "airport", "lake como", "schengen", "things to do", "tickets", "visa"]
    out: List[str] = []
    seen: set[str] = set()
    for q in queries:
        s = (q or "").lower()
        if any(k in s for k in kws):
            if q not in seen:
                seen.add(q)
                out.append(q)
        if len(out) >= 6:
            break
    return out


# Broader query gatherer for snapshot
def _gather_snapshot_queries(expanded: List[dict], G: nx.Graph, *, per_session: int = 15, max_total: int = 800) -> List[str]:
    """Collect a broader, evidence-grounded query list for snapshot extraction.

    We intentionally look beyond `top_queries` by also pulling:
    - query evidence from evidence_primary/supporting
    - queries connected to representative sessions

    This prevents false "Not enough evidence" when the signal exists but isn't in the top-k list.
    """
    out: List[str] = []
    seen: set[str] = set()

    def _add(q: str) -> None:
        q = (q or "").strip()
        if not q:
            return
        if q in seen:
            return
        seen.add(q)
        out.append(q)

    for card in expanded:
        for q in (card.get("top_queries") or [])[:60]:
            _add(q)

        # Pull high-psignal query evidence from primary/supporting lists
        for ev in (card.get("evidence_primary") or []):
            if isinstance(ev, dict) and ev.get("kind") == "query":
                ps = float(ev.get("psignal", 0.0) or 0.0)
                if ps >= 0.45:
                    _add(ev.get("text") or "")
        for ev in (card.get("evidence_supporting") or []):
            if isinstance(ev, dict) and ev.get("kind") == "query":
                ps = float(ev.get("psignal", 0.0) or 0.0)
                if ps >= 0.45:
                    _add(ev.get("text") or "")

        # Pull queries from representative sessions (these often contain long-tail personal signal)
        for sid in (card.get("top_sessions") or [])[:10]:
            s_node = sid
            if isinstance(s_node, str) and not s_node.startswith("s:"):
                s_node = f"s:{s_node}"
            if not isinstance(s_node, str) or not G.has_node(s_node):
                continue
            n = 0
            for nbr in G.neighbors(s_node):
                if isinstance(nbr, str) and nbr.startswith("q:"):
                    _add(nbr[2:])
                    n += 1
                    if n >= int(per_session):
                        break

        if len(out) >= int(max_total):
            break

    return out[: int(max_total)]


def _simple_snapshot(expanded: List[dict], G: nx.Graph) -> dict:
    """Non-LLM fallback snapshot: evidence-grounded and minimal (no invention)."""
    top_labels = [s.get("label", "") for s in expanded if s.get("label")][:6]

    all_q: List[str] = _gather_snapshot_queries(expanded, G, per_session=15, max_total=800)

    places = _extract_place_mentions_from_queries(all_q, max_items=8)
    fashion_hits = _extract_fashion_signals(all_q)
    travel_hits = _extract_travel_signals(all_q)

    snap = {
        "location": "Not enough evidence to confidently infer location.",
        "lifestyle": ("Themes suggest: " + ", ".join([x for x in top_labels if x])) if top_labels else "Not enough evidence.",
        "fashion": "Evidence suggests some fashion/shopping activity." if fashion_hits else "Not enough evidence.",
        "travel": "Evidence suggests some travel planning activity." if travel_hits else "Not enough evidence.",
        "work": "Not enough evidence.",
        "notes": "Evidence-grounded snapshot (no invention).",
    }

    if places:
        snap["other_places_searched"] = ", ".join(places[:6])
    if fashion_hits:
        snap["fashion_examples"] = fashion_hits
    if travel_hits:
        snap["travel_examples"] = travel_hits

    return snap


# Post-processor to enrich snapshots using evidence-grounded extraction
def _enrich_snapshot_with_evidence(snapshot: dict, expanded: List[dict], G: nx.Graph) -> dict:
    """Ensure snapshot reflects evidence when present (no hallucination).

    If an LLM snapshot is conservative (or if top-k lists miss signals), we backfill
    fashion/travel/places using evidence-grounded extraction.
    """
    try:
        all_q = _gather_snapshot_queries(expanded, G, per_session=15, max_total=800)
        places = _extract_place_mentions_from_queries(all_q, max_items=8)
        fashion_hits = _extract_fashion_signals(all_q)
        travel_hits = _extract_travel_signals(all_q)

        snap = dict(snapshot or {})

        # Backfill other places if present
        if places and not snap.get("other_places_searched"):
            snap["other_places_searched"] = ", ".join(places[:6])

        # Upgrade fashion/travel from "Not enough evidence" if we have evidence
        if fashion_hits:
            if (str(snap.get("fashion", "")) or "").lower().startswith("not enough"):
                snap["fashion"] = "Evidence suggests some fashion/shopping activity."
            snap["fashion_examples"] = snap.get("fashion_examples") or fashion_hits

        if travel_hits:
            if (str(snap.get("travel", "")) or "").lower().startswith("not enough"):
                snap["travel"] = "Evidence suggests some travel planning activity."
            snap["travel_examples"] = snap.get("travel_examples") or travel_hits

        return snap
    except Exception:
        return snapshot or {}
