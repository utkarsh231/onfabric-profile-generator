"""
What this file does:
- Loads Google Takeout / Chrome history JSON (already exported into a flat list of events).
- Normalizes timestamps, URLs (including Google redirect URLs), and domains.
- Extracts *search queries* ONLY from true "Searched for ..." events.

Main entrypoint:
- load_events(json_path) -> List[Event]

Outputs:
- Event dataclass objects (sorted by time, UTC)

Notes:
- We intentionally do NOT synthesize queries for visit/view events; doing so creates supernodes and
  collapses communities.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse


SEARCH_RE = re.compile(r"^Searched for (.+)$", re.IGNORECASE)
VISIT_RE = re.compile(r"^Visited (.+)$", re.IGNORECASE)
VIEW_RE = re.compile(r"^Viewed (.+)$", re.IGNORECASE)

EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Event:
    id: str
    time: datetime
    title: str
    title_url: Optional[str]
    event_type: str  # search|visit|view|other
    query: str       # extracted query/title text (normalized)
    url: Optional[str]
    domain: str
    subtitles: List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["time"] = self.time.isoformat()
        return d


def _parse_time(s: Any) -> datetime:
    """
    Takeout often uses ISO strings like '2024-01-05T08:29:34.280Z'.
    """
    if not isinstance(s, str) or not s:
        # fallback: epoch start UTC
        return EPOCH_UTC

    # normalize 'Z'
    s2 = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return EPOCH_UTC


def _clean_google_redirect(maybe_url: Any) -> Optional[str]:
    if not isinstance(maybe_url, str) or not maybe_url:
        return None

    url = maybe_url.strip()
    try:
        u = urlparse(url)
        host = (u.netloc or "").lower()

        # Common Google redirect shapes:
        # - https://www.google.com/url?q=<dest>
        # - https://www.google.com/url?url=<dest>
        # - sometimes 'q' is present on other google domains too
        if "google" in host and u.path.startswith("/url"):
            qs = parse_qs(u.query)
            for key in ("q", "url"):
                if key in qs and qs[key]:
                    return unquote(qs[key][0])

        return url
    except Exception:
        return url


def _normalize_domain(host: str) -> str:
    """Normalize domains to reduce fragmentation (www/mobile/language/country variants).

    Deterministic + explainable (regex-based). We normalize only a few high-impact
    families that commonly fragment graphs.
    """
    h = (host or "").strip().lower()

    # strip any port
    if ":" in h:
        h = h.split(":", 1)[0]

    # strip leading www
    if h.startswith("www."):
        h = h[4:]

    # -----------------------------
    # Google: collapse country TLD variants like google.co.uk, google.com.au, google.de
    # but keep meaningful service subdomains like scholar.google.com.
    # -----------------------------
    if h in {"google.com", "local.google.com"}:
        return "google.com"

    # google.<ccTLD>, google.co.<ccTLD>, google.com.<ccTLD> -> google.com
    # Examples: google.de, google.co.uk, google.com.au
    if re.match(r"^google\.(?:[a-z]{2,3}|co\.[a-z]{2}|com\.[a-z]{2})$", h):
        return "google.com"

    # -----------------------------
    # YouTube: collapse mobile/other subdomains
    # Examples: m.youtube.com -> youtube.com
    # -----------------------------
    if h == "youtu.be":
        return "youtu.be"
    if h == "youtube.com" or h.endswith(".youtube.com"):
        return "youtube.com"

    # -----------------------------
    # Wikipedia: collapse language/mobile subdomains
    # Examples: en.m.wikipedia.org, en.wikipedia.org -> wikipedia.org
    # -----------------------------
    if h == "wikipedia.org" or h.endswith(".wikipedia.org"):
        return "wikipedia.org"

    return h


def _extract_domain(url: Optional[str]) -> str:
    if not url:
        return ""
    try:
        host = (urlparse(url).netloc or "")
        return _normalize_domain(host)
    except Exception:
        return ""


def _infer_event_type(title: str) -> str:
    t = (title or "").strip()
    if SEARCH_RE.match(t):
        return "search"
    if VISIT_RE.match(t):
        return "visit"
    if VIEW_RE.match(t):
        return "view"
    return "other"


def _extract_query(title: str) -> str:
    """Extract a *search query* only for true search events.

    Important: we intentionally do NOT synthesize queries for visited/viewed events.
    Domains belong in `Event.domain`; turning visits into `q:<domain>` creates
    supernodes (e.g., `q:google.com`) that collapse communities.
    """
    t = (title or "").strip()

    m = SEARCH_RE.match(t)
    if m:
        return m.group(1).strip()

    # For visited/viewed/other events, leave query empty.
    return ""


def _normalize_query(q: str) -> str:
    """Normalize queries for stable graph keys (regex-based, explainable)."""
    q = (q or "").strip().lower()

    # normalize common unicode quotes/apostrophes
    q = q.replace("\u2019", "'").replace("\u2018", "'")
    q = q.replace("\u201c", '"').replace("\u201d", '"')

    # collapse whitespace
    q = re.sub(r"\s+", " ", q)

    # strip surrounding quotes
    q = q.strip("\"'“”‘’")

    # drop trailing punctuation that often creates duplicate keys (.,!?;:)
    q = re.sub(r"[\.,!?;:]+$", "", q).strip()

    return q


def load_events(json_path: str) -> List[Event]:
    """
    Load and noralize browsing/search events from an exported takeout JSON file.

    Parameters:
    - json_path: Path to the JSON file containing the browsing/search events.
    Returns:
    - List[Event]: List of Event dataclass objects, sorted by time (UTC).
    
    Behavior:
    - Parses event time into a timezone aware UTC datetime (invalid times fall back to EPOCH_UTC 
    to keep ingestion consistent).
    - Cleans up Google redirect URLs to extract the original destination URL.
    - Extracts domains from the cleaned URL and normalizes common families (Google/Youtube/Wikipedia) 
    to reduce fragmentation.
    - Classifies events as search/visit/view/other based on the title text.

    Notes:
    We intentionally do not synthesize queries for visit/view events; 
    doing so creates supernodes (large high degree nodes) like `q:google.com` that
    collapse communities and make the graph less meaningful.
    """

    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"search_history.json not found: {p}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {p}: {e}") from e

    if not isinstance(data, list):
        raise ValueError(f"Expected list of events in {p}, got {type(data).__name__}")

    out: List[Event] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            continue

        time = _parse_time(row.get("time"))
        title = str(row.get("title") or "")
        title_url = row.get("titleUrl")

        url = _clean_google_redirect(title_url)
        domain = _extract_domain(url)
        event_type = _infer_event_type(title)

        subtitles_raw = row.get("subtitles") or []
        subtitles: List[str] = []
        if isinstance(subtitles_raw, list):
            for s in subtitles_raw:
                if isinstance(s, dict) and isinstance(s.get("name"), str):
                    subtitles.append(s["name"])
                elif isinstance(s, str):
                    subtitles.append(s)

        query = _normalize_query(_extract_query(title))

        out.append(
            Event(
                id=str(row.get("id") or f"evt_{i}"),
                time=time,
                title=title,
                title_url=title_url if isinstance(title_url, str) else None,
                event_type=event_type,
                query=query,
                url=url,
                domain=domain,
                subtitles=subtitles,
            )
        )

    # sort by time
    out.sort(key=lambda e: e.time)
    return out
