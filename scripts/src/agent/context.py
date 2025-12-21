from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from urllib.parse import urlparse

def _safe_url_parts(url: str) -> Tuple[str, str]:
    """Return (domain, path_tokens_str) for a URL."""
    try:
        u = urlparse(url)
        host = (u.netloc or "").lower()
        path = (u.path or "").lower()
        toks = [t for t in re.findall(r"[a-z0-9]+", path) if 2 <= len(t) <= 20]
        return host, " ".join(toks[:12])
    except Exception:
        return "", ""

def build_query_context(events) -> Dict[str, dict]:
    ctx: Dict[str, dict] = {}
    dom_ctr: Dict[str, Counter[str]] = defaultdict(Counter)
    path_ctr: Dict[str, Counter[str]] = defaultdict(Counter)
    title_samples: Dict[str, List[str]] = defaultdict(list)
    url_samples: Dict[str, List[str]] = defaultdict(list)

    for e in events:
        q = (e.query or "").strip()
        if not q:
            continue

        d = (e.domain or "").lower().strip()
        if d.startswith("www."):
            d = d[4:]

        host, path_toks = ("", "")
        if getattr(e, "url", None):
            host, path_toks = _safe_url_parts(e.url)
            host = (host or "").lower().strip()
            if host.startswith("www."):
                host = host[4:]

        eff = ""
        if d == "google.com" and host and host != "google.com":
            eff = host
        elif d:
            eff = d
        elif host:
            eff = host

        if eff:
            dom_ctr[q][eff] += 1
        if host and host != eff:
            dom_ctr[q][host] += 1
        if path_toks:
            for t in path_toks.split():
                path_ctr[q][t] += 1
        if getattr(e, "url", None) and len(url_samples[q]) < 3:
            url_samples[q].append(e.url)
        if getattr(e, "title", None) and len(title_samples[q]) < 3:
            title_samples[q].append(e.title)

    for q in set(list(dom_ctr.keys()) + list(title_samples.keys()) + list(url_samples.keys())):
        ctx[q] = {
            "domains": [d for d, _ in dom_ctr[q].most_common(4)],
            "paths": [t for t, _ in path_ctr[q].most_common(6)],
            "titles": title_samples.get(q, [])[:3],
            "urls": url_samples.get(q, [])[:3],
        }
    return ctx