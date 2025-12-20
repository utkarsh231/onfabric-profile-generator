"""build_graph.py

What it does:
- Builds the Phase-1 heterogeneous graph from sessionized 'Event's.
- Current graph schema is intentionally minimal and stable:
    sessions (s:), domains (d:), queries (q:)
- Utility/admin queries are retained but downweighted and excluded from topic clustering.

Main entrypoints:
- build_history_graph(events) -> nx.Graph
- basic_graph_stats(G) -> Dict[str, int]

Notes:
- We keep the graph small/clean to improve community detection stability.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import math
import re

import networkx as nx

from src.ingest.parse_takeout import Event


# -----------------------------
# Phase-1 graph policy (sessions + domains + queries only)
# -----------------------------

MIN_QUERY_QUALITY = 0.25

_UTILITY_QUERY_PATTERNS = (
    re.compile(r"\b(speed\s*test|internet\s*speed|fast\.com|ookla)\b", re.I),
    re.compile(r"\b(weather|temperature|forecast)\b", re.I),
    re.compile(r"\b(time\s+in|timezone|utc\s*to)\b", re.I),
    re.compile(r"\b(translate|meaning\s+of|define\s+)\b", re.I),
    re.compile(r"\b(map|directions\s+to|near\s+me)\b", re.I),
    re.compile(r"\b(convert|converter|cm\s+to|kg\s+to|usd\s+to|inr\s+to)\b", re.I),
    re.compile(r"\b(login|sign\s*in|password|otp)\b", re.I),
    re.compile(r"\b(after\s*tax|tax\s*calculator|salary\s*calculator)\b", re.I),
)

HUB_DOMAINS = {
    "google.com",
    "youtube.com",
    "wikipedia.org",
}


def is_utility_query(q: str) -> bool:
    qq = (q or "").strip().lower()
    if not qq:
        return True
    return any(p.search(qq) is not None for p in _UTILITY_QUERY_PATTERNS)


def _query_quality(q: str) -> float:
    """Heuristic query quality in [0,1]. Low => likely noise/admin/fragment."""
    qq = (q or "").strip().lower()
    if not qq:
        return 0.0

    qq = re.sub(r"\s+", " ", qq).strip()

    # too short / fragments
    if len(qq) <= 1:
        return 0.0
    if len(qq) <= 2:
        return 0.10

    toks = [t for t in qq.split() if t]
    if not toks:
        return 0.0

    # penalize mostly-numeric
    alpha = sum(ch.isalpha() for ch in qq)
    if alpha / max(1, len(qq)) < 0.30:
        return 0.25

    score = 0.55
    if len(qq) >= 8:
        score += 0.15
    if len(toks) >= 2:
        score += 0.15
    if len(toks) >= 3:
        score += 0.05

    if re.fullmatch(r"[a-z]{1,2}", qq):
        score -= 0.35
    if re.search(r"\b(test|speedtest)\b", qq):
        score -= 0.25

    return float(max(0.0, min(1.0, score)))


def build_history_graph(events: List[Event]) -> nx.Graph:
    """Build heterogeneous graph with only sessions, domains, queries.

    Nodes:
      - s:<session_id>  (ntype=session)
      - d:<domain>      (ntype=domain)
      - q:<query>       (ntype=query, qclass=interest|utility, qquality=float)

    Edges (weighted):
      - session-domain (etype=session-domain)
      - session-query  (etype=session-query)
      - domain-query   (etype=domain-query)

    Notes:
      - Utility queries are kept but downweighted and excluded from topical clustering.
      - Hub domains are downweighted to avoid dominating communities.
      - Visited/viewed events do NOT become queries (parse_takeout leaves query empty).
    """

    G = nx.Graph()

    # Aggregate within session
    session_domains: Dict[str, Counter[str]] = defaultdict(Counter)
    session_queries_interest: Dict[str, Counter[str]] = defaultdict(Counter)
    session_queries_utility: Dict[str, Counter[str]] = defaultdict(Counter)

    all_sessions = set()

    for e in events:
        sid = e.id.split(":", 1)[0]
        all_sessions.add(sid)

        if e.domain:
            session_domains[sid][e.domain] += 1

        if e.query:
            if is_utility_query(e.query):
                session_queries_utility[sid][e.query] += 1
            else:
                session_queries_interest[sid][e.query] += 1

    n_sessions = max(1, len(all_sessions))

    # Domain DF / IDF across sessions
    domain_df: Counter[str] = Counter()
    for s, dctr in session_domains.items():
        for d in dctr.keys():
            domain_df[d] += 1

    def domain_idf(d: str) -> float:
        df = domain_df.get(d, 0)
        return math.log((1.0 + n_sessions) / (1.0 + df)) + 1.0

    # Query DF / IDF across sessions (interest only)
    query_df: Counter[str] = Counter()
    for s, qctr in session_queries_interest.items():
        for q in qctr.keys():
            query_df[q] += 1

    def query_idf(q: str) -> float:
        df = query_df.get(q, 0)
        return math.log((1.0 + n_sessions) / (1.0 + df)) + 1.0

    # Weighted edges
    w_sd: Counter[Tuple[str, str]] = Counter()
    w_sq: Counter[Tuple[str, str]] = Counter()
    w_dq: Counter[Tuple[str, str]] = Counter()

    # session-domain edges
    for s, dctr in session_domains.items():
        sn = f"s:{s}"
        for d, c in dctr.items():
            dn = f"d:{d}"
            if d in HUB_DOMAINS:
                w = math.log1p(c) * domain_idf(d) * 0.15
            else:
                w = math.log1p(c) * domain_idf(d) * 0.90
            w_sd[(sn, dn)] += float(w)

    # session-query edges (interest + utility)
    for s, qctr in session_queries_interest.items():
        sn = f"s:{s}"
        for q, c in qctr.items():
            qqual = _query_quality(q)
            if qqual < MIN_QUERY_QUALITY:
                continue
            qn = f"q:{q}"
            w = math.log1p(c) * query_idf(q) * qqual
            w_sq[(sn, qn)] += float(w)

    for s, qctr in session_queries_utility.items():
        sn = f"s:{s}"
        for q, c in qctr.items():
            qn = f"q:{q}"
            # keep utility, but downweight heavily so it doesn't dominate
            w = math.log1p(c) * 0.25
            w_sq[(sn, qn)] += float(w)

    # domain-query co-occurrence within session (interest only)
    TOP_DOMAINS_PER_SESSION = 10
    TOP_QUERIES_PER_SESSION = 10
    for s in all_sessions:
        doms = session_domains.get(s, Counter()).most_common(TOP_DOMAINS_PER_SESSION)
        qs = session_queries_interest.get(s, Counter()).most_common(TOP_QUERIES_PER_SESSION)
        for d, dc in doms:
            if d in HUB_DOMAINS:
                continue
            dn = f"d:{d}"
            d_w = domain_idf(d)
            for q, qc in qs:
                qqual = _query_quality(q)
                if qqual < MIN_QUERY_QUALITY:
                    continue
                qn = f"q:{q}"
                w = float(min(dc, qc)) * d_w * query_idf(q) * qqual * 0.65
                w_dq[(dn, qn)] += float(w)

    # Materialize nodes/edges
    for s in all_sessions:
        G.add_node(f"s:{s}", ntype="session")

    for (u, v), w in w_sd.items():
        G.add_node(u, ntype="session")
        G.add_node(v, ntype="domain")
        G.add_edge(u, v, weight=float(w), etype="session-domain")

    for (u, v), w in w_sq.items():
        q_text = v.split(":", 1)[1] if ":" in v else v
        qclass = "utility" if is_utility_query(q_text) else "interest"
        qqual = _query_quality(q_text)
        G.add_node(u, ntype="session")
        G.add_node(v, ntype="query", qclass=qclass, qquality=float(qqual))
        if G.has_edge(u, v):
            G[u][v]["weight"] += float(w)
        else:
            G.add_edge(u, v, weight=float(w), etype="session-query")

    for (u, v), w in w_dq.items():
        q_text = v.split(":", 1)[1] if ":" in v else v
        qclass = "utility" if is_utility_query(q_text) else "interest"
        qqual = _query_quality(q_text)
        G.add_node(u, ntype="domain")
        G.add_node(v, ntype="query", qclass=qclass, qquality=float(qqual))
        if G.has_edge(u, v):
            G[u][v]["weight"] += float(w)
        else:
            G.add_edge(u, v, weight=float(w), etype="domain-query")

    return G


def basic_graph_stats(G: nx.Graph) -> Dict[str, int]:
    ntypes = Counter(nx.get_node_attributes(G, "ntype").values())
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "sessions": int(ntypes.get("session", 0)),
        "domains": int(ntypes.get("domain", 0)),
        "queries": int(ntypes.get("query", 0)),
    }