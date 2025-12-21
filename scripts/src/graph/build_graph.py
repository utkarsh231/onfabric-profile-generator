"""build_graph.py

What it does:
- Builds the Phase-1 heterogeneous graph from sessionized 'Event's.
- Current graph schema is intentionally minimal and stable:
    sessions (s:), domains (d:), queries (q:)
- No keyword-based "utility" rules.
- Instead, computes an unsupervised per-query signal score `psignal` from this user's own behavior.

Main entrypoints:
- build_history_graph(events) -> nx.Graph
- basic_graph_stats(G) -> Dict[str, int]

Notes:
- This file is intentionally deterministic + interpretable.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import math
import re

import networkx as nx

from src.ingest.parse_takeout import Event


# -----------------------------
# Phase-1 graph policy
# -----------------------------

MIN_QUERY_QUALITY = 0.25

HUB_DOMAINS = {
    "google.com",
    "youtube.com",
    "wikipedia.org",
}


def _query_quality(q: str) -> float:
    """Heuristic query quality in [0,1]. Low => fragment / mostly numeric / junk."""
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

    return float(max(0.0, min(1.0, score)))


def _entropy(ctr: Counter[str]) -> float:
    tot = float(sum(ctr.values()))
    if tot <= 0:
        return 0.0
    h = 0.0
    for c in ctr.values():
        p = float(c) / tot
        h -= p * math.log(p + 1e-12)
    return float(h)


def _qclass_from_psignal(ps: float) -> str:
    """Data-driven: low psignal behaves like "utility/admin-like" (not dropped, just de-emphasized)."""
    return "utility" if float(ps) < 0.30 else "interest"


def build_history_graph(events: List[Event]) -> nx.Graph:
    """Build heterogeneous graph with only sessions, domains, queries.

    Nodes:
      - s:<session_id>  (ntype=session)
      - d:<domain>      (ntype=domain)
      - q:<query>       (ntype=query, qclass=interest|utility, qquality=float, psignal=float)

    Edges (weighted):
      - session-domain (etype=session-domain)
      - session-query  (etype=session-query)
      - domain-query   (etype=domain-query)

    Notes:
      - No keyword lists are used for noise/utility. `psignal` is computed from user-only stats.
      - Low-signal queries are NOT deleted; they are softly de-emphasized via weights.
    """

    G = nx.Graph()

    # Aggregate within session
    session_domains: Dict[str, Counter[str]] = defaultdict(Counter)
    session_queries: Dict[str, Counter[str]] = defaultdict(Counter)
    all_sessions: set[str] = set()

    for e in events:
        sid = e.id.split(":", 1)[0]
        all_sessions.add(sid)

        if e.domain:
            session_domains[sid][e.domain] += 1

        if e.query:
            session_queries[sid][e.query] += 1

    n_sessions = max(1, len(all_sessions))

    # Domain DF / IDF across sessions
    domain_df: Counter[str] = Counter()
    for _s, dctr in session_domains.items():
        for d in dctr.keys():
            domain_df[d] += 1

    def domain_idf(d: str) -> float:
        df = domain_df.get(d, 0)
        return math.log((1.0 + n_sessions) / (1.0 + df)) + 1.0

    # Query DF / IDF across sessions (all queries)
    query_df: Counter[str] = Counter()
    for _s, qctr in session_queries.items():
        for q in qctr.keys():
            query_df[q] += 1

    def query_idf(q: str) -> float:
        df = query_df.get(q, 0)
        return math.log((1.0 + n_sessions) / (1.0 + df)) + 1.0

    # -----------------------------
    # Unsupervised psignal per query (user-only)
    # -----------------------------
    # Intuition (interpretable):
    # - Higher if it appears across many sessions (distinctive, persistent)
    # - Higher if it co-occurs with many domains (not tied to one narrow admin flow)
    # - Lower if it is very bursty (spam repeats in a short time)
    # - Always gated by query quality (filters fragments/numeric noise)

    query_total: Counter[str] = Counter()
    for _s, _qctr in session_queries.items():
        query_total.update(_qctr)

    # query -> Counter(domain -> cooccur count)
    query_domain_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for sid in all_sessions:
        doms = session_domains.get(sid, Counter()).most_common(10)
        qctr = session_queries.get(sid, Counter())
        if not doms or not qctr:
            continue
        for q, qc in qctr.items():
            for d, dc in doms:
                query_domain_counts[q][d] += int(min(qc, dc))

    query_psignal: Dict[str, float] = {}
    for q in query_df.keys():
        df = max(1, int(query_df.get(q, 1)))
        idf = float(query_idf(q))
        qqual = float(_query_quality(q))

        # repeats per session where it appears (burstiness)
        burst = float(query_total.get(q, 0)) / float(df)

        # Normalize pieces to ~[0,1] in monotone, interpretable ways
        idf_n = 1.0 - math.exp(-0.7 * max(0.0, idf - 1.0))

        dom_ctr = query_domain_counts.get(q, Counter())
        ent = float(_entropy(dom_ctr))
        ent_max = math.log(float(max(1, len(dom_ctr)))) if len(dom_ctr) > 1 else 0.0
        ent_n = (ent / ent_max) if ent_max > 0 else 0.0
        ent_n = float(max(0.0, min(1.0, ent_n)))

        # Burstiness: higher burst => lower signal
        burst_n = 1.0 - 1.0 / (1.0 + math.log1p(max(0.0, burst)))
        burst_keep = float(max(0.0, min(1.0, 1.0 - burst_n)))

        raw = (0.55 * idf_n + 0.25 * ent_n + 0.20 * burst_keep) * qqual
        query_psignal[q] = float(max(0.0, min(1.0, raw)))

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

    # session-query edges (all queries; low-psignal is de-emphasized, not dropped)
    for s, qctr in session_queries.items():
        sn = f"s:{s}"
        for q, c in qctr.items():
            qqual = _query_quality(q)
            if qqual < MIN_QUERY_QUALITY:
                continue
            ps = float(query_psignal.get(q, 0.0))
            qn = f"q:{q}"
            w = math.log1p(c) * query_idf(q) * qqual * (0.20 + 0.80 * ps)
            w_sq[(sn, qn)] += float(w)

    # domain-query co-occurrence within session
    TOP_DOMAINS_PER_SESSION = 10
    TOP_QUERIES_PER_SESSION = 10
    for s in all_sessions:
        doms = session_domains.get(s, Counter()).most_common(TOP_DOMAINS_PER_SESSION)
        qs = session_queries.get(s, Counter()).most_common(TOP_QUERIES_PER_SESSION)

        for d, dc in doms:
            if d in HUB_DOMAINS:
                continue
            dn = f"d:{d}"
            d_w = domain_idf(d)

            for q, qc in qs:
                qqual = _query_quality(q)
                if qqual < MIN_QUERY_QUALITY:
                    continue
                ps = float(query_psignal.get(q, 0.0))
                qn = f"q:{q}"

                w = float(min(dc, qc)) * d_w * query_idf(q) * qqual * (0.20 + 0.80 * ps) * 0.65
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
        qqual = _query_quality(q_text)
        ps = float(query_psignal.get(q_text, 0.0))
        qclass = _qclass_from_psignal(ps)

        G.add_node(u, ntype="session")
        G.add_node(v, ntype="query", qclass=qclass, qquality=float(qqual), psignal=float(ps))

        if G.has_edge(u, v):
            G[u][v]["weight"] += float(w)
        else:
            G.add_edge(u, v, weight=float(w), etype="session-query")

    for (u, v), w in w_dq.items():
        q_text = v.split(":", 1)[1] if ":" in v else v
        qqual = _query_quality(q_text)
        ps = float(query_psignal.get(q_text, 0.0))
        qclass = _qclass_from_psignal(ps)

        G.add_node(u, ntype="domain")
        G.add_node(v, ntype="query", qclass=qclass, qquality=float(qqual), psignal=float(ps))

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