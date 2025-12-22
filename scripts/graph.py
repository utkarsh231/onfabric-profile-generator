"""
profile_v2_graph_only.py

Graph-only profile builder designed to be more stable than the Phase-2 suit expansion.
It does NOT modify existing code and does NOT require an LLM.

Key ideas:
- Use Phase-1 topic communities on the domain–query projection (drop session bridges).
- Remove bridge-y nodes using a specificity filter:
    specificity(node) = internal_weight / total_weight
- Pick representative sessions using a purity filter:
    purity(session, topic) = topic_weight / total_weight_over_topics
- Build a broad snapshot by scoring "facets" (travel, fashion, fitness, etc.)
  against evidence (queries, domains, titles).

Run:
  python scripts/profile_v2_graph_only.py --json search_history.json --out artifacts_profile_v2

Outputs:
  - profile_v2.json
  - PROFILE_V2.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import networkx as nx

from src.ingest.parse_takeout import load_events
from src.ingest.sessionize import assign_sessions
from src.graph.build_graph import build_history_graph, MIN_QUERY_QUALITY
from src.graph.communities import build_domain_query_projection, detect_communities

# Domains that act like "super hubs" and glue unrelated queries together.
# In Google Takeout history, google.* often appears as the referrer/search engine domain.
# Dropping it from the domain–query projection typically improves community purity.
DROP_PROJECTION_DOMAINS = {
    "google.com",
    "www.google.com",
    "google.co.uk",
    "www.google.co.uk",
}


def build_domain_query_projection_filtered(G: nx.Graph, drop_domains: Optional[set[str]] = None) -> nx.Graph:
    """Build a domain–query projection while excluding noisy hub domains."""
    drop_domains = drop_domains or set()
    H = nx.Graph()

    for u, v, data in G.edges(data=True):
        # Keep the same intent as the original projection: only domain-query edges.
        et = data.get("etype")
        if et and et != "domain-query":
            continue

        if isinstance(u, str) and u.startswith("d:") and isinstance(v, str) and v.startswith("q:"):
            d_node, q_node = u, v
        elif isinstance(v, str) and v.startswith("d:") and isinstance(u, str) and u.startswith("q:"):
            d_node, q_node = v, u
        else:
            continue

        d_name = d_node[2:]
        if d_name in drop_domains:
            continue

        w = float(data.get("weight", 1.0))
        if w <= 0:
            continue

        if H.has_edge(d_node, q_node):
            H[d_node][q_node]["weight"] += w
        else:
            H.add_edge(d_node, q_node, weight=w)

    return H


def _topic_label(top_q: List[str], top_d: List[str]) -> str:
    """Heuristic label that is less random than `top_queries[0]`."""
    dom_text = " ".join(top_d).lower()

    if any(x in dom_text for x in [
        "selfridges", "johnlewis", "vogue", "harpersbazaar", "whowhatwear", "tedbaker",
        "thefoldlondon", "houseoffraser", "asos", "zara", "cultbeauty", "sephora",
    ]):
        return "Fashion & Shopping"

    if any(x in dom_text for x in [
        "booking", "skyscanner", "tripadvisor", "rome2rio", "expedia", "airbnb", "kayak",
    ]):
        return "Travel & Trips"

    if any(x in dom_text for x in [
        "ncbi", "mayoclinic", "webmd", "healthline", "cdc", "nhs.uk",
    ]):
        return "Health"

    if any(x in dom_text for x in [
        "github", "stackoverflow", "developer.apple", "numpy", "pandas", "scikit",
    ]):
        return "Engineering / Coding"

    if any(x in dom_text for x in [
        "pitchbook", "crunchbase", "finextra", "statista", "deloitte", "investopedia", "nerdwallet",
    ]):
        return "Startups / Finance"

    if any(x in dom_text for x in [
        "ikea", "wayfair", "argos", "fully", "standingdesk", "standing", "desk",
    ]):
        return "Home / Furniture"

    if any(x in dom_text for x in [
        "kcl.ac.uk", "mykcl", "keats", "nyu.edu",
    ]):
        return "Education / Admin"

    return (top_q[0] if top_q else (top_d[0] if top_d else "Topic"))


# -----------------------------
# Small context builder (copied/trimmed from your archive phase2_agent.py)
# -----------------------------

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
    """Lightweight per-query context from events (domains/titles/urls/path tokens)."""
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
        if getattr(e, "title", None) and len(title_samples[q]) < 3 and e.title:
            title_samples[q].append(e.title)

    for q in set(list(dom_ctr.keys()) + list(title_samples.keys()) + list(url_samples.keys())):
        ctx[q] = {
            "domains": [d for d, _ in dom_ctr[q].most_common(5)],
            "paths": [t for t, _ in path_ctr[q].most_common(8)],
            "titles": title_samples.get(q, [])[:3],
            "urls": url_samples.get(q, [])[:3],
        }
    return ctx


# -----------------------------
# Topic building
# -----------------------------

def _node_type(n: str) -> str:
    if isinstance(n, str) and n.startswith("q:"):
        return "query"
    if isinstance(n, str) and n.startswith("d:"):
        return "domain"
    if isinstance(n, str) and n.startswith("s:"):
        return "session"
    return "unknown"


def _specificity(H: nx.Graph, node: str, comm_nodes: set[str]) -> float:
    """internal_weight / total_weight for node (on projection graph H)."""
    tot = 0.0
    inside = 0.0
    for nb in H.neighbors(node):
        w = float(H[node][nb].get("weight", 1.0))
        tot += w
        if nb in comm_nodes:
            inside += w
    return float(inside / tot) if tot > 0 else 0.0


def _weighted_degree_within(H: nx.Graph, node: str, comm_nodes: set[str]) -> float:
    s = 0.0
    for nb in H.neighbors(node):
        if nb in comm_nodes:
            s += float(H[node][nb].get("weight", 1.0))
    return float(s)


def _topk_by_score(xs: List[Tuple[float, str]], k: int) -> List[str]:
    xs_sorted = sorted(xs, key=lambda t: t[0], reverse=True)[:k]
    return [x for _, x in xs_sorted]


def build_topic_cards(
    *,
    G: nx.Graph,
    min_comm_size: int = 8,
    max_topics: int = 8,
    top_queries: int = 10,
    top_domains: int = 8,
    specificity_min: float = 0.62,
) -> Tuple[List[dict], Dict[str, int]]:
    """
    Returns topic cards from projection communities, filtered for purity via specificity.
    """
    H = build_domain_query_projection_filtered(G, drop_domains=DROP_PROJECTION_DOMAINS)
    node_to_comm = detect_communities(H, min_size=int(min_comm_size))

    # group nodes by community id
    comm_to_nodes: Dict[int, List[str]] = defaultdict(list)
    for n, cid in node_to_comm.items():
        if cid != -1:
            comm_to_nodes[int(cid)].append(n)

    # sort biggest first
    comm_ids = sorted(comm_to_nodes.keys(), key=lambda cid: len(comm_to_nodes[cid]), reverse=True)[: int(max_topics)]

    cards: List[dict] = []
    for cid in comm_ids:
        nodes = comm_to_nodes[cid]
        comm_nodes = set(nodes)

        # filter nodes by specificity (removes bridge queries/domains)
        kept: List[str] = []
        for n in nodes:
            sp = _specificity(H, n, comm_nodes)
            if sp >= float(specificity_min):
                kept.append(n)

        if len(kept) < max(4, int(min_comm_size // 2)):
            # if too aggressive for sparse data, fall back to original nodes
            kept = list(nodes)
            comm_nodes = set(kept)

        # rank within-community by internal weighted degree
        q_scores: List[Tuple[float, str]] = []
        d_scores: List[Tuple[float, str]] = []
        for n in kept:
            t = _node_type(n)
            s = _weighted_degree_within(H, n, comm_nodes)
            if t == "query":
                q_scores.append((s, n))
            elif t == "domain":
                d_scores.append((s, n))

        top_q_nodes = _topk_by_score(q_scores, int(top_queries))
        top_d_nodes = _topk_by_score(d_scores, int(top_domains))

        top_q = [n[2:] for n in top_q_nodes]
        top_d = [n[2:] for n in top_d_nodes]

        # Less-random label than `top_queries[0]`
        label = _topic_label(top_q, top_d)

        cards.append(
            {
                "topic_id": int(cid),
                "size": int(len(nodes)),
                "label": str(label).strip()[:64],
                "top_queries": top_q,
                "top_domains": top_d,
                "specificity_min": float(specificity_min),
            }
        )

    return cards, {str(k): int(v) for k, v in node_to_comm.items()}


# -----------------------------
# Representative sessions (purity gated)
# -----------------------------

def _sessions(G: nx.Graph) -> List[str]:
    return [n for n in G.nodes if isinstance(n, str) and n.startswith("s:")]


def attach_representative_sessions(
    *,
    G: nx.Graph,
    cards: List[dict],
    node_to_comm: Dict[str, int],
    k_sessions: int = 8,
    purity_min: float = 0.65,
) -> List[dict]:
    """
    For each topic, select sessions with high topic-weight AND high purity to that topic,
    so sessions don't reintroduce mixed interests.
    """
    all_sessions = _sessions(G)
    for c in cards:
        cid = int(c["topic_id"])
        if cid is None:
            c["top_sessions"] = []
            continue

        sess_scores: List[Tuple[float, str]] = []
        for s in all_sessions:
            w_total = 0.0
            w_topic = 0.0

            for nb in G.neighbors(s):
                if not isinstance(nb, str):
                    continue
                if not (nb.startswith("q:") or nb.startswith("d:")):
                    continue

                w = float(G[s][nb].get("weight", 0.0))
                if w <= 0:
                    continue

                w_total += w
                nb_cid = int(node_to_comm.get(nb, -1))
                if nb_cid == cid:
                    w_topic += w

            if w_total <= 0 or w_topic <= 0:
                continue

            purity = float(w_topic / w_total)
            if purity >= float(purity_min):
                sess_scores.append((w_topic, s))

        sess_scores.sort(reverse=True, key=lambda t: t[0])
        top = [sid[2:] for _, sid in sess_scores[: int(k_sessions)]]
        c["top_sessions"] = top
        c["session_purity_min"] = float(purity_min)

    return cards


# -----------------------------
# Snapshot via facet scoring (broad + stable)
# -----------------------------

_STOP = {
    "the","a","an","and","or","to","of","in","for","on","at","near","me",
    "is","are","was","were","be","with","from","by","how","best","what","why",
}

def _tokenize(s: str) -> List[str]:
    xs = re.findall(r"[a-z0-9]+", (s or "").lower())
    return [t for t in xs if t and t not in _STOP and len(t) >= 2]


@dataclass
class FacetDef:
    name: str
    keywords: List[str]
    domain_hints: List[str]


FACETS: List[FacetDef] = [
    FacetDef("Travel", ["flight","hotel","visa","schengen","airport","things to do","itinerary","tickets","trip"], ["skyscanner","booking","expedia","airbnb","kayak"]),
    FacetDef(
        "Fashion & Shopping",
        ["dress","shoes","sneakers","heels","bag","jewelry","jewellery","outfit","skincare","jacket","coat","perfume","makeup"],
        ["selfridges","johnlewis","vogue.co.uk","harpersbazaar","whowhatwear.co.uk","tedbaker","thefoldlondon","houseoffraser","asos","zara","sephora","cultbeauty","net-a-porter"],
    ),
    FacetDef("Fitness", ["gym","workout","protein","running","lifting","cardio","planet fitness"], ["planetfitness","strava","myfitnesspal"]),
    FacetDef("Tech / AI", ["llm","openai","anthropic","pytorch","cuda","gpu","rag","agent","transformer"], ["github","arxiv","huggingface","paperswithcode"]),
    FacetDef(
        "Finance",
        ["loan","credit","mortgage","stocks","etf","tax","salary","interest rate","fintech","gross profit","annual report"],
        ["pitchbook","crunchbase","finextra","revolut","moneybox","chime","statista","deloitte","investopedia","nerdwallet","ft.com"],
    ),
    FacetDef("Food", ["recipe","cook","restaurant","pasta","calories","meal prep"], ["allrecipes","serious eats","ubereats","doordash"]),
    FacetDef("Health", ["symptoms","clinic","hospital","insurance","nhs","appointment"], ["mayoclinic","webmd","nhs.uk"]),
    FacetDef("Education / Career", ["resume","cv","interview","phd","application","deadline","fellowship"], ["linkedin","indeed","glassdoor","nyu.edu"]),
]


def build_snapshot_from_evidence(
    *,
    query_ctx: Dict[str, dict],
    cards: List[dict],
    max_evidence_per_facet: int = 4,
) -> dict:
    """
    Evidence-grounded snapshot: returns top facets with example evidence.
    """
    # gather evidence pool: top queries + their domains + titles
    queries: List[str] = []
    domains: List[str] = []
    titles: List[str] = []

    for c in cards[:10]:
        for q in (c.get("top_queries") or [])[:20]:
            queries.append(q)
            ctx = query_ctx.get(q, {})
            domains.extend(ctx.get("domains") or [])
            titles.extend(ctx.get("titles") or [])

    q_text = " ".join(queries)
    t_text = " ".join(titles)
    d_text = " ".join(domains)

    facet_scores: List[Tuple[float, FacetDef, List[str]]] = []

    for f in FACETS:
        s = 0.0
        evid: List[str] = []

        # keyword matches in queries/titles
        base_text = (q_text + " " + t_text).lower()
        for kw in f.keywords:
            if kw.lower() in base_text:
                s += 1.0
                if len(evid) < max_evidence_per_facet:
                    evid.append(f"keyword: {kw}")

        # domain hints
        for dh in f.domain_hints:
            if dh.lower() in d_text.lower():
                s += 1.3
                if len(evid) < max_evidence_per_facet:
                    evid.append(f"domain: {dh}")

        facet_scores.append((s, f, evid))

    facet_scores.sort(reverse=True, key=lambda t: t[0])
    top = [(s, f, ev) for (s, f, ev) in facet_scores if s >= 1.6][:5]

    if not top:
        return {
            "summary": "Not enough evidence to infer stable interests without an LLM.",
            "top_facets": [],
            "notes": "Graph-only snapshot uses conservative facet scoring.",
        }

    top_facets = []
    for s, f, ev in top:
        top_facets.append(
            {
                "facet": f.name,
                "score": float(s),
                "evidence": ev[:max_evidence_per_facet],
            }
        )

    return {
        "summary": "Top interests inferred from consistent query/domain/title patterns.",
        "top_facets": top_facets,
        "notes": "Graph-only snapshot uses conservative facet scoring (no invention).",
    }


# -----------------------------
# Output writers
# -----------------------------

def _save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_md(path: Path, payload: dict) -> None:
    cards = payload.get("topics") or []
    snap = payload.get("snapshot") or {}

    lines: List[str] = []
    lines.append("# Profile V2 (Graph-only)")
    lines.append("")
    lines.append("## Snapshot")
    lines.append("")
    lines.append(f"- {snap.get('summary', '')}")
    lines.append("")
    for f in (snap.get("top_facets") or []):
        lines.append(f"- **{f.get('facet')}** (score={f.get('score'):.2f}) — evidence: {', '.join(f.get('evidence') or [])}")
    if snap.get("notes"):
        lines.append("")
        lines.append(f"_Notes: {snap.get('notes')}_")
    lines.append("")

    lines.append("## Topics")
    lines.append("")
    for c in cards:
        lines.append(f"### {c.get('label')} (topic_id={c.get('topic_id')}, size={c.get('size')})")
        lines.append("")
        tq = c.get("top_queries") or []
        td = c.get("top_domains") or []
        ts = c.get("top_sessions") or []
        if td:
            lines.append("**Top domains**")
            for d in td[:10]:
                lines.append(f"- {d}")
            lines.append("")
        if tq:
            lines.append("**Top queries**")
            for q in tq[:15]:
                lines.append(f"- {q}")
            lines.append("")
        if ts:
            lines.append("**Representative sessions (purity-gated)**")
            for sid in ts[:10]:
                lines.append(f"- {sid}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Profile V2: graph-only (stable topics + facet snapshot)")
    ap.add_argument("--json", dest="json_path", type=str, default="search_history.json")
    ap.add_argument("--out", dest="out_dir", type=str, default="artifacts_profile_v2")
    ap.add_argument("--gap-min", dest="gap_minutes", type=int, default=30)

    ap.add_argument("--min-community-size", dest="min_size", type=int, default=8)
    ap.add_argument("--max-topics", dest="max_topics", type=int, default=8)
    ap.add_argument("--top-queries", dest="top_queries", type=int, default=10)
    ap.add_argument("--top-domains", dest="top_domains", type=int, default=8)

    ap.add_argument("--specificity-min", dest="spec_min", type=float, default=0.72)
    ap.add_argument("--session-purity-min", dest="purity_min", type=float, default=0.70)
    ap.add_argument("--sessions-per-topic", dest="k_sessions", type=int, default=8)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(args.json_path)
    events, _ = assign_sessions(events, gap_minutes=int(args.gap_minutes))

    qctx = build_query_context(events)
    G = build_history_graph(events)

    cards, node_to_comm = build_topic_cards(
        G=G,
        min_comm_size=int(args.min_size),
        max_topics=int(args.max_topics),
        top_queries=int(args.top_queries),
        top_domains=int(args.top_domains),
        specificity_min=float(args.spec_min),
    )
    cards = attach_representative_sessions(
        G=G,
        cards=cards,
        node_to_comm=node_to_comm,
        k_sessions=int(args.k_sessions),
        purity_min=float(args.purity_min),
    )

    snapshot = build_snapshot_from_evidence(query_ctx=qctx, cards=cards)

    payload = {
        "config": {
            "gap_minutes": int(args.gap_minutes),
            "min_community_size": int(args.min_size),
            "max_topics": int(args.max_topics),
            "top_queries": int(args.top_queries),
            "top_domains": int(args.top_domains),
            "specificity_min": float(args.spec_min),
            "session_purity_min": float(args.purity_min),
            "sessions_per_topic": int(args.k_sessions),
        },
        "n_events": len(events),
        "topics": cards,
        "snapshot": snapshot,
        "notes": "Graph-only profile intended to reduce topic mixing (no LLM).",
    }

    _save_json(out_dir / "profile_v2.json", payload)
    _write_md(out_dir / "PROFILE_V2.md", payload)

    print(f"Wrote: {out_dir / 'profile_v2.json'}")
    print(f"Wrote: {out_dir / 'PROFILE_V2.md'}")


if __name__ == "__main__":
    main()