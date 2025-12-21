from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import networkx as nx

from src.graph.build_graph import MIN_QUERY_QUALITY
from .config import SuitConfig
from .suits import ItemInfo, Suit
from .text import cosine, signature_token_set, item_overlap_score

def _dedupe_evidence(evs: List["Evidence"]) -> List["Evidence"]:
    seen: set[str] = set()
    out: List[Evidence] = []
    for e in evs:
        if not e.item_id or e.item_id in seen:
            continue
        seen.add(e.item_id)
        out.append(e)
    return out

def _is_generic_domain(domain: str, df_sessions: int, n_sessions: int, *, max_frac: float) -> bool:
    if not domain:
        return True
    frac = float(df_sessions) / float(max(1, n_sessions))
    return frac >= float(max_frac)

def _sessions_for_item(G: nx.Graph, item_id: str) -> List[str]:
    if not G.has_node(item_id):
        return []
    return [nb for nb in G.neighbors(item_id) if isinstance(nb, str) and nb.startswith("s:")]

@dataclass
class Evidence:
    item_id: str
    kind: str
    text: str
    cosine: float
    psignal: float
    df_sessions: int
    mass: float
    reason: str

def expand_suit(
    G: nx.Graph,
    suit: Suit,
    vecs: Dict[str, Dict[str, float]],
    item_info: Dict[str, ItemInfo],
    cfg: SuitConfig,
) -> Dict[str, object]:
    n_sessions = len([n for n in G.nodes if isinstance(n, str) and n.startswith("s:")])
    sig = signature_token_set(suit.centroid, k=24)

    scored_all: List[Tuple[float, str]] = []
    for item_id, v in vecs.items():
        sim = cosine(v, suit.centroid)
        if sim >= cfg.expand_sim_threshold:
            scored_all.append((float(sim), item_id))

    scored_all.sort(reverse=True, key=lambda t: t[0])

    primary: List[Evidence] = []
    for sim, item_id in scored_all[:400]:
        x = item_info.get(item_id)
        if not x:
            continue
        if x.kind == "query" and x.qquality < MIN_QUERY_QUALITY:
            continue
        primary.append(
            Evidence(
                item_id=item_id,
                kind=x.kind,
                text=x.text,
                cosine=float(sim),
                psignal=float(x.psignal),
                df_sessions=int(x.df_sessions),
                mass=float(x.mass),
                reason="semantic-affinity",
            )
        )

    sess_scores: Counter[str] = Counter()
    for ev in primary:
        if ev.kind != "query":
            continue
        for s in _sessions_for_item(G, ev.item_id):
            sess_scores[s] += float(ev.cosine)

    top_sessions = [s for s, _ in sess_scores.most_common(cfg.top_sessions_per_suit)]

    supporting: List[Evidence] = []
    for s in top_sessions:
        neigh = []
        for nb in G.neighbors(s):
            if not isinstance(nb, str) or not (nb.startswith("q:") or nb.startswith("d:")):
                continue
            x = item_info.get(nb)
            if not x:
                continue
            if x.kind == "query" and x.qquality < MIN_QUERY_QUALITY:
                continue
            v = vecs.get(nb) or {}
            sim = cosine(v, suit.centroid)
            neigh.append((float(sim), nb))

        for sim, nb in sorted(neigh, reverse=True, key=lambda t: t[0])[: cfg.session_expand_items]:
            x = item_info[nb]

            if x.kind == "domain":
                if _is_generic_domain(x.text, x.df_sessions, n_sessions, max_frac=cfg.domain_max_session_frac):
                    continue

            overlap = 0
            if x.kind == "query":
                overlap = item_overlap_score(x.text, sig)

            keep = (float(sim) >= float(cfg.session_gate_sim)) or (
                x.kind == "query" and overlap >= 1 and float(x.psignal) >= 0.35
            )
            if not keep:
                continue

            supporting.append(
                Evidence(
                    item_id=nb,
                    kind=x.kind,
                    text=x.text,
                    cosine=float(sim),
                    psignal=float(x.psignal),
                    df_sessions=int(x.df_sessions),
                    mass=float(x.mass),
                    reason="session-context-gated",
                )
            )

    supporting = _dedupe_evidence(supporting)

    q_primary = [e for e in primary if e.kind == "query"]
    d_primary = [e for e in primary if e.kind == "domain"]

    top_queries = [e.text for e in q_primary[: cfg.top_queries_per_suit]]

    top_domains = []
    for e in d_primary:
        if _is_generic_domain(e.text, e.df_sessions, n_sessions, max_frac=cfg.domain_max_session_frac):
            continue
        if int(e.df_sessions) < 2:
            continue
        top_domains.append(e.text)
        if len(top_domains) >= cfg.top_domains_per_suit:
            break

    if not top_domains:
        for e in d_primary:
            if _is_generic_domain(e.text, e.df_sessions, n_sessions, max_frac=cfg.domain_max_session_frac):
                continue
            top_domains.append(e.text)
            if len(top_domains) >= cfg.top_domains_per_suit:
                break

    if len(top_domains) < cfg.top_domains_per_suit:
        d2 = [e for e in supporting if e.kind == "domain"]
        for e in d2:
            if _is_generic_domain(e.text, e.df_sessions, n_sessions, max_frac=cfg.domain_max_session_frac):
                continue
            if int(e.df_sessions) < 2:
                continue
            if e.text not in top_domains:
                top_domains.append(e.text)
            if len(top_domains) >= cfg.top_domains_per_suit:
                break

    if top_queries and top_domains:
        para = (
            f"This theme is supported by searches like {', '.join(top_queries[:4])} "
            f"and activity on sites such as {', '.join(top_domains[:3])}."
        )
    elif top_queries:
        para = f"This theme is supported by searches like {', '.join(top_queries[:5])}."
    elif top_domains:
        para = f"This theme is supported by frequent activity on sites such as {', '.join(top_domains[:5])}."
    else:
        para = "This theme appears in the history, but the evidence is sparse."

    return {
        "suit_id": int(suit.suit_id),
        "label": suit.label,
        "mass": float(suit.mass),
        "top_queries": top_queries,
        "top_domains": top_domains,
        "top_sessions": [s[2:] if s.startswith("s:") else s for s in top_sessions],
        "paragraph": para,
        "evidence_primary": [asdict(e) for e in primary[:60]],
        "evidence_supporting": [asdict(e) for e in supporting[:80]],
    }