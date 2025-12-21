from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx

from src.graph.build_graph import MIN_QUERY_QUALITY
from .config import SuitConfig
from .text import build_tfidf, cosine, vec_add, vec_scale, top_tokens

@dataclass
class ItemInfo:
    item_id: str
    kind: str  # query|domain
    text: str
    psignal: float
    qquality: float
    df_sessions: int
    mass: float  # total session-edge weight

@dataclass
class Suit:
    suit_id: int
    label: str
    seed_item_ids: List[str]
    centroid: Dict[str, float]
    mass: float

def _extract_items_from_graph(G: nx.Graph) -> Tuple[Dict[str, ItemInfo], Dict[str, str]]:
    item_info: Dict[str, ItemInfo] = {}
    item_text: Dict[str, str] = {}

    for n, data in G.nodes(data=True):
        if not isinstance(n, str):
            continue
        ntype = data.get("ntype")
        if ntype not in {"query", "domain"}:
            continue

        sess = [nb for nb in G.neighbors(n) if isinstance(nb, str) and nb.startswith("s:")]
        df_sessions = len(set(sess))

        mass = 0.0
        for s in sess:
            try:
                mass += float(G[s][n].get("weight", 0.0))
            except Exception:
                continue

        if ntype == "query":
            q = n.split(":", 1)[1]
            ps = float(data.get("psignal", 0.0))
            qq = float(data.get("qquality", 1.0))
            item_info[n] = ItemInfo(n, "query", q, ps, qq, int(df_sessions), float(mass))
            item_text[n] = q
        else:
            d = n.split(":", 1)[1]
            item_info[n] = ItemInfo(n, "domain", d, 1.0, 1.0, int(df_sessions), float(mass))
            item_text[n] = f"site {d}"

    return item_info, item_text

def _seed_score(x: ItemInfo, max_df: int, max_mass: float) -> float:
    if x.kind != "query":
        return 0.0
    if x.qquality < MIN_QUERY_QUALITY:
        return 0.0
    if x.psignal <= 0:
        return 0.0

    persistence = float(x.df_sessions) / float(max(1, max_df))
    mass_n = float(x.mass) / float(max(1e-9, max_mass))
    return float(x.psignal) * 0.5 * (persistence + mass_n)

def discover_suits(G: nx.Graph, cfg: SuitConfig) -> Tuple[List[Suit], Dict[str, Dict[str, float]], Dict[str, ItemInfo]]:
    item_info, item_text = _extract_items_from_graph(G)
    vecs, _idf, _extra_stop = build_tfidf(item_text)

    queries = [x for x in item_info.values() if x.kind == "query"]
    max_df = max([x.df_sessions for x in queries], default=1)
    max_mass = max([x.mass for x in queries], default=1.0)

    scored: List[Tuple[float, str]] = []
    for item_id, x in item_info.items():
        if x.kind != "query":
            continue
        if x.psignal < cfg.seed_psignal_min:
            continue
        s = _seed_score(x, max_df=max_df, max_mass=max_mass)
        if s <= 0:
            continue
        scored.append((float(s), item_id))

    scored.sort(reverse=True, key=lambda t: t[0])
    scored = scored[: int(cfg.seed_max_items)]

    suits: List[Suit] = []

    for _score, item_id in scored:
        v = vecs.get(item_id) or {}
        if not v:
            continue

        best_idx = -1
        best_sim = -1.0
        for i, suit in enumerate(suits):
            sim = cosine(v, suit.centroid)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx >= 0 and best_sim >= cfg.sim_threshold:
            s = suits[best_idx]
            new_centroid = dict(s.centroid)
            vec_add(new_centroid, v, w=1.0)
            # keep your exact original behavior:
            new_centroid = vec_scale(new_centroid, 1.0 / float(len(s.seed_item_ids) + 1))

            s.seed_item_ids.append(item_id)
            s.centroid = new_centroid
            s.mass += float(item_info[item_id].mass)
        else:
            label = " ".join(top_tokens(v, 4)) or "misc"
            suits.append(
                Suit(
                    suit_id=len(suits),
                    label=label.title(),
                    seed_item_ids=[item_id],
                    centroid=dict(v),
                    mass=float(item_info[item_id].mass),
                )
            )

    suits.sort(key=lambda s: s.mass, reverse=True)
    suits = suits[: int(cfg.max_suits)]
    for i, s in enumerate(suits):
        s.suit_id = int(i)

    return suits, vecs, item_info