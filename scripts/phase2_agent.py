from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import os
import re
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from src.ingest.parse_takeout import load_events
from src.ingest.sessionize import assign_sessions
from src.graph.build_graph import build_history_graph


# -----------------------------
# IO
# -----------------------------

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))



def _json_default(o: Any):
    # Make common non-JSON types serializable
    if isinstance(o, set):
        return sorted(list(o))
    if isinstance(o, Path):
        return str(o)
    # numpy scalars if present
    try:
        import numpy as np  # type: ignore
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
    except Exception:
        pass
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class EvidenceItem:
    community_id: int
    session_id: str
    node_id: str          # "q:..." or "d:..." or "e:..."
    ntype: str            # "query" | "domain" | "entity"
    weight: float
    note: str


@dataclass
class SessionCandidate:
    session_node: str              # "s:s0001"
    session_id: str                # "s0001"
    comm_weights: Dict[int, float] # community -> mass
    domains: Set[str]              # domain strings
    queries: Set[str]              # query strings
    entities: Set[str]             # semantic phrases/entities
    strength: float                # sum of weights session->content


@dataclass
class AgentConfig:
    budget: int = 40
    min_communities: int = 6

    max_sessions: int = 12
    max_items_per_session: int = 8

    per_domain_cap: int = 5

    # Coverage saturation (higher tau => need more mass to count as "covered")
    tau: float = 12.0

    # Penalize overlap with already-selected domains
    diversity_lambda: float = 0.35

    # Penalize picking sessions that would exceed per-domain cap
    domain_cap_gamma: float = 0.20


@dataclass
class StepTrace:
    step: int
    picked_session: str
    gain: float
    cov_gain: float
    overlap: float
    cap_over_frac: float
    covered_communities: List[int]


@dataclass
class TuningTrial:
    trial: int
    config: dict
    reward: float
    coverage: int
    diversity: float
    stability: float
    cap_violations: int


# -----------------------------
# Graph helpers
# -----------------------------

def ntype_of_node(G: nx.Graph, node_id: str) -> str:
    return G.nodes[node_id].get("ntype", "unknown")


def is_utility_query_node(G: nx.Graph, node_id: str) -> bool:
    if not (isinstance(node_id, str) and node_id.startswith("q:")):
        return False
    return G.nodes[node_id].get("qclass") == "utility"


def query_quality_of_node(G: nx.Graph, node_id: str) -> float:
    if not (isinstance(node_id, str) and node_id.startswith("q:")):
        return 1.0
    try:
        return float(G.nodes[node_id].get("qquality", 1.0))
    except Exception:
        return 1.0


def entity_quality_of_node(G: nx.Graph, node_id: str) -> float:
    if not (isinstance(node_id, str) and node_id.startswith("e:")):
        return 1.0
    try:
        return float(G.nodes[node_id].get("equality", 1.0))
    except Exception:
        return 1.0


def is_session(node_id: str) -> bool:
    return isinstance(node_id, str) and node_id.startswith("s:")


def session_short_id(session_node_id: str) -> str:
    # "s:s0001" -> "s0001"
    return session_node_id[2:] if session_node_id.startswith("s:") else session_node_id


def _strip_prefix(n: str) -> str:
    return n.split(":", 1)[1] if ":" in n else n


def session_candidate_to_dict(s: SessionCandidate) -> dict:
    # sets -> lists for JSON
    return {
        "session_node": s.session_node,
        "session_id": s.session_id,
        "comm_weights": {int(k): float(v) for k, v in s.comm_weights.items()},
        "domains": sorted(list(s.domains)),
        "queries": sorted(list(s.queries)),
        "entities": sorted(list(s.entities)),
        "strength": float(s.strength),
    }


def build_session_candidates(G: nx.Graph, node_to_comm: Dict[str, int]) -> List[SessionCandidate]:
    cands: List[SessionCandidate] = []

    for s in G.nodes:
        if not is_session(s):
            continue

        comm_weights: Dict[int, float] = defaultdict(float)
        doms: Set[str] = set()
        qs: Set[str] = set()
        ents: Set[str] = set()
        strength = 0.0

        for nb in G.neighbors(s):
            t = ntype_of_node(G, nb)
            if t not in {"domain", "query", "entity"}:
                continue
            if t == "query":
                if is_utility_query_node(G, nb) or query_quality_of_node(G, nb) < 0.25:
                    continue
            if t == "entity":
                if entity_quality_of_node(G, nb) < 0.25:
                    continue

            cid = node_to_comm.get(nb)
            if cid is None or int(cid) == -1:
                continue

            w = float(G[s][nb].get("weight", 1.0))
            strength += w
            comm_weights[int(cid)] += w

            if t == "domain":
                doms.add(_strip_prefix(nb))
            elif t == "query":
                qs.add(_strip_prefix(nb))
            else:
                ents.add(_strip_prefix(nb))

        if not comm_weights:
            continue

        cands.append(
            SessionCandidate(
                session_node=s,
                session_id=session_short_id(s),
                comm_weights=dict(comm_weights),
                domains=doms,
                queries=qs,
                entities=ents,
                strength=float(strength),
            )
        )

    cands.sort(key=lambda x: x.strength, reverse=True)
    return cands


# -----------------------------
# Objective pieces
# -----------------------------

def _coverage_value(mass: float, tau: float) -> float:
    if tau <= 0:
        return 1.0
    return min(mass / tau, 1.0)


def _marginal_coverage_gain(curr_mass: Dict[int, float], delta: Dict[int, float], tau: float) -> float:
    gain = 0.0
    for c, d in delta.items():
        before = _coverage_value(curr_mass.get(c, 0.0), tau)
        after = _coverage_value(curr_mass.get(c, 0.0) + d, tau)
        gain += (after - before)
    return gain


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


# -----------------------------
# SEARCH-R1 vibe: constrained “agent” with trace
# -----------------------------

def select_sessions_submodular(
    candidates: List[SessionCandidate],
    cfg: AgentConfig,
) -> Tuple[List[SessionCandidate], List[StepTrace]]:
    selected: List[SessionCandidate] = []
    trace: List[StepTrace] = []

    covered_mass: Dict[int, float] = {}
    domain_counts: Counter[str] = Counter()
    selected_domain_union: Set[str] = set()

    def covered_communities() -> Set[int]:
        return {c for c, m in covered_mass.items() if _coverage_value(m, cfg.tau) > 0.0}

    step = 0

    while len(selected) < cfg.max_sessions:
        best: Optional[SessionCandidate] = None
        best_gain = -1e9
        best_cov_gain = 0.0
        best_overlap = 0.0
        best_cap_over_frac = 0.0

        pool = candidates[: max(2000, cfg.max_sessions * 800)]

        for cand in pool:
            if cand in selected:
                continue

            cov_gain = _marginal_coverage_gain(covered_mass, cand.comm_weights, cfg.tau)

            overlap = _jaccard(cand.domains, selected_domain_union)
            div_pen = cfg.diversity_lambda * overlap

            cap_over = 0
            for d in cand.domains:
                if domain_counts[d] >= cfg.per_domain_cap:
                    cap_over += 1
            cap_over_frac = cap_over / max(1, len(cand.domains))
            cap_pen = cfg.domain_cap_gamma * cap_over_frac

            total = cov_gain - div_pen - cap_pen

            if total > best_gain + 1e-9 or (
                abs(total - best_gain) <= 1e-9 and best is not None and cand.strength > best.strength
            ):
                best_gain = total
                best = cand
                best_cov_gain = cov_gain
                best_overlap = overlap
                best_cap_over_frac = cap_over_frac

        if best is None:
            break

        curr_cov = len(covered_communities())
        if curr_cov >= cfg.min_communities and best_gain < 0.02:
            break

        selected.append(best)
        for c, w in best.comm_weights.items():
            covered_mass[c] = covered_mass.get(c, 0.0) + w
        for d in best.domains:
            domain_counts[d] += 1
        selected_domain_union |= set(best.domains)

        trace.append(
            StepTrace(
                step=step,
                picked_session=best.session_id,
                gain=float(best_gain),
                cov_gain=float(best_cov_gain),
                overlap=float(best_overlap),
                cap_over_frac=float(best_cap_over_frac),
                covered_communities=sorted(list(covered_communities())),
            )
        )
        step += 1

    return selected, trace


# -----------------------------
# Evidence extraction (for audit) + Intent cards (for humans)
# -----------------------------

def extract_evidence(
    G: nx.Graph,
    node_to_comm: Dict[str, int],
    selected_sessions: List[SessionCandidate],
    cfg: AgentConfig,
) -> Tuple[List[EvidenceItem], Set[int]]:
    evidence: List[EvidenceItem] = []
    used_domain_counts: Counter[str] = Counter()
    covered: Set[int] = set()

    def remaining() -> int:
        return cfg.budget - len(evidence)

    for cand in selected_sessions:
        if remaining() <= 0:
            break

        s = cand.session_node
        sid = cand.session_id

        neigh: List[Tuple[float, str, str, int]] = []
        for nb in G.neighbors(s):
            t = ntype_of_node(G, nb)
            if t not in {"domain", "query", "entity"}:
                continue
            if t == "query":
                if is_utility_query_node(G, nb) or query_quality_of_node(G, nb) < 0.25:
                    continue
            if t == "entity":
                if entity_quality_of_node(G, nb) < 0.25:
                    continue
            c = node_to_comm.get(nb)
            if c is None or int(c) == -1:
                continue
            w = float(G[s][nb].get("weight", 1.0))
            neigh.append((w, nb, t, int(c)))

        neigh.sort(reverse=True, key=lambda x: x[0])

        picked_here = 0
        for w, nb, t, c in neigh:
            if remaining() <= 0 or picked_here >= cfg.max_items_per_session:
                break

            if t == "domain":
                dom = _strip_prefix(nb)
                if used_domain_counts[dom] >= cfg.per_domain_cap:
                    continue
                used_domain_counts[dom] += 1

            evidence.append(
                EvidenceItem(
                    community_id=c,
                    session_id=sid,
                    node_id=nb,
                    ntype=t,
                    weight=float(w),
                    note="from-selected-session",
                )
            )
            picked_here += 1
            covered.add(c)

    return evidence, covered


def _query_df_from_graph(G: nx.Graph) -> Counter[str]:
    q_df: Counter[str] = Counter()
    for n in G.nodes:
        if not isinstance(n, str) or not n.startswith("q:"):
            continue
        if is_utility_query_node(G, n) or query_quality_of_node(G, n) < 0.25:
            continue
        sess = {nb for nb in G.neighbors(n) if isinstance(nb, str) and nb.startswith("s:")}
        q_df[_strip_prefix(n)] = len(sess)
    return q_df


def _entity_df_from_graph(G: nx.Graph) -> Counter[str]:
    e_df: Counter[str] = Counter()
    for n in G.nodes:
        if not isinstance(n, str) or not n.startswith("e:"):
            continue
        if entity_quality_of_node(G, n) < 0.25:
            continue
        sess = {nb for nb in G.neighbors(n) if isinstance(nb, str) and nb.startswith("s:")}
        e_df[_strip_prefix(n)] = len(sess)
    return e_df



def _guess_topic_label(domains: List[str], queries: List[str], entities: Optional[List[str]] = None) -> str:
    d = " ".join(domains).lower()
    q = " ".join(queries).lower()
    e = " ".join(entities or []).lower()

    if any(x in d for x in ["tripadvisor", "booking", "skyscanner", "makemytrip", "expedia"]):
        return "Travel planning"
    if any(x in d for x in ["github", "stackoverflow", "pypi", "developer.apple"]):
        return "Software engineering / ML"
    if any(x in d for x in ["vogue", "selfridges", "harpers", "whowhatwear", "net-a-porter", "mango", "burberry"]):
        return "Fashion / shopping"
    if any(x in d for x in ["bbc", "guardian", "ft.com", "reuters", "news.sky"]):
        return "News / geopolitics"
    if any(x in d for x in ["barclays", "hsbc", "monzo", "revolut", "investopedia", "bloomberg"]):
        return "Finance / banking"
    if any(x in d for x in ["nhs", "pubmed", "ncbi", "mayo", "webmd"]):
        return "Health / medicine"

    # Entities can sharpen weak domain signals
    if any(x in e for x in ["flight", "hotel", "itinerary", "visa", "passport"]):
        return "Travel / logistics"
    if any(x in e for x in ["mortgage", "salary", "after tax", "valuation", "discounted cash flow", "dcf"]):
        return "Finance / life admin"

    if any(x in q for x in ["mortgage", "after tax", "salary", "visa", "settlement scheme"]):
        return "Life admin (jobs/visa/finance)"
    return "Mixed interest"


def build_intent_cards(
    G: nx.Graph,
    node_to_comm: Dict[str, int],
    comm_summaries: List[dict],
    session_trails: dict,
    selected_sessions: List[SessionCandidate],
    *,
    per_session_max_intents: int = 2,
    min_comm_mass_frac: float = 0.28,
) -> List[dict]:
    """Build intent cards for humans.

    IMPORTANT: Cards are generated per (session, dominant community).

    This avoids “mixed” cards where multiple unrelated themes are merged into a single label.
    """

    q_df = _query_df_from_graph(G)
    e_df = _entity_df_from_graph(G)
    n_sess = max(1, sum(1 for n in G.nodes if isinstance(n, str) and n.startswith("s:")))

    def q_idf(q: str) -> float:
        df = q_df.get(q, 0)
        return math.log((1.0 + n_sess) / (1.0 + df)) + 1.0

    def e_idf(e: str) -> float:
        df = e_df.get(e, 0)
        return math.log((1.0 + n_sess) / (1.0 + df)) + 1.0

    # Index summaries by community id for fast lookup
    summ_by_id: Dict[int, dict] = {}
    for s in comm_summaries:
        try:
            summ_by_id[int(s.get("community_id"))] = s
        except Exception:
            continue

    def _dedup(xs: List[str]) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for x in xs:
            if not x:
                continue
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    cards: List[dict] = []

    for cand in selected_sessions:
        s = cand.session_node
        sid = cand.session_id

        # Gather neighbors with community ids
        neigh: List[Tuple[float, str, str, int]] = []
        for nb in G.neighbors(s):
            t = ntype_of_node(G, nb)
            if t not in {"domain", "query", "entity"}:
                continue
            if t == "query":
                if is_utility_query_node(G, nb) or query_quality_of_node(G, nb) < 0.25:
                    continue
            if t == "entity":
                if entity_quality_of_node(G, nb) < 0.25:
                    continue
            cid = node_to_comm.get(nb)
            if cid is None or int(cid) == -1:
                continue
            w = float(G[s][nb].get("weight", 1.0))
            neigh.append((w, nb, t, int(cid)))

        if not neigh:
            continue

        # Compute community mass for this session
        comm_mass: Dict[int, float] = defaultdict(float)
        for w, _, _, c in neigh:
            comm_mass[int(c)] += float(w)

        total_mass = sum(comm_mass.values()) or 1.0
        ranked = sorted(comm_mass.items(), key=lambda kv: kv[1], reverse=True)

        # Choose dominant communities for this session
        chosen: List[int] = []
        for c, m in ranked:
            if m / total_mass < float(min_comm_mass_frac):
                continue
            chosen.append(int(c))
            if len(chosen) >= int(per_session_max_intents):
                break

        # Fallback to top community if threshold filters all
        if not chosen and ranked:
            chosen = [int(ranked[0][0])]

        # Representative titles are session-level (still useful even if a session has 2 themes)
        preview = session_trails.get(sid) or {}
        reps = preview.get("representative_titles", []) if isinstance(preview, dict) else []

        # Build one card per (session, community)
        for c in chosen:
            bucket = [(w, nb, t) for (w, nb, t, cid) in neigh if int(cid) == int(c)]
            if not bucket:
                continue
            bucket.sort(reverse=True, key=lambda x: x[0])

            # Rank queries/entities by (edge_weight * idf) to favor distinctive intent
            q_scored = [(w * q_idf(_strip_prefix(nb)), w, _strip_prefix(nb)) for (w, nb, t) in bucket if t == "query"]
            q_scored.sort(reverse=True, key=lambda x: x[0])

            d_scored = [(w, _strip_prefix(nb)) for (w, nb, t) in bucket if t == "domain"]
            d_scored.sort(reverse=True, key=lambda x: x[0])

            e_scored = [(w * e_idf(_strip_prefix(nb)), w, _strip_prefix(nb)) for (w, nb, t) in bucket if t == "entity"]
            e_scored.sort(reverse=True, key=lambda x: x[0])

            top_queries = _dedup([q for _, _, q in q_scored[:8]])[:4]
            top_domains = _dedup([d for _, d in d_scored[:6]])[:3]
            top_entities = _dedup([e for _, _, e in e_scored[:10]])[:5]

            # Label from that single community summary (not merged across communities)
            summ = summ_by_id.get(int(c))
            if summ:
                label = _guess_topic_label(
                    (summ.get("top_domains", []) or [])[:6],
                    (summ.get("top_queries", []) or [])[:6],
                    (summ.get("top_entities", []) or [])[:6],
                )
            else:
                label = _guess_topic_label(top_domains, top_queries, top_entities)

            cards.append(
                {
                    "session_id": sid,
                    "community_id": int(c),
                    "label": label,
                    "top_queries": top_queries,
                    "top_entities": top_entities,
                    "supporting_domains": top_domains,
                    # strength is community-specific mass so sorting produces coherent rabbit holes
                    "strength": float(comm_mass.get(int(c), 0.0)),
                    "representative_titles": reps[:6],
                }
            )

    cards.sort(key=lambda x: float(x.get("strength", 0.0)), reverse=True)
    return cards


def build_profile_paragraph(covered: Set[int], comm_summaries: List[dict]) -> str:
    by_id = {int(c["community_id"]): c for c in comm_summaries}
    covered_sorted = sorted(list(covered), key=lambda cid: int(by_id.get(cid, {}).get("size", 0)), reverse=True)

    snippets = []
    for cid in covered_sorted[:6]:
        summ = by_id.get(cid)
        if not summ:
            continue
        top_d = summ.get("top_domains", [])[:5]
        top_q = summ.get("top_queries", [])[:5]
        label = _guess_topic_label(top_d, top_q, summ.get("top_entities", [])[:6])

        dom_ex = ", ".join(top_d[:3])
        qry_ex = ", ".join(top_q[:3])
        if dom_ex and qry_ex:
            snippets.append(f"{label} (e.g., {dom_ex}; searches like {qry_ex})")
        elif dom_ex:
            snippets.append(f"{label} (e.g., {dom_ex})")
        elif qry_ex:
            snippets.append(f"{label} (searches like {qry_ex})")
        else:
            snippets.append(label)

    if not snippets:
        return "The browsing history suggests recurring interests, but the signal is too sparse to summarize confidently."

    return "This browsing history suggests recurring themes across " + ", ".join(snippets) + "."


# -----------------------------
# Scoring + tuning (bandit style)
# -----------------------------

def _domain_entropy(evidence: List[EvidenceItem]) -> float:
    doms = [(_strip_prefix(e.node_id)) for e in evidence if e.ntype == "domain"]
    if not doms:
        return 0.0
    ctr = Counter(doms)
    total = sum(ctr.values())
    ent = 0.0
    for c in ctr.values():
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return float(ent)


def score_result(cfg: AgentConfig, result: dict) -> Tuple[float, dict]:
    coverage = len(result.get("covered_communities", []))
    evidence = [EvidenceItem(**e) for e in result.get("evidence", [])]

    diversity = _domain_entropy(evidence)

    cap = cfg.per_domain_cap
    doms = [(_strip_prefix(e.node_id)) for e in evidence if e.ntype == "domain"]
    ctr = Counter(doms)
    cap_violations = sum(max(0, v - cap) for v in ctr.values())

    reward = (2.0 * coverage) + (1.0 * diversity) - (1.5 * cap_violations)

    metrics = {
        "coverage": int(coverage),
        "diversity": float(diversity),
        "cap_violations": int(cap_violations),
        "reward": float(reward),
    }
    return float(reward), metrics


def _subsample_events(events: List, frac: float, seed: int) -> List:
    rng = random.Random(seed)
    if frac >= 1.0:
        return list(events)
    n = max(1, int(len(events) * frac))
    idx = list(range(len(events)))
    rng.shuffle(idx)
    keep = set(idx[:n])
    return [e for i, e in enumerate(events) if i in keep]


def stability_score(base_cards: List[dict], other_cards: List[dict]) -> float:
    def sig(cards: List[dict]) -> Set[str]:
        s: Set[str] = set()
        for c in cards[:8]:
            for q in c.get("top_queries", [])[:3]:
                if isinstance(q, str) and q:
                    s.add(q)
        return s

    a = sig(base_cards)
    b = sig(other_cards)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(len(a & b)) / float(len(a | b))


def tune_parameters(
    events: List,
    node_to_comm: Dict[str, int],
    comm_summaries: List[dict],
    session_trails: dict,
    trials: int,
    seed: int,
) -> Tuple[AgentConfig, List[TuningTrial]]:
    rng = random.Random(seed)

    def sample_cfg() -> AgentConfig:
        return AgentConfig(
            budget=40,
            min_communities=6,
            max_sessions=rng.choice([8, 10, 12, 14]),
            max_items_per_session=rng.choice([6, 8, 10]),
            per_domain_cap=rng.choice([3, 4, 5]),
            tau=rng.choice([8.0, 10.0, 12.0, 14.0, 16.0]),
            diversity_lambda=rng.choice([0.15, 0.25, 0.35, 0.45]),
            domain_cap_gamma=rng.choice([0.10, 0.20, 0.30]),
        )

    best_cfg: Optional[AgentConfig] = None
    best_reward = -1e18
    all_trials: List[TuningTrial] = []

    G_full = build_history_graph(events)

    for t in range(trials):
        cfg = sample_cfg()
        res = run_phase2(G_full, node_to_comm, comm_summaries, session_trails, cfg)

        base_cards = res.get("intent_cards", [])
        st_scores = []
        for k in range(2):
            sub = _subsample_events(events, 0.7, seed + 10_000 + (t * 10) + k)
            G_sub = build_history_graph(sub)
            sub_res = run_phase2(G_sub, node_to_comm, comm_summaries, session_trails, cfg)
            st_scores.append(stability_score(base_cards, sub_res.get("intent_cards", [])))
        stab = float(statistics.mean(st_scores)) if st_scores else 0.0

        reward, metrics = score_result(cfg, res)
        reward2 = reward + (2.0 * stab)

        all_trials.append(
            TuningTrial(
                trial=t,
                config=asdict(cfg),
                reward=float(reward2),
                coverage=int(metrics["coverage"]),
                diversity=float(metrics["diversity"]),
                stability=float(stab),
                cap_violations=int(metrics["cap_violations"]),
            )
        )

        if reward2 > best_reward:
            best_reward = reward2
            best_cfg = cfg

    assert best_cfg is not None
    all_trials.sort(key=lambda x: x.reward, reverse=True)
    return best_cfg, all_trials



# -----------------------------
# Optional LLM hook (OFF by default)
# -----------------------------

def render_summary_short(profile_paragraph: str, intent_cards: List[dict]) -> str:
    """2–4 sentences, reviewer-friendly."""
    base = (profile_paragraph or "").strip()
    # Dedup theme labels
    themes: List[str] = []
    for c in intent_cards[:6]:
        lbl = (c.get("label") or "").strip()
        if not lbl:
            continue
        if lbl.lower() in {"mixed interest", "mixed"}:
            continue
        if lbl not in themes:
            themes.append(lbl)
    if themes:
        t = ", ".join(themes[:4])
        if base:
            return (base + f"\n\nThemes: {t}.").strip() + "\n"
        return (f"Themes: {t}.").strip() + "\n"
    return (base or "").strip() + "\n"


def render_summary_expanded(profile_paragraph: str, intent_cards: List[dict]) -> str:
    """1–2 short paragraphs + grounded bullets."""
    base = (profile_paragraph or "").strip()
    lines = [base, "", "## Evidence-backed themes"] if base else ["## Evidence-backed themes"]
    for c in intent_cards[:8]:
        lbl = (c.get("label") or "Intent").strip()
        qs = ", ".join((c.get("top_queries") or [])[:3])
        dom = ", ".join((c.get("supporting_domains") or [])[:3])
        lines.append(f"- **{lbl}** — queries: {qs} | domains: {dom}")
    return "\n".join(lines).strip() + "\n"



def _safe_compact_text(s: str, max_len: int = 220) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


# --- LLM JSON postprocessing helpers ---

def _parse_json_object_from_text(text: str) -> dict:
    """Parse a JSON object from an LLM response.

    Handles common cases:
      - empty/whitespace output
      - ```json fenced blocks
      - extra preface/epilogue text around a JSON object
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("LLM returned empty text")

    # Strip common fenced code blocks
    if raw.startswith("```"):
        lines = raw.splitlines()
        # drop first fence line
        if lines:
            lines = lines[1:]
        # drop trailing fence if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    # If the model added commentary, extract the first top-level JSON object
    if not raw.startswith("{"):
        i = raw.find("{")
        j = raw.rfind("}")
        if i != -1 and j != -1 and j > i:
            raw = raw[i : j + 1].strip()

    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("LLM JSON was not an object")
    return obj


def _anthropic_list_models(limit: int = 200, timeout: int = 30) -> List[str]:
    """Return model IDs visible to this API key (ordered newest-first per Anthropic)."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    url = f"https://api.anthropic.com/v1/models?limit={int(limit)}"
    req = urllib.request.Request(url, method="GET")
    req.add_header("x-api-key", api_key)
    req.add_header("anthropic-version", "2023-06-01")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic HTTPError {e.code} (list models): {err}")

    obj = json.loads(body)
    data = obj.get("data", [])
    ids: List[str] = []
    for it in data:
        if isinstance(it, dict) and isinstance(it.get("id"), str):
            ids.append(it["id"])
    return ids


def _anthropic_pick_model(requested: Optional[str] = None) -> str:
    """Pick a working model.

    If `requested` is provided, we try it first.
    Otherwise, we prefer a Sonnet-class model, then fall back to the first available.
    """
    ids = _anthropic_list_models()
    if not ids:
        raise RuntimeError("Anthropic Models API returned no models for this key")

    if requested:
        if requested in ids:
            return requested
        # Allow aliases like `claude-sonnet-4-5` that may not appear in /v1/models.
        # We'll still attempt them; if they 404, we'll fall back.
        return requested

    # Prefer commonly useful aliases/IDs, but only if present.
    preferred = [
        "claude-sonnet-4-5",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-5",
        "claude-haiku-4-5",
    ]
    for p in preferred:
        if p in ids:
            return p

    # Heuristic: pick the newest Sonnet-like model if present.
    for mid in ids:
        if "sonnet" in mid:
            return mid

    # Otherwise take the newest model overall.
    return ids[0]


def _anthropic_call(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 700,
    timeout: int = 45,
) -> str:
    """Minimal Anthropic Messages API call using stdlib.

    Requires env var: ANTHROPIC_API_KEY

    If `model` is None, we auto-pick a model your key can access.
    If `model` is provided but not accessible, we fall back to the newest available model.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    chosen = _anthropic_pick_model(model)

    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": chosen,
        "max_tokens": int(max_tokens),
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("x-api-key", api_key)
    req.add_header("anthropic-version", "2023-06-01")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        # If the requested/alias model is unavailable, fall back once to a known-available model.
        if e.code == 404 and chosen != None:
            try:
                fallback_ids = _anthropic_list_models()
                if fallback_ids:
                    fallback = fallback_ids[0]
                    payload["model"] = fallback
                    data2 = json.dumps(payload).encode("utf-8")
                    req2 = urllib.request.Request(url, data=data2, method="POST")
                    req2.add_header("Content-Type", "application/json")
                    req2.add_header("x-api-key", api_key)
                    req2.add_header("anthropic-version", "2023-06-01")
                    with urllib.request.urlopen(req2, timeout=timeout) as resp2:
                        body2 = resp2.read().decode("utf-8")
                    obj2 = json.loads(body2)
                    blocks2 = obj2.get("content", [])
                    texts2: List[str] = []
                    for b in blocks2:
                        if isinstance(b, dict) and b.get("type") == "text":
                            texts2.append(b.get("text", ""))
                    out2 = "\n".join(texts2).strip()
                    if not out2:
                        preview2 = (body2 or "")[:600]
                        raise RuntimeError(f"Anthropic returned no text content. Raw response preview: {preview2}")
                    return out2
            except Exception:
                pass

        raise RuntimeError(
            f"Anthropic HTTPError {e.code}: {err}\n"
            f"(Attempted model: {chosen!r}. Tip: run with --anthropic-model <id> or check /v1/models.)"
        )

    obj = json.loads(body)
    blocks = obj.get("content", [])
    texts: List[str] = []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "text":
            texts.append(b.get("text", ""))

    out = "\n".join(texts).strip()
    if not out:
        # Surface a useful error so we don't later crash with json.loads('')
        preview = (body or "")[:600]
        raise RuntimeError(f"Anthropic returned no text content. Raw response preview: {preview}")
    return out


def apply_label_map(intent_cards: List[dict], label_map: dict) -> List[dict]:
    out = []
    for c in intent_cards:
        sid = str(c.get("session_id"))
        new = dict(c)
        v = label_map.get(sid) if isinstance(label_map, dict) else None
        if isinstance(v, str) and v.strip():
            new["label"] = v.strip()
        out.append(new)
    return out


def llm_postprocess_cards_and_summaries(intent_cards: List[dict], profile_paragraph: str, anthropic_model: Optional[str] = None) -> dict:
    """Return {label_map, summary_small, summary_big, truths}. Uses only already-selected evidence."""
    evidence = []
    for c in intent_cards[:10]:
        evidence.append(
            {
                "session_id": c.get("session_id"),
                "label": c.get("label"),
                "top_queries": (c.get("top_queries") or [])[:6],
                "supporting_domains": (c.get("supporting_domains") or [])[:6],
                "trail": [_safe_compact_text(t, 140) for t in (c.get("representative_titles") or [])[:4]],
            }
        )

    prompt = (
        "You are refining a user's browsing profile produced by a deterministic graph agent.\n"
        "You ONLY have the selected intent cards below as evidence.\n\n"
        "Tasks:\n"
        "1) Provide improved short labels per session (3–6 words).\n"
        "2) Write a SHORT summary (2–4 sentences).\n"
        "3) Write an EXPANDED summary (1–2 short paragraphs) + a bullet list of 4–7 stable truths.\n\n"
        "Constraints:\n"
        "- Do not invent facts. If uncertain, hedge.\n"
        "- Prefer concrete themes; avoid 'mixed interest'.\n"
        "- Output MUST be a single JSON object (no markdown, no code fences, no commentary).\n"
        "- Keys must be exactly: label_map, summary_small, summary_big, truths.\n\n"
        f"BASE_PARAGRAPH:\n{profile_paragraph}\n\n"
        f"EVIDENCE_JSON:\n{json.dumps(evidence, ensure_ascii=False)}\n"
    )

    text = _anthropic_call(prompt, model=anthropic_model)
    try:
        return _parse_json_object_from_text(text)
    except Exception as e:
        preview = _safe_compact_text(text, 600)
        raise RuntimeError(f"Failed to parse LLM JSON: {e}. LLM output preview: {preview}")


def maybe_llm_summarize(cards: List[dict], fallback_paragraph: str, provider: str = "off", anthropic_model: Optional[str] = None) -> Tuple[dict, List[dict]]:
    """Optional LLM post-processing (bounded and auditable).

    Returns (llm_out, updated_cards) where llm_out is a dict with:
      - summary_small
      - summary_big
      - truths
      - label_map (optional)

    IMPORTANT: LLM only sees already-selected evidence (cards). It does NOT choose evidence.
    """
    if provider in {"off", "none", ""}:
        llm_out = {
            "summary_small": render_summary_short(fallback_paragraph, cards).strip(),
            "summary_big": render_summary_expanded(fallback_paragraph, cards).strip(),
            "truths": [],
        }
        return llm_out, cards

    if provider != "anthropic":
        llm_out = {
            "summary_small": render_summary_short(fallback_paragraph, cards).strip(),
            "summary_big": render_summary_expanded(fallback_paragraph, cards).strip(),
            "truths": [],
            "warning": f"provider '{provider}' not implemented; using deterministic summaries",
        }
        return llm_out, cards

    try:
        out = llm_postprocess_cards_and_summaries(cards, fallback_paragraph, anthropic_model=anthropic_model)
        updated = apply_label_map(cards, out.get("label_map", {}))
        # Ensure keys exist
        llm_out = {
            "summary_small": (out.get("summary_small") or "").strip() or render_summary_short(fallback_paragraph, updated).strip(),
            "summary_big": (out.get("summary_big") or "").strip() or render_summary_expanded(fallback_paragraph, updated).strip(),
            "truths": out.get("truths") if isinstance(out.get("truths"), list) else [],
            "label_map": out.get("label_map") if isinstance(out.get("label_map"), dict) else {},
        }
        return llm_out, updated
    except Exception as e:
        llm_out = {
            "summary_small": render_summary_short(fallback_paragraph, cards).strip(),
            "summary_big": render_summary_expanded(fallback_paragraph, cards).strip(),
            "truths": [],
            "error": str(e),
        }
        return llm_out, cards


# -----------------------------
# Phase 2 runner
# -----------------------------

def run_phase2(
    G: nx.Graph,
    node_to_comm: Dict[str, int],
    comm_summaries: List[dict],
    session_trails: dict,
    cfg: AgentConfig,
) -> dict:
    candidates = build_session_candidates(G, node_to_comm)
    selected_sessions, trace = select_sessions_submodular(candidates, cfg)

    evidence, covered = extract_evidence(G, node_to_comm, selected_sessions, cfg)
    cards = build_intent_cards(G, node_to_comm, comm_summaries, session_trails, selected_sessions)

    profile_para = build_profile_paragraph(covered, comm_summaries)

    # Markdown report
    lines: List[str] = []
    lines.append("# Phase 2 Profile (Constrained Graph Agent)")
    lines.append("")
    lines.append(profile_para)
    lines.append("")
    lines.append(f"- Selected sessions: **{len(selected_sessions)}** (cap {cfg.max_sessions})")
    lines.append(f"- Evidence items: **{len(evidence)}** / {cfg.budget}")
    lines.append(f"- Communities covered: **{len(covered)}** (target ≥ {cfg.min_communities})")
    lines.append("")

    lines.append("## Intent cards (compressed evidence)")
    lines.append("")
    for card in cards[:8]:
        lines.append(f"### {card.get('label','Intent')} — session {card.get('session_id','')}")
        qs = card.get("top_queries", [])
        ds = card.get("supporting_domains", [])
        if qs:
            lines.append("- intents: " + ", ".join([f"`{x}`" for x in qs]))
        if ds:
            lines.append("- sites: " + ", ".join([f"`{x}`" for x in ds]))
        reps = card.get("representative_titles", [])
        if reps:
            lines.append("- trail: " + " | ".join([f"`{t}`" for t in reps[:3]]))
        lines.append("")

    return {
        "config": asdict(cfg),
        "covered_communities": sorted(list(covered)),
        "selected_sessions": [session_candidate_to_dict(s) for s in selected_sessions],
        "trace": [asdict(t) for t in trace],
        "intent_cards": cards,
        "evidence": [asdict(e) for e in evidence],
        "profile_paragraph": profile_para,
        "profile_md": "\n".join(lines),
    }

def _ntype_counts(G: nx.Graph) -> Counter[str]:
    ctr = Counter()
    for n, data in G.nodes(data=True):
        ctr[data.get("ntype", "unknown")] += 1
    return ctr


def _top_weighted_degree(G: nx.Graph, prefix: str, k: int = 10) -> List[Tuple[str, float]]:
    scored: List[Tuple[str, float]] = []
    for n in G.nodes:
        if not (isinstance(n, str) and n.startswith(prefix)):
            continue
        s = 0.0
        for _, _, d in G.edges(n, data=True):
            s += float(d.get("weight", 1.0))
        scored.append((_strip_prefix(n), float(s)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def _theme_mass_all_sessions(G: nx.Graph, node_to_comm: Dict[str, int]) -> Counter[int]:
    """Total community mass across ALL session->(domain/query) edges."""
    mass: Counter[int] = Counter()
    for s, sdata in G.nodes(data=True):
        if sdata.get("ntype") != "session":
            continue
        for nb in G.neighbors(s):
            nbtype = G.nodes[nb].get("ntype")
            if nbtype not in {"domain", "query"}:
                continue
            cid = node_to_comm.get(nb)
            if cid is None or int(cid) == -1:
                continue
            w = float(G[s][nb].get("weight", 1.0))
            mass[int(cid)] += w
    return mass


def _label_for_comm(comm_summaries: List[dict], cid: int) -> str:
    summ = next((x for x in comm_summaries if int(x.get("community_id", -999)) == int(cid)), None)
    if not summ:
        return f"Community {cid}"
    return _guess_topic_label(summ.get("top_domains", [])[:6], summ.get("top_queries", [])[:6], summ.get("top_entities", [])[:6])


def _top_queries_wrapped(G: nx.Graph, k: int = 10) -> List[Tuple[str, float]]:
    """Rank queries by (weighted degree * IDF) to favor distinctive intent."""
    q_df = _query_df_from_graph(G)
    n_sess = max(1, sum(1 for n in G.nodes if isinstance(n, str) and n.startswith("s:")))

    def q_idf(q: str) -> float:
        df = q_df.get(q, 0)
        return math.log((1.0 + n_sess) / (1.0 + df)) + 1.0

    scored = []
    for n in G.nodes:
        if not (isinstance(n, str) and n.startswith("q:")):
            continue
        if is_utility_query_node(G, n) or query_quality_of_node(G, n) < 0.25:
            continue
        q = _strip_prefix(n)
        deg = 0.0
        for _, _, d in G.edges(n, data=True):
            deg += float(d.get("weight", 1.0))
        scored.append((q, float(deg) * float(q_idf(q))))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def _top_utility_queries(G: nx.Graph, k: int = 10) -> List[Tuple[str, float]]:
    """Rank utility/admin queries by weighted degree (for a separate lane)."""
    scored: List[Tuple[str, float]] = []
    for n in G.nodes:
        if not (isinstance(n, str) and n.startswith("q:")):
            continue
        if not is_utility_query_node(G, n):
            continue
        deg = 0.0
        for _, _, d in G.edges(n, data=True):
            deg += float(d.get("weight", 1.0))
        scored.append((_strip_prefix(n), float(deg)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def render_wrapped_md(
    G: nx.Graph,
    node_to_comm: Dict[str, int],
    comm_summaries: List[dict],
    result: dict,
    metrics: dict,
) -> str:
    """Spotify-Wrapped-style markdown summary."""
    ntypes = _ntype_counts(G)
    sessions_total = int(ntypes.get("session", 0))
    domains_total = int(ntypes.get("domain", 0))
    queries_total = int(ntypes.get("query", 0))
    entities_total = int(ntypes.get("entity", 0))

    # Themes across ALL sessions
    mass = _theme_mass_all_sessions(G, node_to_comm)
    total_mass = sum(mass.values()) or 1.0
    top_themes = mass.most_common(7)

    # Top domains/queries/entities overall
    top_domains = _top_weighted_degree(G, "d:", k=12)
    top_queries = _top_queries_wrapped(G, k=12)
    top_utility = _top_utility_queries(G, k=10)
    top_entities = _top_weighted_degree(G, "e:", k=12)

    # Rabbit holes from intent cards (selected evidence)
    cards = result.get("intent_cards", []) or []
    cards_sorted = sorted(cards, key=lambda x: float(x.get("strength", 0.0)), reverse=True)
    rabbit_holes = cards_sorted[:5]

    llm_block = result.get("llm") if isinstance(result.get("llm"), dict) else {}
    truths = llm_block.get("truths") if isinstance(llm_block.get("truths"), list) else []

    lines: List[str] = []
    lines.append("# 🌈 Web Wrapped")
    lines.append("")
    lines.append(f"_Generated: {metrics.get('timestamp','')}_")
    lines.append("")
    lines.append("## Your year in signals")
    lines.append("")
    lines.append(f"- Total sessions: **{sessions_total:,}**")
    lines.append(f"- Unique domains: **{domains_total:,}**")
    lines.append(f"- Unique search queries: **{queries_total:,}**")
    lines.append(f"- Unique entities: **{entities_total:,}**")
    lines.append(f"- Selected sessions (agent): **{len(result.get('selected_sessions', []))}**")
    lines.append(f"- Evidence budget used: **{len(result.get('evidence', []))} / {result.get('config', {}).get('budget', 40)}**")
    lines.append("")

    lines.append("## Top themes (what keeps coming back)")
    lines.append("")
    for cid, m in top_themes:
        pct = 100.0 * float(m) / float(total_mass)
        label = _label_for_comm(comm_summaries, int(cid))
        lines.append(f"- **{label}** — {pct:.1f}% of signal (community `{cid}`)")
    lines.append("")

    lines.append("## Top sites (weighted)")
    lines.append("")
    for d, s in top_domains[:10]:
        lines.append(f"- `{d}` (score {s:.1f})")
    lines.append("")

    lines.append("## Top entities (semantic)")
    lines.append("")
    for e, s in top_entities[:10]:
        lines.append(f"- `{e}` (score {s:.1f})")
    lines.append("")

    lines.append("## Top searches (distinctive intent)")
    lines.append("")
    for q, s in top_queries[:10]:
        lines.append(f"- `{q}` (score {s:.1f})")
    lines.append("")

    lines.append("## Top utility/admin searches (separate lane)")
    lines.append("")
    if top_utility:
        for q, s in top_utility[:10]:
            lines.append(f"- `{q}` (score {s:.1f})")
    else:
        lines.append("- (none detected)")
    lines.append("")

    lines.append("## Rabbit holes (deep dives)")
    lines.append("")
    for c in rabbit_holes:
        label = c.get("label", "Rabbit hole")
        sid = c.get("session_id", "")
        qs = (c.get("top_queries") or [])[:4]
        ds = (c.get("supporting_domains") or [])[:3]
        lines.append(f"### {label} — session `{sid}`")
        if qs:
            lines.append("- intents: " + ", ".join([f"`{x}`" for x in qs]))
        if ds:
            lines.append("- sites: " + ", ".join([f"`{x}`" for x in ds]))
        lines.append("")

    lines.append("## Stable truths")
    lines.append("")
    if truths:
        for t in truths:
            lines.append(f"- {t}")
    else:
        # deterministic fallback: top labels from cards
        seen = []
        for c in cards_sorted[:8]:
            lbl = (c.get("label") or "").strip()
            if lbl and lbl not in seen and lbl.lower() not in {"mixed interest", "mixed"}:
                seen.append(lbl)
        for t in seen[:6]:
            lines.append(f"- {t}")
    lines.append("")

    lines.append("---")
    lines.append("### How this was produced (one-liner)")
    lines.append(
        "We build a session–domain–query graph, detect topical communities, then a constrained agent selects a small evidence set; an optional LLM only *labels/summarizes* the selected evidence."
    )

    return "\n".join(lines).strip() + "\n"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to search_history.json")
    ap.add_argument("--artifacts", default="artifacts_phase1", help="Phase1 artifacts dir")
    ap.add_argument("--out", default="artifacts_phase2", help="Output dir for Phase2 artifacts")

    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--min-communities", type=int, default=6)
    ap.add_argument("--max-sessions", type=int, default=18)
    ap.add_argument("--per-domain-cap", type=int, default=5)
    ap.add_argument("--items-per-session", type=int, default=8)

    ap.add_argument("--tau", type=float, default=12.0)
    ap.add_argument("--diversity-lambda", type=float, default=0.35)
    ap.add_argument("--domain-cap-gamma", type=float, default=0.20)

    ap.add_argument("--tune", action="store_true", help="Bandit-style tuning over parameter configs")
    ap.add_argument("--trials", type=int, default=30, help="Number of tuning trials (default 30)")
    ap.add_argument("--seed", type=int, default=7, help="Random seed for tuning/stability")
    ap.add_argument("--llm", type=str, default="off", help="Optional LLM post-processing: off|anthropic|openai")
    ap.add_argument("--anthropic-model", type=str, default="", help="Anthropic model id/alias (optional). If empty, auto-pick from /v1/models")

    args = ap.parse_args()

    artifacts = Path(args.artifacts)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    node_to_comm = load_json(artifacts / "node_to_comm.json")
    comm_summaries = load_json(artifacts / "community_summaries.json")
    session_trails = load_json(artifacts / "session_trails.json")

    events = load_events(args.json)
    events, _ = assign_sessions(events, gap_minutes=30)

    if args.tune:
        cfg, trials = tune_parameters(
            events, node_to_comm, comm_summaries, session_trails, trials=args.trials, seed=args.seed
        )
        save_json(outdir / "tuning_results.json", [asdict(t) for t in trials])
    else:
        cfg = AgentConfig(
            budget=args.budget,
            min_communities=args.min_communities,
            max_sessions=args.max_sessions,
            max_items_per_session=args.items_per_session,
            per_domain_cap=args.per_domain_cap,
            tau=args.tau,
            diversity_lambda=args.diversity_lambda,
            domain_cap_gamma=args.domain_cap_gamma,
        )

    G = build_history_graph(events)
    result = run_phase2(G, node_to_comm, comm_summaries, session_trails, cfg)

    llm_out, cards = maybe_llm_summarize(
        result.get("intent_cards", []),
        result.get("profile_paragraph", ""),
        provider=args.llm,
        anthropic_model=(args.anthropic_model.strip() or None),
    )
    result["intent_cards"] = cards
    result["llm"] = llm_out

    reward, metrics = score_result(cfg, result)
    metrics["timestamp"] = datetime.utcnow().isoformat() + "Z"
    metrics["llm"] = args.llm
    metrics["config"] = asdict(cfg)
    metrics["reward_final"] = float(reward)

    save_json(outdir / "selected_sessions.json", result["selected_sessions"])
    save_json(outdir / "evidence.json", {"config": result["config"], "evidence": result["evidence"]})
    save_json(outdir / "intent_cards.json", result.get("intent_cards", []))
    save_json(outdir / "trace.json", result.get("trace", []))
    save_json(outdir / "metrics.json", metrics)

    (outdir / "PROFILE.md").write_text(result["profile_md"], encoding="utf-8")

    # Spotify-Wrapped-style output
    wrapped_md = render_wrapped_md(G, node_to_comm, comm_summaries, result, metrics)
    (outdir / "WRAPPED.md").write_text(wrapped_md, encoding="utf-8")

    # Two summary versions (always)
    llm_block = result.get("llm") if isinstance(result.get("llm"), dict) else {}
    small = (llm_block.get("summary_small") or "").strip() or render_summary_short(result.get("profile_paragraph", ""), result.get("intent_cards", [])).strip()
    big = (llm_block.get("summary_big") or "").strip() or render_summary_expanded(result.get("profile_paragraph", ""), result.get("intent_cards", [])).strip()
    truths = llm_block.get("truths") if isinstance(llm_block.get("truths"), list) else []

    short_md = "# Profile (Short)\n\n" + small.strip() + "\n"
    expanded_md = "# Profile (Expanded)\n\n" + big.strip() + "\n"
    if truths:
        expanded_md += "\n## Stable truths\n" + "\n".join([f"- {t}" for t in truths]) + "\n"

    (outdir / "SUMMARY_SHORT.md").write_text(short_md, encoding="utf-8")
    (outdir / "SUMMARY_EXPANDED.md").write_text(expanded_md, encoding="utf-8")
    save_json(outdir / "llm_output.json", llm_block)

    if args.llm not in {"off", "none", ""}:
        print(short_md)
    else:
        print(result["profile_md"])


if __name__ == "__main__":
    main()