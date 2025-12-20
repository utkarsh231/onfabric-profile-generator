"""communities.py

What it does:
- Produces topical communities by clustering a projection of the heterogeneous graph.
- We project onto domain/query nodes only (dropping session bridges) for cleaner topics.

Main entrypoints:
- detect_topic_communities(G, min_size=8) -> node_to_comm
- summarize_topic_communities(G, node_to_comm, top_k=8) -> summaries

Notes:
- Uses greedy modularity (NetworkX) for deterministic-ish clustering without extra deps.
"""

from __future__ import annotations

from typing import Dict, List

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

from src.graph.build_graph import MIN_QUERY_QUALITY


def _node_type(G: nx.Graph, n: str) -> str:
    t = G.nodes[n].get("ntype")
    if t:
        return t
    if isinstance(n, str) and n.startswith("s:"):
        return "session"
    if isinstance(n, str) and n.startswith("d:"):
        return "domain"
    if isinstance(n, str) and n.startswith("q:"):
        return "query"
    return "unknown"


def build_domain_query_projection(G: nx.Graph) -> nx.Graph:
    """Project the heterogeneous graph onto domain/query nodes only.

    Why: session nodes act as high-degree bridges that can glue unrelated topics
    into mega-communities. We keep sessions for trails/explanations, but we cluster
    on the domain–query subgraph for cleaner topical communities.

    This function keeps only:
      - nodes where ntype is domain or query
      - edges where etype == 'domain-query' (if present) and always uses weights
    """
    H = nx.Graph()

    # Keep only domain/query nodes
    for n in G.nodes:
        t = _node_type(G, n)
        if t == "domain":
            H.add_node(n, **G.nodes[n])
            continue
        if t == "query":
            # Route utility + very low-quality queries away from topical communities
            qclass = G.nodes[n].get("qclass")
            qqual = float(G.nodes[n].get("qquality", 1.0))
            if qclass == "utility" or qqual < MIN_QUERY_QUALITY:
                continue
            H.add_node(n, **G.nodes[n])
            continue

    # Keep only domain-query edges (ignore any edges touching session nodes)
    for u, v, data in G.edges(data=True):
        if _node_type(G, u) == "session" or _node_type(G, v) == "session":
            continue

        et = data.get("etype")
        if et is not None and et != "domain-query":
            continue

        # IMPORTANT: only keep edges whose endpoints survived node filtering above.
        # Otherwise NetworkX will auto-create missing nodes and defeat our filters.
        if not H.has_node(u) or not H.has_node(v):
            continue

        w = float(data.get("weight", 1.0))
        if H.has_edge(u, v):
            H[u][v]["weight"] += w
        else:
            H.add_edge(u, v, weight=w)

    return H


def detect_topic_communities(G: nx.Graph, *, min_size: int = 8) -> Dict[str, int]:
    """Detect communities on the domain–query projection for cleaner topics."""
    H = build_domain_query_projection(G)
    return detect_communities(H, min_size=min_size)


def summarize_topic_communities(G: nx.Graph, node_to_comm: Dict[str, int], top_k: int = 8) -> List[dict]:
    """Summarize communities using the same projection graph used for clustering."""
    H = build_domain_query_projection(G)
    return summarize_communities(H, node_to_comm, top_k=top_k)


def detect_communities(G: nx.Graph, *, min_size: int = 8) -> Dict[str, int]:
    """
    Deterministic-ish community detection using greedy modularity (no extra deps).
    Returns node -> community_id for communities >= min_size; others get -1.
    """
    # Work on largest connected component for stability
    if G.number_of_nodes() == 0:
        return {}

    # greedy_modularity_communities supports 'weight'
    comms = list(greedy_modularity_communities(G, weight="weight"))
    comms_sorted = sorted(comms, key=lambda c: len(c), reverse=True)

    node_to_comm: Dict[str, int] = {}
    cid = 0
    for c in comms_sorted:
        if len(c) < min_size:
            continue
        for n in c:
            node_to_comm[n] = cid
        cid += 1

    # mark leftovers
    for n in G.nodes():
        if n not in node_to_comm:
            node_to_comm[n] = -1

    return node_to_comm


def summarize_communities(G: nx.Graph, node_to_comm: Dict[str, int], top_k: int = 8) -> List[dict]:
    """
    For each community, show top domains/queries by weighted degree within the community.
    """
    # group nodes
    comm_to_nodes: Dict[int, List[str]] = {}
    for n, cid in node_to_comm.items():
        comm_to_nodes.setdefault(cid, []).append(n)

    summaries: List[dict] = []
    for cid, nodes in sorted(comm_to_nodes.items(), key=lambda x: len(x[1]), reverse=True):
        if cid == -1:
            continue
        sub = G.subgraph(nodes).copy()

        # weighted degree
        deg = dict(sub.degree(weight="weight"))

        domains = [n for n in nodes if n.startswith("d:")]
        queries = [
            n
            for n in nodes
            if n.startswith("q:")
            and sub.nodes[n].get("qclass") != "utility"
            and float(sub.nodes[n].get("qquality", 1.0)) >= MIN_QUERY_QUALITY
        ]

        domains_sorted = sorted(domains, key=lambda n: deg.get(n, 0.0), reverse=True)[:top_k]
        queries_sorted = sorted(queries, key=lambda n: deg.get(n, 0.0), reverse=True)[:top_k]

        summaries.append(
            {
                "community_id": cid,
                "size": len(nodes),
                "top_domains": [d[2:] for d in domains_sorted],
                "top_queries": [q[2:] for q in queries_sorted],
            }
        )

    return summaries