from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# Import your existing pipeline
from src.ingest.parse_takeout import load_events
from src.ingest.sessionize import assign_sessions
from src.graph.build_graph import build_history_graph, basic_graph_stats


# -----------------------------
# Helpers
# -----------------------------

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def strip_prefix(n: str) -> str:
    return n.split(":", 1)[1] if ":" in n else n

def node_type(G: nx.Graph, n: str) -> str:
    return G.nodes[n].get("ntype", "unknown")

def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def sort_edges_by_weight(G: nx.Graph) -> List[Tuple[str, str, float]]:
    out = []
    for u, v, d in G.edges(data=True):
        out.append((u, v, safe_float(d.get("weight", 1.0), 1.0)))
    out.sort(key=lambda t: t[2], reverse=True)
    return out

def filter_to_top_edges(G: nx.Graph, max_edges: int = 2500) -> nx.Graph:
    """Return a subgraph containing only the strongest edges (keeps all incident nodes)."""
    if G.number_of_edges() <= max_edges:
        return G
    edges = sort_edges_by_weight(G)[:max_edges]
    H = nx.Graph()
    for u, v, w in edges:
        H.add_node(u, **G.nodes[u])
        H.add_node(v, **G.nodes[v])
        attrs = dict(G.get_edge_data(u, v) or {})
        # Avoid passing weight twice (some graphs already store it in attrs)
        attrs["weight"] = float(w)
        H.add_edge(u, v, **attrs)
    # drop isolates
    isolates = [n for n in H.nodes if H.degree(n) == 0]
    H.remove_nodes_from(isolates)
    return H

def pyvis_html(
    G: nx.Graph,
    height_px: int = 820,
    max_nodes: int = 500,
) -> str:
    """Render an interactive graph with PyVis and return HTML."""
    # If too many nodes, keep highest weighted-degree nodes.
    H = G
    if H.number_of_nodes() > max_nodes:
        deg = dict(H.degree(weight="weight"))
        keep = sorted(H.nodes, key=lambda n: deg.get(n, 0.0), reverse=True)[:max_nodes]
        H = H.subgraph(keep).copy()

    net = Network(height=f"{height_px}px", width="100%", bgcolor="#0b0b0b", font_color="#f3f3f3")
    net.force_atlas_2based(gravity=-30, central_gravity=0.01, spring_length=120, spring_strength=0.02)

    for n, data in H.nodes(data=True):
        t = data.get("ntype", "unknown")
        label = strip_prefix(n)

        # Keep labels short
        if len(label) > 42:
            label = label[:41] + "…"

        title_lines = [
            f"<b>{n}</b>",
            f"ntype: {t}",
        ]
        if t == "query":
            title_lines.append(f"qclass: {data.get('qclass', '')}")
            title_lines.append(f"qquality: {safe_float(data.get('qquality', 1.0), 1.0):.2f}")

        # Shape helps readability without relying on color
        shape = "dot"
        if t == "query":
            shape = "box"
        elif t == "session":
            shape = "triangle"
        elif t == "domain":
            shape = "dot"

        net.add_node(
            n,
            label=label,
            title="<br/>".join(title_lines),
            shape=shape,
        )

    for u, v, d in H.edges(data=True):
        w = safe_float(d.get("weight", 1.0), 1.0)
        et = d.get("etype", "")
        net.add_edge(u, v, value=w, title=f"weight={w:.2f}<br/>etype={et}")

    # better UX
    net.set_options(
        """
        var options = {
          "physics": {
            "enabled": true,
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 120,
              "springConstant": 0.02
            },
            "solver": "forceAtlas2Based",
            "stabilization": { "enabled": true, "iterations": 600 }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 120,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """
    )

    return net.generate_html()


@st.cache_resource(show_spinner=False)
def build_graph_cached(json_path: str, gap_minutes: int) -> Tuple[nx.Graph, dict]:
    events = load_events(json_path)
    events, _ = assign_sessions(events, gap_minutes=gap_minutes)
    G = build_history_graph(events)
    stats = basic_graph_stats(G)
    return G, stats


@st.cache_data(show_spinner=False)
def load_phase1_artifacts(artifacts_dir: str) -> Tuple[List[dict], Dict[str, int], dict]:
    ad = Path(artifacts_dir)
    comm_summaries = load_json(ad / "community_summaries.json")
    node_to_comm = load_json(ad / "node_to_comm.json")
    session_trails = load_json(ad / "session_trails.json")
    # node_to_comm sometimes stores as str->int; ensure ints
    node_to_comm2 = {k: int(v) for k, v in node_to_comm.items()}
    return comm_summaries, node_to_comm2, session_trails


def community_subgraph(
    G: nx.Graph,
    node_to_comm: Dict[str, int],
    community_id: int,
    include_sessions: bool = False,
    max_session_nodes: int = 60,
) -> nx.Graph:
    nodes = [n for n, c in node_to_comm.items() if int(c) == int(community_id) and n in G]
    H = G.subgraph(nodes).copy()

    # Optionally pull in session neighbors (for “why this exists”)
    if include_sessions:
        sess = set()
        for n in list(H.nodes):
            if node_type(G, n) in {"domain", "query"}:
                for nb in G.neighbors(n):
                    if isinstance(nb, str) and nb.startswith("s:"):
                        sess.add(nb)

        # Keep only the most connected sessions into this community
        if sess:
            # score by total edge weight into community nodes
            scores = []
            comm_nodes = set(H.nodes)
            for s in sess:
                tot = 0.0
                for nb in G.neighbors(s):
                    if nb in comm_nodes:
                        tot += safe_float(G[s][nb].get("weight", 1.0), 1.0)
                scores.append((s, tot))
            scores.sort(key=lambda x: x[1], reverse=True)
            keep_sessions = [s for s, _ in scores[:max_session_nodes]]

            # union + induced subgraph
            nodes2 = set(H.nodes) | set(keep_sessions)
            H = G.subgraph(nodes2).copy()

    # Reduce clutter: keep only strongest edges
    H = filter_to_top_edges(H, max_edges=2500)
    return H


def ego_subgraph(G: nx.Graph, center_node: str, radius: int = 2) -> nx.Graph:
    if center_node not in G:
        return nx.Graph()
    H = nx.ego_graph(G, center_node, radius=radius, center=True, undirected=True)
    H = filter_to_top_edges(H, max_edges=2500)
    return H


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Graph Wrapped Explorer", layout="wide")

st.title("🕸️ Graph Wrapped Explorer")
st.caption("Explore your session–domain–query graph, topics, and trails interactively.")

with st.sidebar:
    st.header("Inputs")
    json_path = st.text_input("search_history.json path", value="search_history.json")
    artifacts_dir = st.text_input("Phase 1 artifacts dir", value="artifacts_phase1")
    gap_minutes = st.number_input("Session gap (minutes)", min_value=5, max_value=240, value=30, step=5)

    st.divider()
    st.header("Graph display")
    max_nodes = st.slider("Max nodes to render", 150, 900, 500, step=50)
    include_sessions = st.checkbox("Include session nodes in community view", value=False)
    radius = st.slider("Ego graph radius", 1, 4, 2)

    st.divider()
    build_btn = st.button("🔄 Load / Rebuild", type="primary")

# Always try loading (cache makes it fast)
if build_btn:
    st.cache_resource.clear()
    st.cache_data.clear()

# Validate paths early
if not Path(json_path).exists():
    st.error(f"Cannot find JSON file: {json_path}")
    st.stop()
if not Path(artifacts_dir).exists():
    st.error(f"Cannot find artifacts dir: {artifacts_dir}")
    st.stop()

with st.spinner("Loading artifacts + building graph…"):
    comm_summaries, node_to_comm, session_trails = load_phase1_artifacts(artifacts_dir)
    G, stats = build_graph_cached(json_path, gap_minutes)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Communities", "Sessions", "Explorer"])

# -----------------------------
# Overview
# -----------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", f"{stats['nodes']:,}")
    c2.metric("Edges", f"{stats['edges']:,}")
    c3.metric("Sessions", f"{stats['sessions']:,}")
    c4.metric("Queries", f"{stats['queries']:,}")

    # Quick noise diagnostics
    q_nodes = [n for n in G.nodes if isinstance(n, str) and n.startswith("q:")]
    q_df = pd.DataFrame({
        "q": [strip_prefix(n) for n in q_nodes],
        "qclass": [G.nodes[n].get("qclass", "") for n in q_nodes],
        "qquality": [safe_float(G.nodes[n].get("qquality", 1.0), 1.0) for n in q_nodes],
        "degree_w": [safe_float(G.degree(n, weight="weight"), 0.0) for n in q_nodes],
    })
    colA, colB = st.columns([1, 1])
    with colA:
        st.subheader("Query quality distribution")
        st.bar_chart(q_df["qquality"].round(2).value_counts().sort_index())
    with colB:
        st.subheader("Query class counts")
        st.bar_chart(q_df["qclass"].fillna("").replace("", "(missing)").value_counts())

    st.subheader("Top weighted queries (debug)")
    topq = q_df.sort_values("degree_w", ascending=False).head(25)
    st.dataframe(topq[["q", "qclass", "qquality", "degree_w"]], use_container_width=True)

# -----------------------------
# Communities
# -----------------------------
with tab2:
    st.subheader("Topic communities (domain–query projection)")

    comm_df = pd.DataFrame(comm_summaries)
    if comm_df.empty:
        st.warning("No community summaries found.")
        st.stop()

    # choose community
    comm_id = st.selectbox(
        "Select community_id",
        options=comm_df["community_id"].tolist(),
        index=0,
    )

    row = comm_df[comm_df["community_id"] == comm_id].iloc[0]
    st.write(f"**Size:** {int(row['size'])}")

    cL, cR = st.columns([1, 1])
    with cL:
        st.markdown("**Top domains**")
        st.write(row["top_domains"])
    with cR:
        st.markdown("**Top queries**")
        st.write(row["top_queries"])

    st.divider()

    st.subheader("Community subgraph (interactive)")
    H = community_subgraph(G, node_to_comm, int(comm_id), include_sessions=include_sessions)

    if H.number_of_nodes() == 0:
        st.info("No nodes in this community subgraph.")
    else:
        html = pyvis_html(H, height_px=820, max_nodes=max_nodes)
        components.html(html, height=850, scrolling=True)

    st.subheader("Top nodes in this community (weighted degree)")
    deg = dict(H.degree(weight="weight"))
    df_nodes = pd.DataFrame({
        "node": list(H.nodes),
        "ntype": [node_type(H, n) for n in H.nodes],
        "label": [strip_prefix(n) for n in H.nodes],
        "qclass": [H.nodes[n].get("qclass", "") for n in H.nodes],
        "qquality": [safe_float(H.nodes[n].get("qquality", 1.0), 1.0) for n in H.nodes],
        "deg_w": [safe_float(deg.get(n, 0.0), 0.0) for n in H.nodes],
    }).sort_values("deg_w", ascending=False)

    st.dataframe(df_nodes.head(60), use_container_width=True)

# -----------------------------
# Sessions
# -----------------------------
with tab3:
    st.subheader("Session trails")

    sess_ids = sorted(session_trails.keys())
    sess_pick = st.selectbox("Pick a session_id", options=sess_ids, index=0)

    trail = session_trails.get(sess_pick, {})
    if not isinstance(trail, dict):
        st.warning("Malformed trail.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("n_events", int(trail.get("n_events", 0)))
        c2.metric("start", str(trail.get("start_time", ""))[:19])
        c3.metric("end", str(trail.get("end_time", ""))[:19])
        c4.metric("domains (top)", str(len(trail.get("top_domains", []) or [])))

        st.markdown("**Top domains**")
        st.write(trail.get("top_domains", []))

        st.markdown("**Top interest queries**")
        st.write(trail.get("top_queries", []))

        st.markdown("**Top utility/admin queries**")
        st.write(trail.get("top_queries_utility", []))

        st.markdown("**Representative titles**")
        for t in trail.get("representative_titles", [])[:10]:
            st.write(f"- {t}")

    st.divider()
    st.subheader("Ego graph around this session (radius=1)")
    s_node = f"s:{sess_pick}"
    if s_node in G:
        Hs = ego_subgraph(G, s_node, radius=1)
        html = pyvis_html(Hs, height_px=700, max_nodes=min(max_nodes, 350))
        components.html(html, height=740, scrolling=True)
    else:
        st.info("Session node not found in graph (did you rebuild with same sessionization gap?).")

# -----------------------------
# Explorer
# -----------------------------
with tab4:
    st.subheader("Node explorer (ego graph)")

    # Provide a searchable list of nodes (can be heavy; keep it reasonable)
    # Prefer domain+query nodes for UX
    dq_nodes = [n for n in G.nodes if isinstance(n, str) and (n.startswith("d:") or n.startswith("q:"))]
    # Show a text input + best-effort match
    query = st.text_input("Search for a node (type part of domain or query text)", value="")

    # Create suggestions
    suggestions = []
    qlow = query.strip().lower()
    if qlow:
        for n in dq_nodes[:200000]:  # safety cap
            lab = strip_prefix(n).lower()
            if qlow in lab:
                suggestions.append(n)
            if len(suggestions) >= 80:
                break

    pick = st.selectbox(
        "Pick a node to inspect",
        options=suggestions if suggestions else dq_nodes[:200],
        index=0,
        help="Tip: type in the search box first to narrow choices.",
    )

    st.write(f"Selected: `{pick}` | ntype={node_type(G, pick)}")
    if pick.startswith("q:"):
        st.write(
            f"qclass={G.nodes[pick].get('qclass','')} | "
            f"qquality={safe_float(G.nodes[pick].get('qquality', 1.0),1.0):.2f}"
        )

    H = ego_subgraph(G, pick, radius=radius)
    if H.number_of_nodes() == 0:
        st.info("Nothing to show.")
    else:
        html = pyvis_html(H, height_px=820, max_nodes=max_nodes)
        components.html(html, height=850, scrolling=True)

    st.subheader("Top neighbors (by edge weight)")
    neigh = []
    for nb in G.neighbors(pick):
        w = safe_float(G[pick][nb].get("weight", 1.0), 1.0)
        neigh.append((w, nb, node_type(G, nb)))
    neigh.sort(reverse=True, key=lambda x: x[0])
    dfn = pd.DataFrame(
        [{"weight": w, "node": nb, "ntype": t, "label": strip_prefix(nb)} for (w, nb, t) in neigh[:60]]
    )
    st.dataframe(dfn, use_container_width=True)