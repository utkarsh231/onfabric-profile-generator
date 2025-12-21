import json
from typing import Dict, Iterable, List, Optional, Tuple
import os
import re

def _extract_json_obj(s: str) -> Optional[dict]:
    if not isinstance(s, str) or not s.strip():
        return None
    # try direct
    try:
        return json.loads(s)
    except Exception:
        pass
    # try to find first {...} block
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _anthropic_messages(
    api_key: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 700,
    base_url: Optional[str] = None,
) -> str:
    """Minimal Anthropic Messages API call via requests (no extra deps)."""
    import requests

    # Allow custom base URL for proxies / gateways / Bedrock-compatible routers.
    # Examples:
    #   ANTHROPIC_BASE_URL=https://api.anthropic.com
    #   ANTHROPIC_BASE_URL=https://your-proxy.example.com
    resolved_base = (base_url or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").strip()
    if not resolved_base:
        resolved_base = "https://api.anthropic.com"

    if resolved_base.endswith("/v1/messages"):
        url = resolved_base
    else:
        url = resolved_base.rstrip("/") + "/v1/messages"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": int(max_tokens),
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=45)
    if r.status_code >= 400:
        # Surface response body for debugging (Anthropic and many proxies return JSON error bodies).
        body = ""
        try:
            body = r.text
        except Exception:
            body = ""

        # Special-case 404: often indicates a wrong base URL or a proxy/firewall rewriting the request.
        if r.status_code == 404:
            raise RuntimeError(
                "Anthropic request returned 404 Not Found. "
                "This usually means you're hitting the wrong base URL, or a proxy/firewall is intercepting the request. "
                f"URL={url} | Response={body[:800]}"
            )

        raise RuntimeError(
            f"Anthropic HTTPError {r.status_code}: URL={url} | Response={body[:800]}"
        )

    data = r.json()
    # content is a list of blocks
    blocks = data.get("content") or []
    parts: List[str] = []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "\n".join(parts).strip()


def llm_refine_suit_card(
    suit_card: dict,
    *,
    query_ctx: Dict[str, dict],
    all_suit_labels: List[str],
    model: str,
    api_key: str,
    cache: Dict[str, dict],
    base_url: Optional[str] = None,
) -> dict:
    """Use an LLM as a *judge* to improve label + prune off-topic top queries.

    We keep interpretability by:
    - never deleting evidence_primary/supporting
    - only rewriting: label, paragraph, and a filtered/reordered top_queries list
    - adding `llm_judge` metadata describing what was removed and why
    """
    top_q = suit_card.get("top_queries", [])[:12]
    evid = []
    for q in top_q:
        c = query_ctx.get(q, {})
        evid.append(
            {
                "query": q,
                "domains": c.get("domains", []),
                "paths": c.get("paths", []),
                "titles": c.get("titles", []),
            }
        )

    sys = (
        "You are a precise judge that cleans topic cards for a user-profile system. "
        "Do NOT invent facts; only use provided evidence. "
        "Your job: (1) propose a clean short label (2-5 words), "
        "(2) identify off-topic queries caused by word-overlap bridges, "
        "(3) output a pruned ordered list of queries that truly match the topic."
    )

    user = {
        "current_label": suit_card.get("label"),
        "other_suits": all_suit_labels,
        "topic_evidence": evid,
        "instruction": (
            "Return JSON with keys: "
            "label, keep_queries, drop_queries, rationale, paragraph. "
            "label: 2-5 words; keep_queries/drop_queries are lists of strings from the provided queries only."
        ),
    }

    cache_key = json.dumps({"label": suit_card.get("label"), "evidence": evid}, sort_keys=True)
    if cache_key in cache:
        out = cache[cache_key]
    else:
        txt = _anthropic_messages(
            api_key,
            model,
            sys,
            json.dumps(user, ensure_ascii=False),
            max_tokens=700,
            base_url=base_url,
        )
        out = _extract_json_obj(txt) or {}
        cache[cache_key] = out

    keep = out.get("keep_queries") if isinstance(out, dict) else None
    drop = out.get("drop_queries") if isinstance(out, dict) else None

    if isinstance(keep, list):
        keep = [str(x) for x in keep if isinstance(x, (str, int, float))]
    else:
        keep = top_q

    if isinstance(drop, list):
        drop = [str(x) for x in drop if isinstance(x, (str, int, float))]
    else:
        drop = []

    # Apply: only update the presentational fields
    refined = dict(suit_card)
    if isinstance(out.get("label"), str) and out.get("label").strip():
        refined["label"] = out.get("label").strip()

    refined["top_queries"] = [q for q in keep if q in top_q]

    if isinstance(out.get("paragraph"), str) and out.get("paragraph").strip():
        refined["paragraph"] = out.get("paragraph").strip()

    refined["llm_judge"] = {
        "model": model,
        "drop_queries": [q for q in drop if q in top_q],
        "rationale": out.get("rationale", "") if isinstance(out, dict) else "",
    }
    return refined

def llm_build_profile_snapshot(*, expanded: List[dict], query_ctx: Dict[str, dict], model: str, api_key: str, base_url: Optional[str]) -> dict:
    compact = []
    for s in expanded[:10]:
        tq = s.get("top_queries", [])[:8]
        rows = []
        for q in tq:
            c = query_ctx.get(q, {})
            rows.append({"q": q, "domains": c.get("domains", [])[:3], "titles": c.get("titles", [])[:2]})
        compact.append({"label": s.get("label"), "evidence": rows})

    sys = (
        "You are generating a user profile snapshot from web history evidence. "
        "Do NOT invent facts. If something is not supported, say 'Not enough evidence'. "
        "Be concise and concrete."
    )

    user = {
        "evidence": compact,
        "instruction": (
            "Return ONLY valid JSON with the keys: "
            "location, lifestyle, fashion, travel, work, other_interests, confidence_notes. "
            "Ground claims in the evidence."
        ),
    }

    txt = _anthropic_messages(
        api_key=api_key,
        model=model,
        system=sys,
        user=json.dumps(user, ensure_ascii=False),
        max_tokens=650,
        base_url=base_url,
    )
    out = _extract_json_obj(txt)
    return out if isinstance(out, dict) else {}
