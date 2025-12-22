from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

_STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "at", "near", "me",
    "is", "are", "was", "were", "be", "with", "from", "by",
}

def tokens(text: str, *, extra_stop: Optional[set[str]] = None, use_bigrams: bool = True) -> List[str]:
    s = (text or "").lower()
    xs = re.findall(r"[a-z0-9]+", s)
    stop = set(_STOP)
    if extra_stop:
        stop |= set(extra_stop)

    toks = [t for t in xs if t and t not in stop and len(t) >= 2]

    if not use_bigrams:
        return toks

    bigrams: List[str] = []
    for i in range(len(toks) - 1):
        a, b = toks[i], toks[i + 1]
        if a in stop or b in stop:
            continue
        bigrams.append(f"{a}_{b}")
    return toks + bigrams

def build_tfidf(items: Dict[str, str]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], set[str]]:
    N = max(1, len(items))
    extra_stop: set[str] = set()

    # global_counts: Counter[str] = Counter()
    # for _item_id, text in items.items():
    #     toks = tokens(text, extra_stop=None, use_bigrams=False)
    #     global_counts.update(toks)

    # extra_stop: set[str] = set()
    # for tok, _c in global_counts.most_common(180):
    #     if tok.isdigit():
    #         continue
    #     extra_stop.add(tok)

    tfs: Dict[str, Counter[str]] = {}
    df: Counter[str] = Counter()
    for item_id, text in items.items():
        toks = tokens(text, extra_stop=None, use_bigrams=True)
        c = Counter(toks)
        tfs[item_id] = c
        for tok in set(c.keys()):
            df[tok] += 1

    idf: Dict[str, float] = {}
    for tok, dfi in df.items():
        idf[tok] = math.log((N + 1.0) / (dfi + 1.0)) + 1.0

    vecs: Dict[str, Dict[str, float]] = {}
    for item_id, c in tfs.items():
        out: Dict[str, float] = {}
        denom = float(sum(c.values())) or 1.0
        for tok, tf in c.items():
            tf_n = float(tf) / denom
            out[tok] = tf_n * idf.get(tok, 1.0)
        vecs[item_id] = out

    return vecs, idf, extra_stop

def norm(v: Dict[str, float]) -> float:
    return math.sqrt(sum(x * x for x in v.values()))

def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    na = norm(a)
    nb = norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb is not None:
            dot += va * vb
    return float(dot / (na * nb))

def vec_add(acc: Dict[str, float], v: Dict[str, float], w: float = 1.0) -> None:
    for k, x in v.items():
        acc[k] = acc.get(k, 0.0) + float(w) * float(x)

def vec_scale(v: Dict[str, float], s: float) -> Dict[str, float]:
    return {k: float(x) * float(s) for k, x in v.items()}

def top_tokens(v: Dict[str, float], k: int = 4) -> List[str]:
    return [t for t, _ in sorted(v.items(), key=lambda kv: kv[1], reverse=True)[:k]]

def signature_token_set(centroid: Dict[str, float], k: int = 24) -> set[str]:
    return set(top_tokens(centroid, k=k))

def item_overlap_score(item_text: str, sig: set[str]) -> int:
    toks = set(tokens(item_text, extra_stop=None, use_bigrams=True))
    return int(len(toks & sig))