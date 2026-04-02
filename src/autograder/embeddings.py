from __future__ import annotations

import hashlib
import math
from typing import Iterable, List, Sequence, Tuple


EMBEDDING_DIMS = 128


def embed_text_local(text: str, dims: int = EMBEDDING_DIMS) -> List[float]:
    """Deterministic local embedding fallback using hashed token bins."""
    vec = [0.0] * dims
    tokens = [t for t in text.lower().split() if t]
    if not tokens:
        return vec
    for tok in tokens:
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dims
        sign = -1.0 if (h >> 8) & 1 else 1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def retrieve_top_k(
    student_text: str,
    candidates: Iterable[dict],
    top_k: int = 3,
) -> List[Tuple[dict, float]]:
    student_emb = embed_text_local(student_text)
    scored: List[Tuple[dict, float]] = []
    for c in candidates:
        emb = c.get("embedding") or []
        sim = cosine_similarity(student_emb, emb)
        scored.append((c, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
