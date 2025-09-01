from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

# Ensure repo root on sys.path so 'src' imports work when run as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Optional BM25
try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except Exception:
    _HAS_BM25 = False

# Lazy import embedding function used elsewhere in repo
try:
    from src.embed import get_embedding
except Exception:
    try:
        from embed import get_embedding
    except Exception:
        def get_embedding(texts):
            raise RuntimeError("get_embedding not available; configure src.embed or embed.py")


def load_meta(meta_path: str) -> Dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_scores(arr: np.ndarray) -> np.ndarray:
    # min-max to [0,1]
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.ones_like(arr)
    return (arr - lo) / (hi - lo)


class TwoStageRetriever:
    def __init__(self, paper_index_path: str, paper_meta_path: str, chunk_index_path: str, chunk_meta_path: str,
                 top_m_papers: int = 75, mix_weights: Tuple[float, float] = (0.4, 0.6)):
        import faiss
        self.faiss = faiss
        self.paper_idx = faiss.read_index(paper_index_path)
        self.chunk_idx = faiss.read_index(chunk_index_path)
        self.paper_meta = load_meta(paper_meta_path)
        self.chunk_meta = load_meta(chunk_meta_path)
        self.top_m_papers = top_m_papers
        self.mix_weights = mix_weights

        # paper id -> meta list helper
        self.paper_id_list = [m.get("id") for m in self.paper_meta.get("meta", [])]
        self.paper_id_to_idx = {pid: i for i, pid in enumerate(self.paper_id_list) if pid}

        # chunk metas
        self.chunk_metas = self.chunk_meta.get("meta", [])

        # build BM25 on paper title+summary+key_terms if available and rank_bm25 installed
        self._bm25 = None
        if _HAS_BM25:
            docs = []
            for p in self.paper_meta.get("meta", []):
                parts = []
                if p.get("title"):
                    parts.append(p["title"])
                if p.get("summary_short"):
                    parts.append(p["summary_short"])
                if p.get("key_terms"):
                    parts.extend(p.get("key_terms"))
                docs.append(" ".join(parts).lower().split())
            if docs:
                self._bm25 = BM25Okapi(docs)

    def _embed(self, texts: List[str]) -> np.ndarray:
        mat = get_embedding(texts)
        # ensure shape (N, D)
        return np.asarray(mat, dtype="float32")

    def _search_index(self, idx, vecs: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        # returns (ids, scores)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        self.faiss.normalize_L2(vecs)
        D, I = idx.search(vecs, k)
        return I, D

    def paper_stage(self, q: str) -> List[Tuple[str, float]]:
        # embed and search paper index
        qv = self._embed([q])
        I, D = self._search_index(self.paper_idx, qv, k=self.top_m_papers)
        ids = I[0].tolist()
        scores = D[0].tolist()
        papers = []
        for idx, sc in zip(ids, scores):
            try:
                pid = self.paper_meta["meta"][idx].get("id")
            except Exception:
                pid = None
            if pid:
                papers.append((pid, float(sc)))

        # optionally blend BM25
        if self._bm25:
            toks = q.lower().split()
            bm25_scores = self._bm25.get_scores(toks)
            # only keep top_m
            blended = []
            for pid, sc in papers:
                pidx = self.paper_id_to_idx.get(pid)
                sparse = float(bm25_scores[pidx]) if pidx is not None else 0.0
                sc = 0.8 * sc + 0.2 * sparse
                blended.append((pid, sc))
            papers = blended

        return papers

    def chunk_stage(self, q: str, candidate_pids: Optional[List[str]] = None, top_k: int = 200) -> List[Dict]:
        qv = self._embed([q])
        I, D = self._search_index(self.chunk_idx, qv, k=top_k)
        ids = I[0].tolist()
        scores = D[0].tolist()
        out = []
        for ridx, sc in zip(ids, scores):
            try:
                meta = self.chunk_metas[ridx]
            except Exception:
                meta = {}
            pid = meta.get("paper_id")
            if candidate_pids and pid not in candidate_pids:
                continue
            out.append({"meta": meta, "score": float(sc), "ridx": ridx})
        return out

    def retrieve(self, q: str, k: int = 10, k_per_paper: int = 2, top_k_chunks: int = 500) -> List[Dict]:
        # Stage 1: papers
        papers = self.paper_stage(q)
        paper_ids = [p for p, _ in papers]
        paper_scores = {p: s for p, s in papers}

        # Stage 2: chunks (filter to paper_ids)
        chunks = self.chunk_stage(q, candidate_pids=paper_ids, top_k=top_k_chunks)

        if not chunks:
            return []

        # Normalize chunk scores and paper scores
        chunk_scores = np.array([c["score"] for c in chunks], dtype="float32")
        norm_chunk = normalize_scores(chunk_scores)
        # map paper scores to array aligned with chunks
        pscore_arr = np.array([paper_scores.get(c["meta"].get("paper_id"), 0.0) for c in chunks], dtype="float32")
        norm_paper = normalize_scores(pscore_arr)

        w_p, w_c = self.mix_weights
        final_scores = w_p * norm_paper + w_c * norm_chunk

        for i, c in enumerate(chunks):
            c["final_score"] = float(final_scores[i])

        # optional post-filtering: none by default here; leave to caller

        # diversify by paper: keep up to k_per_paper per paper, then top-k papers
        by_pid = {}
        for c in sorted(chunks, key=lambda x: x["final_score"], reverse=True):
            pid = c["meta"].get("paper_id")
            by_pid.setdefault(pid, []).append(c)

        per_paper = []
        for pid, rows in by_pid.items():
            per_paper.append((pid, rows[:k_per_paper], max(r["final_score"] for r in rows)))
        per_paper.sort(key=lambda x: x[2], reverse=True)

        hits = []
        for pid, rows, _ in per_paper[:k]:
            hits.extend(rows)
            if len(hits) >= k:
                break

        # normalize returned hit shape for consumers: top-level id/paper_id/title/score
        out = []
        for h in hits[:k]:
            meta = h.get("meta", {}) if isinstance(h, dict) else {}
            out_h = dict(h)
            out_h.setdefault("id", meta.get("chunk_id") or meta.get("id") or meta.get("paper_id"))
            out_h.setdefault("paper_id", meta.get("paper_id") or meta.get("id"))
            out_h.setdefault("title", meta.get("title") or meta.get("paper_title"))
            # prefer final_score then score
            out_h.setdefault("score", float(out_h.get("final_score", out_h.get("score", 0.0))))
            out.append(out_h)
        return out

    def rerank_by_tfidf(self, query: str, hits: list, top_n: int = 200, keep_fraction: float = 0.5):
        """
        Lightweight TF-IDF reranker: compute TF-IDF over hit titles + snippets (if available)
        and re-score hits by cosine similarity to the query TF-IDF vector. Returns a filtered
        and re-scored subset of hits. This is intentionally simple and dependency-free.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np
        except Exception:
            # sklearn not available; return original hits
            return hits

        texts = []
        ids = []
        for h in hits[:top_n]:
            meta = h.get("meta") or {}
            title = meta.get("title") or meta.get("paper_title") or h.get("title") or h.get("paper_title") or ""
            snippet = meta.get("snippet") or h.get("snippet") or ""
            texts.append((title + " \n " + snippet).strip())
            ids.append(h)

        if not texts:
            return hits

        vec = TfidfVectorizer(stop_words="english", max_features=2048)
        X = vec.fit_transform(texts)
        qv = vec.transform([query])
        # cosine similarity via dot product on normalized vectors
        import numpy.linalg as la
        Xn = X.astype(float)
        # compute cosine similarity
        sims = (Xn @ qv.T).toarray().ravel()
        # attach tfidf_score and sort
        scored = []
        for h, s in zip(ids, sims.tolist()):
            h["tfidf_score"] = float(s)
            scored.append(h)

        scored.sort(key=lambda x: x.get("tfidf_score", 0.0), reverse=True)
        keep = max(1, int(len(scored) * float(keep_fraction)))
        return scored[:keep]

    def multi_query_union(self, queries: List[str], k: int = 10, k_per_paper: int = 2,
                          top_m_per_query: Optional[int] = None, top_k_chunks: int = 1000,
                          paper_weight: Optional[float] = None, chunk_weight: Optional[float] = None) -> List[Dict]:
        """Run multiple queries through the paper-stage, union top papers, then run chunk-stage once.

        - queries: list of query strings (e.g., GK expansions + original)
        - top_m_per_query: how many top papers to take per query (defaults to self.top_m_papers)
        - returns: diversified chunk-level hits (same shape as retrieve)
        """
        if not queries:
            return []

        top_m = int(top_m_per_query) if top_m_per_query is not None else int(self.top_m_papers)

        # collect paper scores across queries (use max aggregation)
        paper_score_map: Dict[str, float] = {}
        for q in queries:
            papers = self.paper_stage(q)
            for pid, sc in papers[:top_m]:
                prev = paper_score_map.get(pid, None)
                if prev is None or sc > prev:
                    paper_score_map[pid] = float(sc)

        if not paper_score_map:
            return []

        candidate_pids = list(paper_score_map.keys())

        # run chunk stage once, filtering to candidate papers
        chunks = self.chunk_stage(queries[0], candidate_pids=candidate_pids, top_k=top_k_chunks)
        if not chunks:
            return []

        # Normalize chunk scores and map paper scores aligned with chunks
        import numpy as _np
        chunk_scores = _np.array([c["score"] for c in chunks], dtype="float32")
        norm_chunk = normalize_scores(chunk_scores)
        pscore_arr = _np.array([paper_score_map.get(c["meta"].get("paper_id"), 0.0) for c in chunks], dtype="float32")
        norm_paper = normalize_scores(pscore_arr)

        # mixing weights
        w_p, w_c = self.mix_weights
        if paper_weight is not None or chunk_weight is not None:
            w_p = float(paper_weight) if paper_weight is not None else w_p
            w_c = float(chunk_weight) if chunk_weight is not None else w_c

        final_scores = w_p * norm_paper + w_c * norm_chunk

        for i, c in enumerate(chunks):
            c["final_score"] = float(final_scores[i])

        # diversify and select top-k per paper similar to retrieve
        by_pid = {}
        for c in sorted(chunks, key=lambda x: x["final_score"], reverse=True):
            pid = c["meta"].get("paper_id")
            by_pid.setdefault(pid, []).append(c)

        per_paper = []
        for pid, rows in by_pid.items():
            per_paper.append((pid, rows[:k_per_paper], max(r["final_score"] for r in rows)))
        per_paper.sort(key=lambda x: x[2], reverse=True)

        hits = []
        for pid, rows, _ in per_paper[:k]:
            hits.extend(rows)
            if len(hits) >= k:
                break

        # normalize hit shape for consumers
        out = []
        for h in hits[:k]:
            meta = h.get("meta", {}) if isinstance(h, dict) else {}
            out_h = dict(h)
            out_h.setdefault("id", meta.get("chunk_id") or meta.get("id") or meta.get("paper_id"))
            out_h.setdefault("paper_id", meta.get("paper_id") or meta.get("id"))
            out_h.setdefault("title", meta.get("title") or meta.get("paper_title"))
            out_h.setdefault("score", float(out_h.get("final_score", out_h.get("score", 0.0))))
            out.append(out_h)
        return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-index", required=True)
    ap.add_argument("--paper-meta", required=True)
    ap.add_argument("--chunk-index", required=True)
    ap.add_argument("--chunk-meta", required=True)
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    tr = TwoStageRetriever(args.paper_index, args.paper_meta, args.chunk_index, args.chunk_meta)
    hits = tr.retrieve(args.q, k=args.k)
    print(json.dumps(hits, ensure_ascii=False, indent=2))
