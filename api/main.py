from __future__ import annotations
import os, json, re
from typing import Dict, Any, List, Tuple, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
import faiss
from dotenv import load_dotenv

# Ensure .env overrides registry/shell for this process
load_dotenv(override=True)

# import embedding helper with a fallback so repo layout differences don't crash the server
try:
    from src.embed import get_embedding
except Exception:
    try:
        from embed import get_embedding
    except Exception:
        # get_embedding may be unavailable during startup; callers should handle failures
        def get_embedding(texts):
            raise RuntimeError("get_embedding not available; configure src.embed or embed.py")

# ---------- Config ----------
INDEX_PATH             = os.getenv("ARP_INDEX", "data/index.faiss")
META_PATH              = os.getenv("ARP_META", INDEX_PATH + ".meta.json")
CHUNKS_PATH            = os.getenv("ARP_CHUNKS", "data/chunks.jsonl")

# Defaults (tunable via env)
CHUNKS_PER_PAPER_DEF   = int(os.getenv("ARP_CHUNKS_PER_PAPER", "2"))
MAX_CTX_CHARS          = int(os.getenv("ARP_MAX_CTX_CHARS", "12000"))
GK_ENABLED             = os.getenv("ARP_GK_ENABLED", "true").lower() in {"1","true","yes","on"}
SC_N                   = int(os.getenv("ARP_SELF_CONSISTENCY_N", "3"))  # candidates to sample in voting
MIN_DISTINCT_SOURCES   = int(os.getenv("ARP_MIN_DISTINCT_SOURCES", "2")) # enforce multi-source when available (raise default to 2)

# ---------- App ----------
app = FastAPI(title="ARPack API", version="0.5.1")
app.mount("/ui", StaticFiles(directory="ui_static", html=True), name="ui")

_index = None
_meta  = None
_is_chunk_index = False
_two_stage_retriever = None

# ---------- OpenAI (lazy) ----------
_OPENAI_CLIENT = None
def _client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        from openai import OpenAI
        _OPENAI_CLIENT = OpenAI()
    return _OPENAI_CLIENT

def _llm_enabled() -> bool:
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())


def _validate_embedding_shape(vec, idx) -> None:
    """Raise HTTPException if embedding shape doesn't match index dim."""
    try:
        shape = getattr(vec, 'shape', None)
        if shape is None or shape[1] != idx.d:
            raise RuntimeError(f"Embedding dimension mismatch: got {shape}, expected {idx.d}. Rebuild indexes with the same embedding model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Load index/meta ----------
def _load():
    """Load FAISS + metadata and detect chunk/abstract mode."""
    global _index, _meta, _is_chunk_index
    if not os.path.exists(INDEX_PATH):
        raise RuntimeError(f"Missing FAISS index at {INDEX_PATH}. Build it first.")
    _index = faiss.read_index(INDEX_PATH)

    if not os.path.exists(META_PATH):
        raise RuntimeError(f"Missing metadata at {META_PATH}.")
    with open(META_PATH, "r", encoding="utf-8") as f:
        _meta = json.load(f)

    _is_chunk_index = bool(_meta.get("meta")) and ("chunk_id" in _meta["meta"][0])

@app.on_event("startup")
def startup_event():
    _load()
    # preload optional two-stage retriever if scripts are available
    try:
        tre = _get_two_stage_retriever()
        if tre is not None:
            # set sensible defaults
            tre.top_m_papers = int(os.getenv("ARP_TOP_M_PAPERS", "75"))
            tre.mix_weights = (float(os.getenv("ARP_PAPER_WEIGHT", "0.4")), float(os.getenv("ARP_CHUNK_WEIGHT", "0.6")))
            global _two_stage_retriever
            _two_stage_retriever = tre
    except Exception:
        pass

# ---------- Small helpers ----------
def _add_links(rec: Dict[str, Any]) -> None:
    pid = rec.get("paper_id") or rec.get("id")
    if pid:
        rec["url"] = f"https://arxiv.org/abs/{pid}"
        rec["pdf"] = f"https://arxiv.org/pdf/{pid}.pdf"

def _sanitize_text(t: str) -> str:
    t = t.replace("\u00ad", "")                    # soft hyphen
    t = re.sub(r"-\n(?=\w)", "", t)               # de-hyphenate across linebreaks
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _build_sources(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for h in hits:
        pid = h.get("paper_id") or h.get("id") or "unknown"
        if pid in seen: continue
        seen.add(pid)
        out.append({
            "paper_id": pid,
            "title": h.get("title", ""),
            "score": h.get("score"),
            "url": f"https://arxiv.org/abs/{pid}",
            "pdf": f"https://arxiv.org/pdf/{pid}.pdf",
        })
    return out

def _extract_citations(text: str) -> List[str]:
    # [2508.12345v1] style citations
    return sorted(set(re.findall(r"\[(\d{4}\.\d{5}(?:v\d+)?)\]", text)))


def _lexical_rerank(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simple lexical reranker used when `rerank=true` is requested on /search.
    This is intentionally lightweight: counts token matches in title/summary/text
    and boosts items with lexical overlap. Returns a re-ordered list.
    """
    qs = [t.strip().lower() for t in re.split(r"\s+", query) if t.strip()]
    if not qs or not results:
        return results

    scores = []
    max_count = 1
    for r in results:
        text = " ".join([str(r.get(k, "")) for k in ("title", "summary", "text")])
        text = text.lower()
        cnt = sum(text.count(qtok) for qtok in qs)
        scores.append(cnt)
        if cnt > max_count:
            max_count = cnt

    out = []
    for r, cnt in zip(results, scores):
        lex_norm = cnt / max_count if max_count > 0 else 0.0
        # combine original vector score (if present) with lexical score
        vec_score = float(r.get("score") or 0.0)
        final = vec_score * 0.75 + lex_norm * 0.25
        new = dict(r)
        new["score"] = final
        out.append(new)

    out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return out

# ---------- Retrieval ----------
def _retrieve_hits(q: str, k: int, k_per_paper: int, multiqueries: Optional[List[str]] = None) -> Tuple[List[Dict[str,Any]], List[str]]:
    """
    Multi-query retrieval (union) with per-paper diversification.
    Returns (hits, used_queries).
    """
    used_queries = [q]
    vecs = [q]

    if multiqueries:
        for mq in multiqueries:
            if mq and mq.strip():
                vecs.append(mq.strip())
        used_queries = vecs[:]

    # embed all queries at once
    Q = get_embedding(vecs)
    _validate_embedding_shape(Q, _index)
    faiss.normalize_L2(Q)

    # search for each query; gather raw hits
    raw: List[Dict[str, Any]] = []
    per_query_k = max(k * max(1, k_per_paper), k)
    for i in range(Q.shape[0]):
        D, I = _index.search(Q[i:i+1], per_query_k)
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx == -1: 
                continue
            rec = dict(_meta["meta"][idx])
            rec["score"] = float(score)
            rec["id"] = _meta["ids"][idx]  # chunk_id in chunk mode; paper id in abstract mode
            raw.append(rec)

    if not raw:
        return [], used_queries

    # diversify: keep up to k_per_paper chunks per paper, then take top-k papers
    by_pid: Dict[str, List[Dict[str, Any]]] = {}
    for r in raw:
        pid = r.get("paper_id") or r.get("id")
        by_pid.setdefault(pid, []).append(r)

    per_paper: List[Tuple[str, List[Dict[str, Any]], float]] = []
    for pid, rows in by_pid.items():
        rows.sort(key=lambda x: x["score"], reverse=True)
        best_score = rows[0]["score"]
        per_paper.append((pid, rows[:max(1, k_per_paper)], best_score))
    per_paper.sort(key=lambda x: x[2], reverse=True)

    hits: List[Dict[str, Any]] = []
    for pid, rows, _ in per_paper[:k]:
        hits.extend(rows)

    return hits, used_queries

def _load_texts_for_hits(hits: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str,Any]]]:
    """
    Returns (context_blocks, snippets) for the provided hits.
    IMPORTANT: Emits XML <ctx paper_id="" chunk_id="" title="">…</ctx>
    so downstream quote selection can copy real IDs.
    """
    context_blocks: List[str] = []
    snippets: List[Dict[str, Any]] = []

    if _is_chunk_index:
        if not os.path.exists(CHUNKS_PATH):
            raise HTTPException(status_code=500, detail=f"Chunk file not found at {CHUNKS_PATH}. Run hydrate_and_chunk.py first.")
        needed_ids = {h["id"] for h in hits}  # chunk_id values
        found: Dict[str, str] = {}
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not needed_ids:
                    break
                row = json.loads(line)
                cid = row.get("chunk_id")
                if cid in needed_ids:
                    found[cid] = _sanitize_text(row.get("text", ""))
                    needed_ids.remove(cid)
        for h in hits:
            text = found.get(h["id"], "")
            pid = h.get("paper_id") or h.get("id") or "unknown"
            title = h.get("title", "")
            # XML context with *real* IDs
            context_blocks.append(f"<ctx paper_id='{pid}' chunk_id='{h['id']}' title='{title}'>\n{text}\n</ctx>")
            snippets.append({"paper_id": pid, "chunk_id": h["id"], "snippet": text[:300]})
    else:
        # abstract-only fallback
        for h in hits:
            pid   = h.get("id") or "unknown"
            title = h.get("title", "")
            summ  = _sanitize_text(h.get("summary", "") or "(abstract-only index; summary unavailable)")
            context_blocks.append(f"<ctx paper_id='{pid}' chunk_id='{pid}' title='{title}'>\n{summ}\n</ctx>")
            snippets.append({"paper_id": pid, "chunk_id": pid, "snippet": summ[:300]})

    return context_blocks, snippets

# ---------- Prompting (advanced) ----------
def _gk_expand(q: str) -> Dict[str, Any]:
    """
    Generated-Knowledge expansion:
    Returns {"subquestions": [...], "terms": [...], "expansions": [...]}
    """
    if not _llm_enabled() or not GK_ENABLED:
        return {"subquestions": [], "terms": [], "expansions": []}

    sys_msg = (
        "Generate helpful retrieval expansions.\n"
        "Rules: produce 3–5 short sub-questions and 5–10 key terms (nouns/phrases), no overlap, no punctuation."
    )
    user_msg = (
        f"Question: {q}\n"
        "Return STRICT JSON with keys: subquestions (list of strings), terms (list of strings). No commentary."
    )
    try:
        resp = _client().chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[{"role":"system","content":sys_msg},{"role":"user","content":user_msg}],
        )
        raw = (resp.choices[0].message.content or "").strip().strip("`").strip()
        data = json.loads(raw)
        s = data.get("subquestions") or []
        t = data.get("terms") or []
        expansions = s + ([", ".join(t[:6])] if t else [])
        return {"subquestions": s, "terms": t, "expansions": [e for e in expansions if e]}
    except Exception:
        return {"subquestions": [], "terms": [], "expansions": []}

def _compose_quote_prompt(q: str, context_blocks: List[str]) -> Dict[str, str]:
    """
    Prompt 1: Quote selection — extract minimal verbatim quotes with provenance.
    We now provide XML <ctx paper_id='' chunk_id='' title=''>…</ctx>.
    The model must COPY those attributes into each <q …>.
    """
    system = (
        "You extract the smallest necessary VERBATIM quotes from provided excerpts.\n"
        "Return ONLY XML with <q paper_id='' chunk_id=''>quote text…</q> nodes. No commentary.\n"
        "For each quote, COPY the paper_id and chunk_id from the nearest <ctx …> block the quote text came from."
    )
    user = (
        f"Question:\n{q}\n\n"
        "From the excerpts below, extract 3–10 short quotes that directly support the answer.\n"
        "Each quote must be ≤30 words, copied verbatim from the text.\n"
        "Each <q> MUST include paper_id and chunk_id copied from its source <ctx>.\n\n"
        "EXCERPTS (XML):\n" + ("\n\n".join(context_blocks))[:MAX_CTX_CHARS]
    )
    return {"system": system, "user": user}

def _compose_answer_prompt(q: str, quotes_xml: str, top_pid: str, style: str, min_sources: int) -> Dict[str, str]:
    """
    Prompt 2: Answer synthesis — structured, grounded, multi-source when available.
    """
    system = (
        "You are a meticulous research assistant. Use ONLY the provided quotes; do not invent facts.\n"
        "Citation policy: After any claim, include [paper_id]. Prefer ≤2 citations per sentence.\n"
        "Safety: Do NOT provide operational hacking instructions; keep discussion high-level."
    )
    user = (
        f"QUESTION:\n{q}\n\n"
        f"INSTRUCTIONS:\n"
        f"- If the top result [{top_pid}] is relevant, CITE IT IN THE FIRST SENTENCE.\n"
        f"- If ≥{min_sources} distinct sources are relevant, use at least {min_sources} distinct [paper_id]s across bullets.\n"
        f"- Answer shape:\n"
        f"  1) One-sentence verdict (with a citation).\n"
        f"  2) 3–6 bullets with key evidence. Each bullet ends with [paper_id].\n"
        f"  3) 'Caveats' section (1–3 bullets) only if limitations/uncertainty are present.\n"
        f"- If you give a %/number, include a ≤12-word direct quote containing that number and a citation.\n"
        f"- Use a {'compact' if style=='concise' else 'moderately detailed'} style. No meta commentary.\n\n"
        f"QUOTES (XML):\n{quotes_xml}"
    )
    return {"system": system, "user": user}

def _select_quotes(q: str, context_blocks: List[str]) -> List[Dict[str, str]]:
    """
    Returns a list of quotes: [{"paper_id":..., "chunk_id":..., "text":...}, ...]
    """
    if not _llm_enabled():
        return []

    msgs = _compose_quote_prompt(q, context_blocks)
    try:
        resp = _client().chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[{"role":"system","content":msgs["system"]},
                      {"role":"user","content":msgs["user"]}],
        )
        xml = (resp.choices[0].message.content or "").strip()
        # Extract <q paper_id='' chunk_id=''>…</q>
        quotes = []
        for m in re.finditer(
            r"<q\s+paper_id=['\"]([^'\"]+)['\"]\s+chunk_id=['\"]([^'\"]+)['\"]\s*>(.*?)</q>",
            xml, re.DOTALL | re.IGNORECASE
        ):
            pid, cid, txt = m.group(1), m.group(2), m.group(3).strip()
            if txt:
                quotes.append({"paper_id": pid, "chunk_id": cid, "text": txt})
        return quotes[:10]
    except Exception:
        return []

def _synthesize_answer(q: str, quotes: List[Dict[str,str]], top_pid: str, style: str, n: int,
                       min_sources: int, available_sources: int) -> Tuple[str, List[str]]:
    """
    Generate n candidate answers and pick the best (self-consistency voting).
    Returns (best_answer, citations_in_answer).
    """
    if not _llm_enabled():
        return "LLM not configured (no OPENAI_API_KEY).", []

    # Build quotes XML once
    xml_parts = [f"<q paper_id='{qv['paper_id']}' chunk_id='{qv['chunk_id']}'>{qv['text']}</q>" for qv in quotes]
    quotes_xml = "<quotes>\n" + "\n".join(xml_parts) + "\n</quotes>"

    candidates: List[str] = []
    for _ in range(max(1, n)):
        msgs = _compose_answer_prompt(q, quotes_xml, top_pid, style, min_sources)
        try:
            resp = _client().chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,   # small diversity for self-consistency
                messages=[{"role":"system","content":msgs["system"]},
                          {"role":"user","content":msgs["user"]}],
            )
            ans = resp.choices[0].message.content or ""
            candidates.append(ans)
        except Exception:
            continue

    if not candidates:
        return "LLM unavailable; returning extracted quotes only.", []

    # Pick the best by (#distinct citations, then length) with a floor if multiple sources exist
    def score_answer(a: str) -> Tuple[int, int]:
        cits = _extract_citations(a)
        return (len(set(cits)), -len(a))  # prefer more sources; shorter tie-break

    best = max(candidates, key=score_answer)
    # Enforce min sources only if we actually retrieved that many distinct papers
    distinct_possible = max(1, available_sources)
    if distinct_possible >= min_sources:
        if len(set(_extract_citations(best))) < min_sources:
            # pick the next best that meets the threshold, if any
            viable = [c for c in candidates if len(set(_extract_citations(c))) >= min_sources]
            if viable:
                best = max(viable, key=score_answer)

    return best, _extract_citations(best)


# ---------- Optional two-stage retriever bridge ----------
def _get_two_stage_retriever(paper_index_path=None, paper_meta=None, chunk_index_path=None, chunk_meta=None):
    """Lazily import and construct TwoStageRetriever from scripts/two_stage_retriever.py if available.
    Returns None if not importable.
    """
    try:
        import importlib.util, sys, os
        # locate the helper script: prefer scripts/two_stage_retriever.py but also
        # accept a repo-root two_stage_retriever.py (different repo layouts)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        script_path_candidates = [
            os.path.join(repo_root, "scripts", "two_stage_retriever.py"),
            os.path.join(repo_root, "two_stage_retriever.py"),
        ]
        script_path = None
        for p in script_path_candidates:
            if os.path.exists(p):
                script_path = p
                break
        if script_path is None:
            return None
        spec = importlib.util.spec_from_file_location("two_stage_retriever", script_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        TwoStageRetriever = getattr(mod, "TwoStageRetriever")
        # Resolve defaults if caller didn't provide explicit paths (safe for startup preload)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        paper_index_path = paper_index_path or os.getenv("ARP_PAPER_INDEX", os.path.join(repo_root, "data", "index_papers.faiss"))
        paper_meta = paper_meta or os.getenv("ARP_PAPER_META", paper_index_path + ".meta.json")
        chunk_index_path = chunk_index_path or os.getenv("ARP_INDEX", os.path.join(repo_root, "data", "index.faiss"))
        chunk_meta = chunk_meta or os.getenv("ARP_META", chunk_index_path + ".meta.json")

        # If expected index files don't exist, avoid instantiating (preload is optional)
        if not (os.path.exists(paper_index_path) and os.path.exists(chunk_index_path)):
            return None

        # instantiate using the helper's expected parameter names (positional)
        return TwoStageRetriever(paper_index_path, paper_meta, chunk_index_path, chunk_meta)
    except Exception:
        return None


@app.get("/two_stage_search")
def two_stage_search(
    q: str = Query(...),
    k: int = Query(8, ge=1, le=50),
    paper_index: str | None = None,
    paper_meta: str | None = None,
    chunk_index: str | None = None,
    chunk_meta: str | None = None,
    top_m: int = Query(75, ge=1, le=500),
    paper_weight: float = Query(0.4, ge=0.0, le=1.0),
    chunk_weight: float = Query(0.6, ge=0.0, le=1.0),
    k_per_paper: int = Query(2, ge=1, le=10),
    rerank: bool = Query(False),
) -> Dict[str, Any]:
    """Run the two-stage retriever (paper-level coarse -> chunk-level fine).
    This endpoint is optional and requires the `scripts/two_stage_retriever.py` helper to be present.
    """
    # Resolve defaults from env if not provided
    paper_index = paper_index or os.getenv("ARP_PAPER_INDEX", "data/index_papers.faiss")
    paper_meta = paper_meta or os.getenv("ARP_PAPER_META", paper_index + ".meta.json")
    chunk_index = chunk_index or os.getenv("ARP_INDEX", "data/index.faiss")
    chunk_meta = chunk_meta or os.getenv("ARP_META", chunk_index + ".meta.json")

    retr = _get_two_stage_retriever(paper_index_path=paper_index, paper_meta=paper_meta, chunk_index_path=chunk_index, chunk_meta=chunk_meta)
    if retr is None:
        # return a 200 with available:false so clients can feature-detect without handling 500s
        return {"query": q, "k": k, "available": False, "results": []}

    # apply runtime tuning into the retriever instance
    try:
        retr.top_m_papers = int(top_m)
        retr.mix_weights = (float(paper_weight), float(chunk_weight))
    except Exception:
        # ignore tuning failures
        pass
    results = retr.retrieve(q, k=k, k_per_paper=k_per_paper)
    # optional lightweight TF-IDF rerank at the client request
    if rerank:
        try:
            results = retr.rerank_by_tfidf(q, results)
        except Exception:
            pass
    return {"query": q, "k": k, "results": results}

# ---------- Routes ----------
@app.get("/")
def root():
    return {"ok": True, "docs": "/docs", "endpoints": ["/healthz", "/search", "/ask", "/ask_pro"]}

@app.get("/healthz")
def healthz():
    # include two-stage retriever readiness and config when available
    tsr_info = None
    try:
        if _two_stage_retriever is not None:
            tsr_info = {
                "loaded": True,
                "top_m_papers": getattr(_two_stage_retriever, "top_m_papers", None),
                "mix_weights": getattr(_two_stage_retriever, "mix_weights", None)
            }
        else:
            tsr_info = {"loaded": False}
    except Exception:
        tsr_info = {"loaded": False}
    # expose whether LLM (OpenAI) is configured so the UI can show helpful guidance
    llm = {"enabled": _llm_enabled()}
    return {"ok": True, "index_path": INDEX_PATH, "meta_path": META_PATH, "chunk_mode": _is_chunk_index, "two_stage": tsr_info, "llm": llm}

@app.get("/search")
def search(q: str = Query(...), k: int = Query(5, ge=1, le=50), rerank: bool = Query(False)) -> Dict[str, Any]:
    vec = get_embedding([q])
    _validate_embedding_shape(vec, _index)
    faiss.normalize_L2(vec)
    D, I = _index.search(vec, k)

    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        rec = dict(_meta["meta"][idx])
        rec["score"] = float(score)
        rec["id"] = _meta["ids"][idx]
        _add_links(rec)
        results.append(rec)

    if rerank:
        try:
            results = _lexical_rerank(q, results)
        except Exception:
            pass

    return {"query": q, "k": k, "results": results}

# Your earlier structured /ask (kept; now feeds XML ctx too)
@app.get("/ask")
def ask(
    q: str = Query(...),
    k: int = Query(6, ge=1, le=50),
    k_per_paper: int | None = Query(None, ge=1, le=10),
    include_context: bool = Query(False),
    style: str = Query("concise"),
    paper_ids: str | None = Query(None),
    chunk_ids: str | None = Query(None),
) -> Dict[str, Any]:
    chunks_per = k_per_paper or CHUNKS_PER_PAPER_DEF
    # If client provided explicit paper_ids, constrain retrieval to those
    if paper_ids:
        ids = [p.strip() for p in paper_ids.split(',') if p.strip()]
        # If client also provided explicit chunk_ids (from two_stage_search), honor those
        hits = []
        if chunk_ids:
            cids = [c.strip() for c in chunk_ids.split(',') if c.strip()]
            # build a fast map from index id -> meta index
            ids_map = { _meta.get('ids', [])[i]: i for i in range(len(_meta.get('ids', []))) }
            for cid in cids:
                idx = ids_map.get(cid)
                if idx is None:
                    continue
                meta = _meta.get('meta', [])[idx]
                rec = dict(meta)
                rec['score'] = 1.0
                rec['id'] = _meta.get('ids', [])[idx]
                hits.append(rec)
        else:
            # best-effort: preserve order of provided paper ids and include up to `chunks_per` chunks per paper
            per_added = {pid: 0 for pid in ids}
            for pid in ids:
                if per_added.get(pid, 0) >= chunks_per:
                    continue
                for i, meta in enumerate(_meta.get('meta', [])):
                    m_pid = meta.get('paper_id') or meta.get('id') or _meta.get('ids', [])[i]
                    if m_pid == pid:
                        rec = dict(meta)
                        rec['score'] = 1.0
                        rec['id'] = _meta.get('ids', [])[i]
                        hits.append(rec)
                        per_added[pid] = per_added.get(pid, 0) + 1
                        if per_added[pid] >= chunks_per:
                            break
        used_queries = [q]
    else:
        hits, used_queries = _retrieve_hits(q, k, chunks_per)
    if not hits:
        return {"query": q, "k": k, "results": [], "sources": [], "snippets": [], "answer": "No results."}

    sources = _build_sources([hits[0]] + hits[1:])
    context_blocks, snippets = _load_texts_for_hits(hits)

    if not _llm_enabled():
        payload = {
            "query": q, "k": k, "results": hits, "sources": sources, "snippets": snippets,
            "answer": "LLM not configured (no OPENAI_API_KEY). Returning retrieved evidence only.",
            "used_queries": used_queries,
        }
        if include_context: payload["context_blocks"] = context_blocks
        return payload

    top_pid = sources[0]["paper_id"] if sources else "unknown"
    system = "You are a helpful research assistant. Use only the provided excerpts and cite like [paper_id]."
    user = (
        f"Answer the user's question. If [{top_pid}] supports it, cite it in the first sentence.\n\n"
        f"Question:\n{q}\n\nContext (XML ctx blocks):\n" + "\n\n".join(context_blocks)[:MAX_CTX_CHARS]
    )

    try:
        resp = _client().chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
        )
        answer = resp.choices[0].message.content or ""
    except Exception:
        answer = "LLM unavailable; showing top retrieved evidence.\n\n" + "\n\n---\n\n".join(context_blocks[:3])

    return {
        "query": q, "k": k, "results": hits, "sources": sources, "snippets": snippets,
        "answer": answer, "citations": _extract_citations(answer), "used_queries": used_queries,
        **({"context_blocks": context_blocks} if include_context else {})
    }

# NEW: Advanced RAG pipeline with GK + Quote Selection + Self-Consistency
@app.get("/ask_pro")
def ask_pro(
    q: str = Query(...),
    k: int = Query(6, ge=1, le=50),
    k_per_paper: int | None = Query(None, ge=1, le=10),
    include_context: bool = Query(False),
    style: str = Query("concise"),
    self_consistency_n: int | None = Query(None, ge=1, le=7),
    min_sources: int | None = Query(None, ge=1, le=5),
    paper_ids: str | None = Query(None),
    chunk_ids: str | None = Query(None),
) -> Dict[str, Any]:
    """
    Advanced pipeline:
      1) GK expansion -> multi-query retrieval (diversified)
      2) Quote selection over retrieved chunks
      3) Answer synthesis from quotes only
      4) Self-consistency voting across n candidates
    """
    chunks_per = k_per_paper or CHUNKS_PER_PAPER_DEF
    min_src = min_sources or MIN_DISTINCT_SOURCES

    # (1) GK expansion
    gk = _gk_expand(q)
    expansions = gk.get("expansions") or []
    multiqueries = ([q] + expansions) if expansions else [q]

    # (2) Retrieval (union) + diversification
    # Prefer the two-stage retriever when available (paper-stage coarse -> chunk-stage fine)
    used_queries = [q]
    if expansions:
        used_queries = [q] + expansions

    retr = _get_two_stage_retriever()
    if paper_ids:
        # If explicit paper_ids passed we will prefer to build hits from them.
        ids = [p.strip() for p in paper_ids.split(',') if p.strip()]
        hits = []
        # If client also provided explicit chunk_ids (from two_stage_search), honor those
        if chunk_ids:
            cids = [c.strip() for c in chunk_ids.split(',') if c.strip()]
            ids_map = { _meta.get('ids', [])[i]: i for i in range(len(_meta.get('ids', []))) }
            for cid in cids:
                idx = ids_map.get(cid)
                if idx is None:
                    continue
                meta = _meta.get('meta', [])[idx]
                rec = dict(meta)
                rec['score'] = 1.0
                rec['id'] = _meta.get('ids', [])[idx]
                hits.append(rec)
        else:
            # best-effort: preserve order of provided paper ids and include up to `chunks_per` chunks per paper
            per_added = {pid: 0 for pid in ids}
            for pid in ids:
                if per_added.get(pid, 0) >= chunks_per:
                    continue
                for i, meta in enumerate(_meta.get('meta', [])):
                    m_pid = meta.get('paper_id') or meta.get('id') or _meta.get('ids', [])[i]
                    if m_pid == pid:
                        rec = dict(meta)
                        rec['score'] = 1.0
                        rec['id'] = _meta.get('ids', [])[i]
                        hits.append(rec)
                        per_added[pid] = per_added.get(pid, 0) + 1
                        if per_added[pid] >= chunks_per:
                            break
        used_queries = [q]
    else:
        if retr is not None:
            try:
                # apply reasonable runtime tuning (can be adjusted via env or caller)
                retr.top_m_papers = int(os.getenv("ARP_TOP_M_PAPERS", "75"))
                retr.mix_weights = (float(os.getenv("ARP_PAPER_WEIGHT", "0.4")), float(os.getenv("ARP_CHUNK_WEIGHT", "0.6")))
                # if GK expansions exist, run multi-query union; otherwise single-query retrieve
                if len(multiqueries) > 1:
                    hits = retr.multi_query_union(multiqueries, k=k, k_per_paper=chunks_per)
                    used_queries = multiqueries
                else:
                    hits = retr.retrieve(q, k=k, k_per_paper=chunks_per)
                    used_queries = [q]
            except Exception:
                hits, used_queries = _retrieve_hits(q, k, chunks_per, multiqueries=multiqueries)
        else:
            hits, used_queries = _retrieve_hits(q, k, chunks_per, multiqueries=multiqueries)
    if not hits:
        return {"query": q, "k": k, "results": [], "sources": [], "snippets": [], "answer": "No results.", "used_queries": used_queries}

    # Normalize hits that come from the two-stage retriever helper which returns
    # items like {"meta": {...}, "score": ..., "ridx": ..., "final_score": ...}
    # into the shape expected elsewhere in this module (top-level 'id', 'paper_id', 'title', 'score').
    for h in hits:
        if isinstance(h, dict) and "meta" in h and isinstance(h["meta"], dict):
            meta = h.get("meta") or {}
            # prefer existing top-level id, otherwise map from meta
            if "id" not in h or not h.get("id"):
                h["id"] = meta.get("chunk_id") or meta.get("id") or meta.get("paper_id")
            if "paper_id" not in h or not h.get("paper_id"):
                h["paper_id"] = meta.get("paper_id") or meta.get("id")
            if "title" not in h or not h.get("title"):
                title = meta.get("title") or meta.get("paper_title")
                if title:
                    h["title"] = title
            # prefer explicit final_score if provided
            if "score" not in h or h.get("score") is None:
                h["score"] = h.get("final_score", 0.0)
            # ensure numeric types
            try:
                h["score"] = float(h.get("score", 0.0))
            except Exception:
                h["score"] = 0.0

    sources = _build_sources([hits[0]] + hits[1:])
    context_blocks, snippets = _load_texts_for_hits(hits)

    # If LLM disabled, return evidence only
    if not _llm_enabled():
        payload = {
            "query": q, "k": k, "results": hits, "sources": sources, "snippets": snippets,
            "answer": "LLM not configured; returning retrieved evidence only.",
            "used_queries": used_queries, "gk": gk
        }
        if include_context: payload["context_blocks"] = context_blocks
        return payload

    # (3) Quote selection (now with real IDs available in <ctx …>)
    quotes = _select_quotes(q, context_blocks)
    if not quotes:
        # Fall back to passing the top few context blocks directly
        top_pid = sources[0]["paper_id"] if sources else "unknown"
        msgs = _compose_answer_prompt(q, "<quotes></quotes>", top_pid, style, min_src)
        try:
            resp = _client().chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                messages=[{"role":"system","content":msgs["system"]},
                          {"role":"user","content":msgs["user"] + "\n\n(Fallback: no quotes extracted; rely on context snippets.)"}],
            )
            ans = resp.choices[0].message.content or ""
            return {
                "query": q, "k": k, "results": hits, "sources": sources, "snippets": snippets,
                "answer": ans, "citations": _extract_citations(ans), "used_queries": used_queries,
                "gk": gk, "quotes": [], **({"context_blocks": context_blocks} if include_context else {})
            }
        except Exception:
            return {
                "query": q, "k": k, "results": hits, "sources": sources, "snippets": snippets,
                "answer": "LLM unavailable; evidence only.", "citations": [],
                "used_queries": used_queries, "gk": gk, "quotes": [],
                **({"context_blocks": context_blocks} if include_context else {})
            }

    # (4) Answer synthesis with self-consistency voting (enforce min sources if available)
    top_pid = sources[0]["paper_id"] if sources else "unknown"
    n = self_consistency_n or SC_N
    best_answer, cited = _synthesize_answer(
        q, quotes, top_pid, style, n=n,
        min_sources=min_src, available_sources=len(sources)
    )

    payload = {
        "query": q, "k": k, "results": hits, "sources": sources, "snippets": snippets,
        "answer": best_answer, "citations": cited, "used_queries": used_queries,
        "gk": gk, "quotes": quotes
    }
    # Preserve the full candidate set separately so clients can display it even
    # if `sources` is pruned to only those actually cited by the LLM.
    payload["candidates"] = list(sources)
    # Prune `sources` to only those actually cited in the final answer to avoid noisy tangential papers
    if cited:
        cited_set = set(cited)
        payload["sources"] = [s for s in sources if s.get("paper_id") in cited_set]
    if include_context:
        payload["context_blocks"] = context_blocks
    return payload
