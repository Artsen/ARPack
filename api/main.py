from __future__ import annotations
import os, json, re
from typing import Dict, Any, List, Tuple, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
import faiss
from dotenv import load_dotenv

# Ensure .env overrides registry/shell for this process
load_dotenv(override=True)

from src.embed import get_embedding

# ---------- Config ----------
INDEX_PATH             = os.getenv("ARP_INDEX", "data/index.faiss")
META_PATH              = os.getenv("ARP_META", INDEX_PATH + ".meta.json")
CHUNKS_PATH            = os.getenv("ARP_CHUNKS", "data/chunks.jsonl")

# Defaults (tunable via env)
CHUNKS_PER_PAPER_DEF   = int(os.getenv("ARP_CHUNKS_PER_PAPER", "2"))
MAX_CTX_CHARS          = int(os.getenv("ARP_MAX_CTX_CHARS", "12000"))
GK_ENABLED             = os.getenv("ARP_GK_ENABLED", "true").lower() in {"1","true","yes","on"}
SC_N                   = int(os.getenv("ARP_SELF_CONSISTENCY_N", "3"))  # candidates to sample in voting
MIN_DISTINCT_SOURCES   = int(os.getenv("ARP_MIN_DISTINCT_SOURCES", "1")) # enforce multi-source when available

# ---------- App ----------
app = FastAPI(title="ARPack API", version="0.5.1")
app.mount("/ui", StaticFiles(directory="ui_static", html=True), name="ui")

_index = None
_meta  = None
_is_chunk_index = False

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

# ---------- Routes ----------
@app.get("/")
def root():
    return {"ok": True, "docs": "/docs", "endpoints": ["/healthz", "/search", "/ask", "/ask_pro"]}

@app.get("/healthz")
def healthz():
    return {"ok": True, "index_path": INDEX_PATH, "meta_path": META_PATH, "chunk_mode": _is_chunk_index}

@app.get("/search")
def search(q: str = Query(...), k: int = Query(5, ge=1, le=50)) -> Dict[str, Any]:
    vec = get_embedding([q])
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
    return {"query": q, "k": k, "results": results}

# Your earlier structured /ask (kept; now feeds XML ctx too)
@app.get("/ask")
def ask(
    q: str = Query(...),
    k: int = Query(6, ge=1, le=50),
    k_per_paper: int | None = Query(None, ge=1, le=10),
    include_context: bool = Query(False),
    style: str = Query("concise")
) -> Dict[str, Any]:
    chunks_per = k_per_paper or CHUNKS_PER_PAPER_DEF
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
    hits, used_queries = _retrieve_hits(q, k, chunks_per, multiqueries=multiqueries)
    if not hits:
        return {"query": q, "k": k, "results": [], "sources": [], "snippets": [], "answer": "No results.", "used_queries": used_queries}

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
    if include_context:
        payload["context_blocks"] = context_blocks
    return payload
