from __future__ import annotations
import os
import sys
import re
import json
import argparse
import urllib.request
import multiprocessing as mp
import subprocess
import shutil
import tempfile
import time
from typing import Iterable, Optional, List, Dict
# Ensure repo root on sys.path so `src` imports work when run as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.chunk import chunk_by_tokens, count_tokens
from tqdm import tqdm
import bisect
import unicodedata
import difflib

# Optional HTML sanitizer / pandoc
try:
    from bs4 import BeautifulSoup  # type: ignore
    _HAS_BS4 = True
except Exception:
    _HAS_BS4 = False

PANDOC_BIN = os.getenv("ARP_PANDOC_PATH") or "pandoc"

def has_pandoc() -> bool:
    if os.path.sep in PANDOC_BIN or PANDOC_BIN.lower().endswith('.exe'):
        return os.path.isfile(PANDOC_BIN)
    return shutil.which(PANDOC_BIN) is not None

def pandoc_convert_text(inp_text: str, from_fmt: str, to_fmt: str = "plain", extra_args: Optional[List[str]] = None, timeout: int = 40) -> Optional[str]:
    if not has_pandoc():
        return None
    extra_args = extra_args or []
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "in.html")
        out_path = os.path.join(tmp, "out.txt")
        with open(in_path, "w", encoding="utf-8") as f:
            f.write(inp_text)
        cmd = [PANDOC_BIN, "-f", from_fmt, "-t", to_fmt, "--wrap=none"] + extra_args + [in_path, "-o", out_path]
        try:
            subprocess.run(cmd, check=True, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            log(f"[pandoc] failed: {e}")
            return None

def html_from_ar5iv(arxiv_id: str, timeout: int = 15) -> Optional[str]:
    url = f"https://ar5iv.org/html/{arxiv_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        html = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", errors="ignore")
        return html
    except Exception:
        return None

def ar5iv_to_plaintext(arxiv_id: str, pandoc_timeout: int = 30) -> Optional[str]:
    html = html_from_ar5iv(arxiv_id)
    if not html:
        return None
    # remove nav/header/footer if bs4 present
    if _HAS_BS4:
        try:
            soup = BeautifulSoup(html, "lxml")
            for sel in ["nav", "header", "footer"]:
                for el in soup.select(sel):
                    el.decompose()
            html = str(soup)
        except Exception:
            pass
    # Convert HTML -> plain text using pandoc (safe for embeddings)
    txt = pandoc_convert_text(html, from_fmt="html", to_fmt="plain", timeout=pandoc_timeout)
    if txt:
        # final sanitize: strip remaining tags if any
        if _HAS_BS4:
            try:
                soup = BeautifulSoup(txt, "lxml")
                return soup.get_text("\n").strip()
            except Exception:
                return txt.strip()
        return re.sub(r"<[^>]+>", "", txt).strip()
    return None

# --- Ensure repo root on sys.path so "src" imports work when run as a script ---
# repo/
#   src/
#     chunk.py
#   scripts/
#     hydrate_and_chunk.py  <-- this file
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# -------- Try your utilities; provide fallbacks if missing --------
try:
    from src.chunk import clean_text, split_into_chunks  # type: ignore
except Exception:
    def clean_text(s: str) -> str:
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"[ \t]+\n", "\n", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    def split_into_chunks(txt: str, target_chars: int = 1200, overlap: int = 150) -> List[str]:
        paras = [p.strip() for p in txt.split("\n\n") if p.strip()]
        chunks: List[str] = []
        cur = ""
        for p in paras:
            if len(cur) + len(p) + 2 <= target_chars:
                cur = (cur + "\n\n" + p) if cur else p
            else:
                if cur:
                    chunks.append(cur)
                cur = p
        if cur:
            chunks.append(cur)
        if overlap > 0 and len(chunks) > 1:
            out: List[str] = []
            for i, ch in enumerate(chunks):
                if i == 0:
                    out.append(ch)
                else:
                    tail = chunks[i-1][-overlap:]
                    out.append((tail + "\n\n" + ch).strip())
            chunks = out
        return chunks

# -------- Minimal helpers --------
UA = "Mozilla/5.0 (compatible; ARPack/1.0; +https://github.com/)"
VERBOSE = os.getenv("ARP_VERBOSE", "0") == "1"

def log(msg: str) -> None:
    if VERBOSE:
        print(msg, flush=True)

def load_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def download_pdf(url: str, dst_path: str, timeout: int = 30, retries: int = 2) -> Optional[str]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r, open(dst_path, "wb") as out:
                out.write(r.read())
            return dst_path
        except Exception as e:
            if attempt < retries:
                time.sleep(0.8 * (attempt + 1))
            else:
                log(f"[download] failed {url} -> {e}")
    return None

# -------- PDF extraction workers (PyMuPDF primary, pdfminer fallback) --------
def _pdf_worker(path: str, q: mp.Queue) -> None:
    """Isolated process: try PyMuPDF; fallback to pdfminer.six."""
    try:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            parts = [page.get_text("text") for page in doc]
            txt = "\n".join(parts)
            if txt.strip():
                q.put(txt)
                return
        except Exception:
            pass

        # Fallback: pdfminer.six
        try:
            from pdfminer.high_level import extract_text  # type: ignore
            txt = extract_text(path) or ""
            q.put(txt)
            return
        except Exception:
            q.put("")
    except Exception:
        q.put("")

def pdf_to_text_with_timeout(pdf_path: str, timeout_sec: int = 90) -> Optional[str]:
    # Try fast in-process extraction first (works for most PDFs and avoids
    # Windows spawn-related hangs). If it fails, fall back to an isolated
    # subprocess worker with a hard timeout.
    try:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            parts = [page.get_text("text") for page in doc]
            txt = "\n".join(parts)
            if txt and txt.strip():
                return txt
        except Exception:
            pass

        try:
            from pdfminer.high_level import extract_text  # type: ignore
            txt = extract_text(pdf_path) or ""
            if txt and txt.strip():
                return txt
        except Exception:
            pass
    except Exception:
        # If any unexpected error in-process, try the isolated worker below.
        pass
    # Fallback: call a subprocess worker (avoids mp.Process spawn quirks on Windows)
    worker = os.path.join(os.path.dirname(__file__), "pdf_worker_subproc.py")
    try:
        res = subprocess.run([sys.executable, worker, pdf_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_sec)
        txt = (res.stdout or "").strip()
        if txt:
            return txt
        # If worker produced stderr, log it for diagnostics
        if res.stderr:
            log(f"[pdf worker stderr] {res.stderr.strip()}")
        return None
    except subprocess.TimeoutExpired:
        log(f"[pdf] timeout after {timeout_sec}s on {pdf_path}")
        return None
    except Exception as e:
        log(f"[pdf worker] error: {e}")
        return None


def extract_pdf_pages_with_timeout(pdf_path: str, timeout_sec: int = 90) -> Optional[List[str]]:
    """Return list of per-page texts when possible. Falls back to single-item list with full text.
    Uses PyMuPDF in-process or pdfminer (splits on form-feed). If those fail, calls subprocess worker and
    returns a single-element list containing the full text.
    """
    try:
        try:
            import fitz
            doc = fitz.open(pdf_path)
            parts = [page.get_text("text") for page in doc]
            if any(p and p.strip() for p in parts):
                return parts
        except Exception:
            pass

        try:
            from pdfminer.high_level import extract_text
            txt = extract_text(pdf_path) or ""
            pages = [p for p in txt.split('\f')]
            if len(pages) > 1 and any(p.strip() for p in pages):
                return pages
            if txt and txt.strip():
                return [txt]
        except Exception:
            pass

        # Last resort: call subprocess worker to get full text
        worker = os.path.join(os.path.dirname(__file__), "pdf_worker_subproc.py")
        try:
            res = subprocess.run([sys.executable, worker, pdf_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_sec)
            txt = (res.stdout or "").strip()
            if txt:
                return [txt]
            return None
        except Exception:
            return None
    except Exception:
        return None

# -------- Main pipeline (PDF-only) --------
def main(
    papers_path: str,
    chunks_out: str,
    pdf_dir: str = "data/pdfs",
    target_chars: int = 1400,
    overlap: int = 200,
    pdf_timeout: int = 90,
    max_papers: Optional[int] = None,
    report_path: Optional[str] = None,
    embed_now: bool = False,
) -> None:
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(os.path.dirname(chunks_out), exist_ok=True)

    rows = list(load_jsonl(papers_path))
    if max_papers is not None:
        rows = rows[:max_papers]

    stats: Dict[str, int] = {
        "rows_total": len(rows),
        "rows_processed": 0,
        "skipped_missing_id": 0,
        "skipped_missing_pdf": 0,
        "download_failed": 0,
        "extract_timeout_or_empty": 0,
        "chunks_written": 0,
    "embeddings_written": 0,
        "papers_with_chunks": 0,
    }
    # aggregate extraction timing (ms)
    stats["total_extraction_time_ms"] = 0
    stats["max_extraction_time_ms"] = 0

    with open(chunks_out, "w", encoding="utf-8") as out:
        for row in tqdm(rows, desc="Hydrate+Chunk [pdf]"):
            pid = row.get("id")
            if not pid:
                stats["skipped_missing_id"] += 1
                log(f"[skip] missing id: {row}")
                continue

            pdf_url = (row.get("links") or {}).get("pdf")
            # Fallback: derive from id if missing
            if not pdf_url:
                pdf_url = f"https://arxiv.org/pdf/{pid}.pdf"

            if not pdf_url:
                stats["skipped_missing_pdf"] += 1
                log(f"[{pid}] no pdf url; skipping")
                continue

            pdf_path = os.path.join(pdf_dir, f"{pid}.pdf")
            if not os.path.exists(pdf_path):
                if not download_pdf(pdf_url, pdf_path, timeout=30):
                    stats["download_failed"] += 1
                    log(f"[{pid}] download failed; skipping")
                    continue

            # Extract per-page when possible; measure elapsed time per-paper
            start_ts = time.time()
            pages = extract_pdf_pages_with_timeout(pdf_path, timeout_sec=pdf_timeout)
            elapsed_ms = int((time.time() - start_ts) * 1000)
            if not pages:
                stats["extract_timeout_or_empty"] += 1
                log(f"[{pid}] pdf extract failed/timeout/empty; skipping")
                stats["total_extraction_time_ms"] += elapsed_ms
                if elapsed_ms > stats.get("max_extraction_time_ms", 0):
                    stats["max_extraction_time_ms"] = elapsed_ms
                continue

            # pages: List[str] where each element corresponds to a PDF page (best effort)
            pages = [clean_text(p) for p in pages]
            full_text = "\n\n".join(pages)
            # prefer token-aware chunking when available; attempt to chunk within page boundaries
            chunks: List[str] = []
            chunk_page_ranges: List[str] = []
            extraction_engine = "token_chunker"
            try:
                for pi, ptxt in enumerate(pages):
                    if not ptxt.strip():
                        continue
                    sub_chunks = chunk_by_tokens(ptxt, target_tokens=max(100, target_chars//2), overlap_tokens=max(50, overlap//2))
                    for sc in sub_chunks:
                        chunks.append(sc)
                        # page indices are 0-based; store 1-based human-friendly
                        chunk_page_ranges.append(f"{pi+1}-{pi+1}")
            except Exception:
                # fallback: chunk full_text using char-based splitter
                extraction_engine = "char_chunker"
                chunks = split_into_chunks(full_text, target_chars=target_chars, overlap=overlap)
                # approximate page ranges: attempt to map chunk start to page by token counts
                # simple fallback: assign all chunks to full doc range
                chunk_page_ranges = ["1-?" for _ in chunks]

            # refine chunk -> page mapping: compute accurate page ranges when possible
            # We'll normalize pages and full_text for more robust matching and build
            # normalized page_starts to map match indices back to page numbers.
            def _normalize_whitespace(s: str) -> str:
                # NFKC normalize and collapse whitespace to single spaces
                s = unicodedata.normalize("NFKC", s)
                s = re.sub(r"\s+", " ", s)
                return s.strip()

            norm_pages = [_normalize_whitespace(p) for p in pages]
            norm_full = _normalize_whitespace(" ".join(norm_pages))
            page_starts: List[int] = []
            cur_off = 0
            for np in norm_pages:
                page_starts.append(cur_off)
                cur_off += len(np) + 1  # pages joined by a single space in norm_full

            # Token-aware page starts: use the same tokenizer/count_tokens as the chunker
            page_token_counts: List[int] = [max(0, count_tokens(p)) for p in norm_pages]
            page_token_starts: List[int] = []
            tcur = 0
            for tc in page_token_counts:
                page_token_starts.append(tcur)
                tcur += tc

            def map_chunks_to_page_ranges(chunks_list: List[dict]) -> List[str]:
                """Map a list of chunk dicts (with token_start/token_end) to page ranges
                using page_token_starts computed earlier. If token spans are missing,
                fall back to the first-page assignment.
                """
                out: List[str] = []
                for ch in chunks_list:
                    if not isinstance(ch, dict):
                        out.append("1-1")
                        continue
                    ts = ch.get("token_start")
                    te = ch.get("token_end")
                    if ts is None or te is None:
                        out.append("1-1")
                        continue
                    si = bisect.bisect_right(page_token_starts, ts) - 1
                    ei = bisect.bisect_right(page_token_starts, te) - 1
                    si = max(0, min(si, len(page_token_starts) - 1))
                    ei = max(0, min(ei, len(page_token_starts) - 1))
                    out.append(f"{si+1}-{ei+1}")
                return out

            wrote_any = False
            # If embedding now, compute embeddings for all chunks of this paper in one call
            embs = None
            if embed_now and chunks:
                try:
                    from src.embed import get_embedding
                    texts = [c["text"] if isinstance(c, dict) else str(c) for c in chunks]
                    mat = get_embedding(texts)
                    # mat: (N, D) numpy array
                    embs = [v.tolist() for v in mat]
                except Exception as e:
                    log(f"[embed] embedding failed for {pid}: {e}")

            # If initial chunk_page_ranges look like fallback placeholders or if pages>1,
            # attempt to compute accurate ranges for all chunks (and token spans).
            # Since chunk_by_tokens now produces token spans when possible, prefer those.
            token_spans = []
            try:
                for ch in chunks:
                    if isinstance(ch, dict):
                        token_spans.append((ch.get("token_start"), ch.get("token_end")))
                    else:
                        token_spans.append((None, None))
            except Exception:
                token_spans = [(None, None) for _ in chunks]

            # Attempt to refine page ranges using token spans mapping
            try:
                refined_ranges = map_chunks_to_page_ranges(chunks)
                if any(r != "1-1" for r in refined_ranges):
                    chunk_page_ranges = refined_ranges
            except Exception:
                pass

            for i, ch in enumerate(chunks):
                # get text and metrics
                text = ch.get("text") if isinstance(ch, dict) else str(ch)
                if not text or not str(text).strip():
                    continue
                tc = None
                if isinstance(ch, dict) and ch.get("token_count") is not None:
                    tc = int(ch.get("token_count"))
                else:
                    tc = max(1, count_tokens(text))

                ts, te = token_spans[i] if i < len(token_spans) else (None, None)

                rec = {
                    "paper_id": pid,
                    "chunk_id": f"{pid}::c{i:04d}",
                    "title": row.get("title"),
                    "authors": row.get("authors"),
                    "categories": row.get("categories"),
                    "published": row.get("published"),
                    "text": text,
                    "token_count": tc,
                    "sentence_count": max(1, len([s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()])),
                    "extraction_engine": extraction_engine,
                    # same extraction time for all chunks produced from this paper
                    "extraction_time_ms": elapsed_ms,
                    "section_path": "",        # pdf-only: no section map
                    "source_type": "pdf",
                    "pdf_path": pdf_path,
                    "page_range": chunk_page_ranges[i] if i < len(chunk_page_ranges) else "1-1",
                    # token_start/token_end (inclusive) if available
                    "token_start": int(ts) if ts is not None else None,
                    "token_end": int(te) if te is not None else None,
                    # small snippet and content hash for quick preview/dedup
                    "snippet": (text[:200].strip()),
                    "sha1": None,
                }
                try:
                    import hashlib as _hashlib
                    rec["sha1"] = _hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()
                except Exception:
                    rec["sha1"] = None

                if embs is not None:
                    try:
                        rec["embedding"] = embs[i]
                        stats["embeddings_written"] += 1
                    except Exception:
                        pass
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["chunks_written"] += 1
                wrote_any = True

            if wrote_any:
                stats["papers_with_chunks"] += 1
                # aggregate timing only for papers that yielded chunks
                stats["total_extraction_time_ms"] += elapsed_ms
                if elapsed_ms > stats.get("max_extraction_time_ms", 0):
                    stats["max_extraction_time_ms"] = elapsed_ms

            stats["rows_processed"] += 1

    # optional report
    if report_path:
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
        except Exception:
            pass

    print(
        f"[summary] rows={stats['rows_total']} processed={stats['rows_processed']} "
        f"papers_with_chunks={stats['papers_with_chunks']} chunks={stats['chunks_written']} "
        f"skips: missing_id={stats['skipped_missing_id']}, missing_pdf={stats['skipped_missing_pdf']}, "
        f"download_failed={stats['download_failed']}, extract_issues={stats['extract_timeout_or_empty']}"
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hydrate + chunk arXiv PDFs into JSONL (PDF-only, stable).")
    ap.add_argument("papers_jsonl", help="e.g., data/papers.jsonl")
    ap.add_argument("chunks_out", help="e.g., data/chunks.jsonl")
    ap.add_argument("--pdf-dir", default="data/pdfs")
    ap.add_argument("--target-chars", type=int, default=1400)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--pdf-timeout", type=int, default=90, help="Hard timeout per-PDF extraction (seconds).")
    ap.add_argument("--max-papers", type=int, default=None)
    ap.add_argument("--report", default=None, help="Optional JSON report path, e.g., data/hydrate_report.json")
    ap.add_argument("--embed-now", action="store_true", help="Compute and store embeddings for chunks immediately")
    args = ap.parse_args()

    main(
        args.papers_jsonl,
        args.chunks_out,
        pdf_dir=args.pdf_dir,
        target_chars=args.target_chars,
        overlap=args.overlap,
        pdf_timeout=args.pdf_timeout,
        max_papers=args.max_papers,
        report_path=args.report,
        embed_now=args.embed_now,
    )
