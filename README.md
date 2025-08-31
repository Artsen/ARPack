# ARPack — ArXiv Research Pack

A compact, portfolio-ready pipeline that fetches recent arXiv papers, **hydrates + chunks** full texts, summarizes with an LLM into a rich schema, builds **embeddings + FAISS**, and serves **search & RAG Q\&A** via FastAPI — with a minimal UI you can open at `/ui`.

> Designed to be extended into your broader “LLM research / RAG” stack.

---

[![ARPack UI](./screenshot.webp)](http://127.0.0.1:8000/ui)

---

## ✨ What’s inside

* **Ingestion**: pulls arXiv metadata using the arXiv API. The ingest pipeline queries the arXiv ATOM API (HTTP) and prefers `feedparser` for parsing; it falls back to the official `arxiv` client when needed. The ingest code always queries starting at `start=0`, does not include an `id_list` parameter, and uses the exact `--limit` value as the `max_results` parameter sent to arXiv.
* **Hydration & Chunking**: resolves PDFs/abs pages, cleans text, splits into chunks with paper and chunk IDs.
* **Summarization**: LLM produces a structured record (problem, method, results, limitations, etc.) for each paper.
* **Embeddings**: OpenAI by default; optional local fallback (`sentence-transformers`) — batched for speed.
* **Indexing**: FAISS (IP + L2-normalized) with sidecar metadata JSON.
* **API** (FastAPI):

  * `GET /healthz` — sanity check
  * `GET /search` — dense retrieval (papers or chunks)
  * `GET /ask` — simple RAG answer
  * `GET /ask_pro` — **advanced** RAG with quote extraction, citations, self-consistency voting, and source minimums
* **UI**: zero-deps static tester at `http://127.0.0.1:8000/ui` to exercise all endpoints.

---

## 🗂️ Project layout (key paths)

```
api/
  main.py                 # FastAPI app (serves the /ui static page if mounted)
data/
  papers.jsonl            # summarized paper records
  chunks.jsonl            # hydrated + chunked text
  index.faiss             # FAISS vector index
  index.faiss.meta.json   # sidecar metadata (ids + paper/chunk meta)
  pdfs/                   # downloaded PDFs (hydration step places files here)
scripts/
  build_index.py          # builds FAISS from papers.jsonl or chunks.jsonl
  hydrate_and_chunk.py    # expands papers -> full text chunks
  check_imports.py        # small dev helper to validate imports
  debug_pdf_worker.py     # PDF worker timeout/debug helper
  pdf_worker_subproc.py   # subprocess PDF extraction helper for Windows
  preview_papers.py       # preview helper for paper JSONL
src/
  ingest.py               # fetch + summarize arXiv -> papers.jsonl
  embed.py                # embedding provider(s)
  chunk.py                # cleaners + splitters
  schema.py               # Pydantic models for records
ui_static/
  index.html              # (optional) static UI mounted at /ui
```

---

## ⚙️ Requirements

* **Python** 3.10+
* **Windows 11 / macOS / Linux** supported
* OpenAI API key (if using OpenAI embeddings/summarization)

Note: The ingest pipeline prefers `feedparser` to parse arXiv's ATOM feed. `feedparser` has been added to `requirements.txt`.

---

## 🚀 Quickstart (updated)

These steps get you from an empty repo to a running API and FAISS index.

### 1) Create venv & install

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes on system/optional deps:

* `faiss-cpu` can be tricky on Windows via pip. If `pip install -r requirements.txt` fails for faiss, use conda:

```powershell
conda install -c pytorch faiss-cpu
```

* Optional but recommended for better chunking and sanitization:
  * `tiktoken` — token-aware chunking (deterministic token spans)
  * `beautifulsoup4` (`bs4`) — HTML cleaning
  * `pandoc` (external) — high-quality ar5iv HTML -> plain text conversion; set `ARP_PANDOC_PATH` if not on PATH
  * `PyMuPDF` (`fitz`) and/or `pdfminer.six` — PDF extraction tools
  * `sentence-transformers` — optional local embedding fallback

Install extras if needed:

```powershell
pip install tiktoken beautifulsoup4 sentence-transformers pymupdf pdfminer.six
# and install pandoc via its installer or your package manager
```

### 2) Configure environment (private)

Copy the example and edit values locally. Keep the real `.env` private — do not commit it.

Windows (PowerShell):

```powershell
copy .env.example .env
notepad .env
```

Verify the running process sees your `.env` (this loads it into process env):

```powershell
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print((os.getenv('OPENAI_API_KEY') or '')[:10])"
```

Key env vars (see `.env.example`):

| Env | Purpose / default |
|-----|-------------------|
| `OPENAI_API_KEY` | OpenAI auth (if using OpenAI; leave empty for local-only runs) |
| `EMBEDDING_PROVIDER` | `openai` or `local` (default: `openai`) |
| `ARP_MIN_DISTINCT_SOURCES` | Min distinct sources for `/ask_pro` (default: `2`) |
| `ARP_SELF_CONSISTENCY_N` | Self-consistency candidate count (default: `5`) |
| `ARP_INDEX` | Path to FAISS index (default: `data/index.faiss`) |
| `ARP_META` | Sidecar metadata path (default: ARP_INDEX + `.meta.json`) |
| `ARP_CHUNKS` | Path to chunks JSONL used by API (default: `data/chunks.jsonl`) |
| `ARP_CHUNKS_PER_PAPER` | Chunks per paper returned (default: `2`) |
| `ARP_MAX_CTX_CHARS` | Max assembled context chars (default: `12000`) |
| `ARP_GK_ENABLED` | Enable generated-knowledge expansions (default: `true`) |
| `ARP_PANDOC_PATH` | Optional path to `pandoc` executable if not on PATH |

IMPORTANT: `.env.example` should be committed. Keep `.env` local and secret. The repo `.gitignore` now excludes `.env`.

### 3) Ingest papers (summaries)

Fetch metadata + structured LLM summaries into `data/papers.jsonl`:

```powershell
python -m src.ingest --cat cs.AI --limit 50 --out data/papers.jsonl
```

Notes:
* The summarization step will call the LLM if `OPENAI_API_KEY` is set; otherwise it writes heuristic fallbacks.
* The ingestion writes richly structured JSONL records (see `src/schema.py`).
* The `--limit` flag is used directly as the `max_results` parameter in the arXiv query (i.e., the exact number you request is what the code asks the API for). The ingest code always queries starting at `start=0` and does not send an `id_list` parameter.
* If you request a large `--limit` you may hit transient arXiv rate/response issues (empty page). If you see `UnexpectedEmptyPageError` or an empty result set for larger limits, try a smaller limit (e.g., 10-50) or re-run after a short delay. If you'd like, the ingest code can be extended to automatically page with smaller per-request sizes and aggregate results; I can add that for you.

### 4) Hydrate + chunk full texts (for deeper RAG)

Hydrate PDFs, extract per-page text, and chunk. Recommended to run on a small batch first.

```powershell
python .\scripts\hydrate_and_chunk.py .\data\papers.jsonl .\data\chunks.jsonl --max-papers 5 --embed-now
```

Options of note:
* `--max-papers N` — process only the first N papers (useful for smoke tests)
* `--embed-now` — compute embeddings at chunk time and embed fields into each chunk record (speeds `build_index.py`)
* `ARP_PANDOC_PATH` controls pandoc usage for ar5iv HTML -> plaintext sanitization (optional but recommended for clean text)

After running, quickly verify the output is non-empty and contains expected fields:

PowerShell checks:

```powershell
Get-Content .\data\chunks.jsonl -TotalCount 5
(Get-Content .\data\chunks.jsonl | Measure-Object -Line).Lines
python -c "import json; print(list(json.loads(open('data/chunks.jsonl').read().splitlines()[0]).keys()))"
```

### 5) Build FAISS (chunk-level recommended)

Prefer indexing `data/chunks.jsonl` for passage-level retrieval. If you intentionally want coarse retrieval, use `data/papers.jsonl`.

Chunk index (recommended):

```powershell
python .\scripts\build_index.py .\data\chunks.jsonl .\data\index.faiss
```

Paper-level index (coarser):

```powershell
python .\scripts\build_index.py .\data\papers.jsonl .\data\index.faiss
```

Notes:
* The indexer prefers precomputed embeddings present on chunk records (key `embedding` or `embeddings.*`) and only embeds missing rows.
* Typical embedding dimension (OpenAI text-embedding-3-small) is 1536; the indexer will normalize vectors before building FAISS.

### 6) Run the API

Start the FastAPI server (reload useful during development):

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

If `OPENAI_API_KEY` is not set the API will still run, but LLM-powered endpoints will return evidence-only fallbacks.

Open the UI:

```
http://127.0.0.1:8000/ui
```

### Developer helpers (what's in `scripts/`)

* `scripts/pdf_worker_subproc.py` — subprocess-based PDF extractor (used by `hydrate_and_chunk.py`); keep this for robust extraction on Windows.
* `scripts/debug_pdf_worker.py` — small multiprocessing timeout tester for PDF extraction (dev helper).
* `scripts/preview_papers.py` — convenience CLI to print compact previews of `data/papers.jsonl` for QA.

If you want me to remove optional dev helpers (e.g., `debug_pdf_worker.py`), tell me and I'll delete them and commit the change.

---

## 🔌 API overview

### `GET /healthz`

* Returns index/config status.

### `GET /search?q=...&k=8`

* Dense retrieval over FAISS.
* Returns `results` (paper/chunk hits) plus basic paper links if available.

### `GET /ask`

Parameters:

* `q` (str) — question
* `k` (int) — total diversified hits (papers)
* `k_per_paper` (int) — chunks per paper
* `style` (`concise` | `detailed`)
* `include_context` (`true`|`false`) — optional `<ctx>` blocks in response

Returns:

* `answer`, `citations` (list of paper\_ids), `sources`, `snippets`, optional `context_blocks`.

### `GET /ask_pro`

Everything in `/ask` **plus**:

* **Generated-Knowledge** expansions: `gk.subquestions`, `gk.terms`, `used_queries`
* **Quote extraction**: `quotes` — the only evidence the LLM can cite
* **Self-consistency**: multiple candidate answers → voting
* **Min source enforcement**: require ≥N distinct citations when available

Parameters (common shown):

* `q`, `k`, `k_per_paper`, `style`, `include_context`
* `self_consistency_n` (int, e.g. 5)
* `min_sources` (int, e.g. 2)

Returns:

* `answer`, `citations`, `sources`, `snippets`, `quotes`, `gk`, `used_queries`, optional `context_blocks`.

---

## 🖥️ Built-in UI (no Node)

A single static page is included for quick testing.

1. Ensure `ui_static/index.html` exists (you can use the one in this repo).
2. In `api/main.py`:

   ```python
   from fastapi.staticfiles import StaticFiles
   app.mount("/ui", StaticFiles(directory="ui_static", html=True), name="ui")
   ```
3. Run the API and open: `http://127.0.0.1:8000/ui`

The UI lets you:

* Set the API base URL
* Call `/healthz`, `/search`, `/ask`, `/ask_pro`
* Inspect **answers**, **citations**, **sources**, **snippets**, **quotes**, **GK expansions**, **context blocks**, and raw JSON.

> Prefer React? You can scaffold a `ui/` Vite app and mount its build at `/ui`. The static version works out of the box and avoids CORS/tooling.

---

## 🧠 Prompting techniques used

* **Multi-query / query expansion (GK)** to widen recall.
* **Quotes-only synthesis**: the answer is composed strictly from extracted quotes (reduces hallucinations).
* **Self-consistency voting**: generate `n` candidates → select best (distinct citations, structure checks).
* **Source minimums**: enforce ≥N distinct sources if available (`min_sources`).
* **Structured outputs**: answer format, bulleting, and citation checks.

Each `/ask_pro` response also returns the **audit trail** (queries, GK, quotes, context) for reproducibility.

---

## 🔧 Configuration

The project reads several environment variables. Keep secrets in a local `.env` file (do not commit).

| Env                        | Purpose / default                   |
| -------------------------- | ----------------------------------- |
| `OPENAI_API_KEY`           | OpenAI auth (if using OpenAI)       |
| `EMBEDDING_PROVIDER`       | `openai` or `local` (default: `openai`) |
| `ARP_MIN_DISTINCT_SOURCES` | Default min citations in `/ask_pro` (default: `2`) |
| `ARP_SELF_CONSISTENCY_N`   | Default candidate count for voting (default: `5`) |
| `ARP_INDEX`                | Path to FAISS index used by API (default: `data/index.faiss`) |
| `ARP_META`                 | Sidecar metadata path (default: ARP_INDEX + `.meta.json`) |
| `ARP_CHUNKS`               | Path to chunks JSONL used by API (default: `data/chunks.jsonl`) |
| `ARP_CHUNKS_PER_PAPER`     | Default chunks per paper returned (default: `2`) |
| `ARP_MAX_CTX_CHARS`        | Max context chars assembled for answers (default: `12000`) |
| `ARP_GK_ENABLED`           | Enable generated-knowledge in `/ask_pro` (`true`/`false`, default: `true`) |
| `ARP_PANDOC_PATH`          | Optional path to `pandoc` executable if not on PATH |

**Embedding model defaults** (see `src/embed.py`):

* OpenAI: `text-embedding-3-small`
* Local: `sentence-transformers/all-MiniLM-L6-v2`

---

## 🧪 Example calls

```bash
# Search
curl "http://127.0.0.1:8000/search?q=diffusion&k=5"

# Ask (concise)
curl "http://127.0.0.1:8000/ask?q=can%20agents%20do%20penetration%20testing%3F&k=8&k_per_paper=2&style=concise&include_context=false"

# Ask Pro (detailed, with context)
curl "http://127.0.0.1:8000/ask_pro?q=can%20agents%20do%20penetration%20testing%3F&k=8&k_per_paper=2&style=detailed&include_context=true&self_consistency_n=5&min_sources=2"
```

---

## 🩺 Troubleshooting

* **`ModuleNotFoundError: No module named 'src'`**
  Run modules with `-m` from the project root, or ensure your CWD is the repo root:

  ```powershell
  python -m src.ingest --cat cs.AI --limit 25
  python .\scripts\hydrate_and_chunk.py .\data\papers.jsonl .\data\chunks.jsonl
  python .\scripts\build_index.py .\data\chunks.jsonl .\data\index.faiss
  ```

* **Wrong API key showing up**
  Windows env vars can override `.env`. Check which key is loaded:

  ```powershell
  python -c "import os; from dotenv import load_dotenv; load_dotenv(); print((os.getenv('OPENAI_API_KEY') or '')[:10])"
  ```

  Update/remove user/system env var if needed and restart your shell.

* **CORS errors (if using a separate frontend dev server)**
  Add CORS middleware to `api.main.py`:

  ```python
  from fastapi.middleware.cors import CORSMiddleware
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["http://localhost:5173","http://127.0.0.1:5173"],
      allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
  )
  ```

---

## 🛣️ Extending ideas

* Schedule ingest (GitHub Actions + releases of `papers.jsonl`, `chunks.jsonl`, `index.faiss`).
* Add a cross-encoder re-ranker (e.g., BGE or OpenAI re-ranker) after FAISS.
* Add incremental ingest with last-seen timestamps & dedupe.
* Enrich schema with citations, code links, and task/dataset tags.
* Add auth + per-user collections.
* Export answers + sources as Markdown reports.

---

## ⚠️ Ethics & Safety

This project surfaces security research. **Do not** use it to attack systems you do not own or have explicit permission to test. Follow applicable laws, licenses, and arXiv terms.

---

## 📝 License

MIT (see `LICENSE`). Contributions welcome!
