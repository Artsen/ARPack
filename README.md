# ARPack — ArXiv Research Pack

A compact, portfolio-ready pipeline that fetches recent arXiv papers, **hydrates + chunks** full texts, summarizes with an LLM into a rich schema, builds **embeddings + FAISS**, and serves **search & RAG Q\&A** via FastAPI — with a minimal UI you can open at `/ui`.

> Designed to be extended into your broader “LLM research / RAG” stack.

---

![ARPack UI](./screenshot.jpg)

---

## ✨ What’s inside

* **Ingestion**: pulls arXiv metadata (official client; no scraping).
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
scripts/
  build_index.py          # builds FAISS from papers.jsonl or chunks.jsonl
  hydrate_and_chunk.py    # expands papers -> full text chunks
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

---

## 🚀 Quickstart

### 1) Create venv & install

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment

Copy and edit:

```bash
cp .env.example .env
```

Required/Useful keys:

| Key                        | What                                        | Example  |
| -------------------------- | ------------------------------------------- | -------- |
| `OPENAI_API_KEY`           | For LLM summarization + embeddings (OpenAI) | `sk-...` |
| `EMBEDDING_PROVIDER`       | `openai` or `local`                         | `openai` |
| `ARP_MIN_DISTINCT_SOURCES` | Default min sources for `/ask_pro`          | `2`      |
| `ARP_SELF_CONSISTENCY_N`   | Default vote count for `/ask_pro`           | `5`      |

> **Windows note:** if an old API key is “stuck”, your system/user environment variables may override `.env`. You can verify what your app sees with:
>
> ```powershell
> python -c "import os; from dotenv import load_dotenv; load_dotenv(); print((os.getenv('OPENAI_API_KEY') or '')[:10])"
> ```
>
> If it shows the wrong prefix, update/remove the Windows env var and reopen your shell.

### 3) Ingest papers (summaries)

Pick a category and limit:

```powershell
# Windows
python -m src.ingest --cat cs.AI --limit 50
```

This writes `data/papers.jsonl`.

### 4) Hydrate + chunk full texts (for deeper RAG)

```powershell
python .\scripts\hydrate_and_chunk.py .\data\papers.jsonl .\data\chunks.jsonl
```

### 5) Build FAISS (from **chunks** or **papers**)

* **Chunk index (recommended for Q\&A quality):**

```powershell
python .\scripts\build_index.py .\data\chunks.jsonl .\data\index.faiss
```

* **Abstract/summary index (faster, coarser):**

```powershell
python .\scripts\build_index.py .\data\papers.jsonl .\data\index.faiss
```

### 6) Run the API

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Open the UI:

```
http://127.0.0.1:8000/ui
```

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

| Env                        | Purpose                             |
| -------------------------- | ----------------------------------- |
| `OPENAI_API_KEY`           | OpenAI auth (if using OpenAI)       |
| `EMBEDDING_PROVIDER`       | `openai` or `local`                 |
| `ARP_MIN_DISTINCT_SOURCES` | Default min citations in `/ask_pro` |
| `ARP_SELF_CONSISTENCY_N`   | Default candidate count for voting  |

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
