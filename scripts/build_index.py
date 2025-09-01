from __future__ import annotations
import sys, json, os

# Ensure repo root on sys.path so `src` imports work when run as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import numpy as np
import faiss

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def _embed_texts_batched(texts, batch_size=128):
    """Embeds a list of texts in batches, returns a float32 (N, D) numpy array."""
    from src.embed import get_embedding
    parts = []
    for i in range(0, len(texts), batch_size):
        parts.append(get_embedding(texts[i:i+batch_size]))
    mat = np.vstack(parts).astype("float32")
    return mat

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/build_index.py <input.jsonl> <out.index>")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Use a single generator so we don't duplicate the first row
    g = load_jsonl(in_path)
    try:
        first = next(g)
    except StopIteration:
        print("Input is empty.")
        sys.exit(2)

    ids, metas = [], []
    rows = [first] + list(g)

    # Detect whether rows are chunk records (have "text" and "chunk_id")
    has_chunks = all(isinstance(r, dict) and "text" in r and "chunk_id" in r for r in rows)

    if has_chunks:
        # Prefer precomputed embeddings on a per-row basis. For rows without
        # precomputed vectors, compute embeddings in batches.
        pre_vecs = []
        to_embed_texts = []
        to_embed_indices = []
        for i, r in enumerate(rows):
            emb = None
            # Accept embeddings stored under 'embedding' or 'embeddings' keys
            if isinstance(r.get("embedding"), list):
                emb = np.array(r["embedding"], dtype="float32")
            else:
                # some workflows store embeddings under embeddings.abstract or embeddings.text
                eobj = r.get("embeddings") or {}
                if isinstance(eobj, dict):
                    # try common keys
                    for k in ("text", "abstract", "summary_short"):
                        if isinstance(eobj.get(k), list):
                            emb = np.array(eobj[k], dtype="float32")
                            break
            if emb is not None:
                pre_vecs.append(emb)
                ids.append(r["chunk_id"])
                # richer, but compact, chunk-level metadata for filtering/rerank/UI
                metas.append({
                    "chunk_id": r["chunk_id"],
                    "paper_id": r.get("paper_id"),
                    "entry_id": r.get("entry_id") or (r.get("links") or {}).get("entry"),
                    "source": r.get("source") or "arxiv",
                    "title": r.get("title"),
                    "authors_short": (lambda a: (", ".join(a[:3]) + (" et al." if len(a) > 3 else "")) if isinstance(a, list) else None)(r.get("authors")),
                    "snippet": r.get("snippet") or (r.get("text")[:300] if isinstance(r.get("text"), str) else None),
                    "page_range": r.get("page_range"),
                    "token_count": r.get("token_count"),
                    "chunk_pos": r.get("chunk_pos"),
                    "categories": r.get("categories"),
                    "published": r.get("published"),
                    "year": (lambda t: int(t[:4]) if isinstance(t, str) and len(t) >= 4 else None)(r.get("published")),
                    "doi": r.get("doi"),
                    "journal_ref": r.get("journal_ref"),
                    "links": r.get("links") or {},
                    "key_terms": r.get("key_terms"),
                    "novelty_score": r.get("novelty_score"),
                    "has_code": bool(r.get("code_repos")),
                    "has_data": bool(r.get("datasets")),
                    "idx_version": os.getenv("ARP_IDX_VERSION", "v1"),
                    "embed_model": os.getenv("ARP_EMBED_MODEL", r.get("embed_model") or os.getenv("EMBEDDING_MODEL") or os.getenv("EMBEDDING_PROVIDER")),
                })
            else:
                to_embed_texts.append(r.get("text", ""))
                to_embed_indices.append(i)

        # Embed missing texts if any
        if to_embed_texts:
            mat_new = _embed_texts_batched(to_embed_texts, batch_size=128)
            # interleave new embeddings into ids/metas in original order
            ni = 0
            for idx in to_embed_indices:
                r = rows[idx]
                vec = mat_new[ni]
                pre_vecs.append(vec)
                ids.append(r["chunk_id"])
                metas.append({
                    "chunk_id": r["chunk_id"],
                    "paper_id": r.get("paper_id"),
                    "title":    r.get("title"),
                })
                ni += 1

        if not pre_vecs:
            print("No vectors found for chunk rows.")
            sys.exit(2)

        # Validate embedding dimensions: all precomputed vectors and newly computed ones must agree
        dims = [int(p.shape[0]) for p in pre_vecs]
        if len(set(dims)) != 1:
            print(f"Inconsistent embedding dimensions found in input chunk rows: {sorted(set(dims))}")
            sys.exit(2)
        mat = np.vstack(pre_vecs)

    else:
        # Treat rows as paper-level metadata; prefer stored embeddings under
        # row.embeddings.abstract or row.embeddings.summary_short. If missing,
        # we try to embed title+abstract as fallback.
        vecs = []
        for r in rows:
            emb = None
            eobj = r.get("embeddings") or {}
            if isinstance(eobj, dict):
                for k in ("abstract", "summary_short", "title"):
                    if isinstance(eobj.get(k), list):
                        emb = np.array(eobj[k], dtype="float32")
                        break
            if emb is not None:
                vecs.append(emb)
                ids.append(r.get("id"))
                metas.append({
                    "id": r.get("id"),
                    "title": r.get("title"),
                })
            else:
                # fallback: assemble a short text for embedding
                txt = " ".join([str(r.get("title", "")), str(r.get("summary_raw", ""))])
                vecs.append(txt)
                ids.append(r.get("id"))
                metas.append({"id": r.get("id"), "title": r.get("title")})

        # If any item in vecs is a string, we need to embed the batch
        if any(isinstance(v, str) for v in vecs):
            texts = [v if isinstance(v, str) else "" for v in vecs]
            mat = _embed_texts_batched(texts, batch_size=128)
        else:
            if not vecs:
                print("No vectors found.")
                sys.exit(2)
            # validate precomputed vectors share the same dim
            dims = [int(v.shape[0]) for v in vecs]
            if len(set(dims)) != 1:
                print(f"Inconsistent embedding dimensions in paper rows: {sorted(set(dims))}")
                sys.exit(2)
            mat = np.vstack(vecs)

    # Cosine similarity via normalized vectors + inner-product index
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    faiss.write_index(index, out_path)
    # add embedding model/dimension metadata to sidecar to help detect mismatches later
    side = {"ids": ids, "meta": metas, "embedding_dim": int(mat.shape[1]), "embed_model": os.getenv("ARP_EMBED_MODEL") or os.getenv("EMBEDDING_MODEL") or os.getenv("EMBEDDING_PROVIDER")}
    with open(out_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(side, f, ensure_ascii=False, indent=2)

    print(f"Wrote index: {out_path} and {out_path}.meta.json (N={len(ids)}, D={mat.shape[1]})")

if __name__ == "__main__":
    main()
