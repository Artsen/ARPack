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
                metas.append({
                    "chunk_id": r["chunk_id"],
                    "paper_id": r.get("paper_id"),
                    "title":    r.get("title"),
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
            mat = np.vstack(vecs)

    # Cosine similarity via normalized vectors + inner-product index
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    faiss.write_index(index, out_path)
    with open(out_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"ids": ids, "meta": metas}, f, ensure_ascii=False, indent=2)

    print(f"Wrote index: {out_path} and {out_path}.meta.json (N={len(ids)}, D={mat.shape[1]})")

if __name__ == "__main__":
    main()
