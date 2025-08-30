from __future__ import annotations
import sys, json, os
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

    is_chunks = ("text" in first) and ("chunk_id" in first)
    ids, metas = [], []

    if is_chunks:
        # INDEX OVER CHUNKS (embed fresh)
        rows = [first] + list(g)
        texts = [r["text"] for r in rows]
        mat = _embed_texts_batched(texts, batch_size=128)  # (N, D)

        # Build metadata arrays in the same order
        for r in rows:
            ids.append(r["chunk_id"])
            metas.append({
                "chunk_id":  r["chunk_id"],
                "paper_id":  r["paper_id"],
                "title":     r["title"],
                "authors":   r["authors"],
                "categories":r["categories"],
                "published": r["published"],
            })

    else:
        # INDEX OVER PAPERS (use stored embeddings)
        vecs = []
        rows = [first] + list(g)
        for row in rows:
            emb = row.get("embeddings", {}).get("abstract") or row.get("embeddings", {}).get("summary_short")
            if emb is None:
                continue
            vecs.append(np.array(emb, dtype="float32"))
            ids.append(row["id"])
            metas.append({
                "id":         row["id"],
                "title":      row["title"],
                "authors":    row["authors"],
                "categories": row["categories"],
                "published":  row["published"],
            })
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
