"""Lightweight index checks: existence + embedding dimension.

Usage:
    python .\scripts\check_indexes.py

Prints a short report and exits 0 if both indexes exist and dims match, else exit 1.
"""
import sys
import os
try:
    import faiss
except Exception:
    print("faiss not importable. Install faiss-cpu (or use conda).")
    sys.exit(2)

PAPER_IDX = os.path.join("data", "index_papers.faiss")
CHUNK_IDX = os.path.join("data", "index.faiss")

def main():
    ok = True
    print(f"paper_index_exists= {os.path.exists(PAPER_IDX)}")
    print(f"chunk_index_exists= {os.path.exists(CHUNK_IDX)}")
    if not os.path.exists(PAPER_IDX) or not os.path.exists(CHUNK_IDX):
        print("One or more index files are missing.")
        sys.exit(1)
    try:
        p = faiss.read_index(PAPER_IDX)
        c = faiss.read_index(CHUNK_IDX)
        print(f"paper_d= {p.d}")
        print(f"chunk_d= {c.d}")
        if p.d != c.d:
            print("Embedding dimensions differ! Rebuild one index with the same embedding model.")
            sys.exit(1)
    except Exception as e:
        print("Error reading indexes:", e)
        sys.exit(2)
    print("OK: both indexes present and dimensions match.")
    sys.exit(0)

if __name__ == '__main__':
    main()
