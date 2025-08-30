from __future__ import annotations
import os, json, sys
import fitz  # PyMuPDF
from tqdm import tqdm
from src.chunk import clean_text, split_into_chunks

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def pdf_to_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    return "\n".join(parts)

def main(papers_path: str, chunks_out: str, pdf_dir: str = "data/pdfs"):
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(os.path.dirname(chunks_out), exist_ok=True)

    with open(chunks_out, "w", encoding="utf-8") as out:
        for row in tqdm(list(load_jsonl(papers_path)), desc="Hydrate+Chunk"):
            pid = row["id"]  # e.g., 2508.21001v1
            pdf_url = row.get("links", {}).get("pdf")
            if not pdf_url:
                continue

            pdf_path = os.path.join(pdf_dir, f"{pid}.pdf")
            if not os.path.exists(pdf_path):
                # simple download
                import urllib.request
                try:
                    urllib.request.urlretrieve(pdf_url, pdf_path)
                except Exception:
                    continue

            try:
                text = pdf_to_text(pdf_path)
            except Exception:
                continue

            text = clean_text(text)
            chunks = split_into_chunks(text, target_chars=1400, overlap=200)
            for i, ch in enumerate(chunks):
                out.write(json.dumps({
                    "paper_id": pid,
                    "chunk_id": f"{pid}::c{i:04d}",
                    "title": row["title"],
                    "authors": row["authors"],
                    "categories": row["categories"],
                    "published": row["published"],
                    "text": ch
                }, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/hydrate_and_chunk.py data/papers.jsonl data/chunks.jsonl")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
