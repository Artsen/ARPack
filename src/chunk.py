from __future__ import annotations
import re
from typing import List, Tuple

def clean_text(txt: str) -> str:
    # collapse whitespace, keep paragraphs
    txt = txt.replace("\r", "")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def split_into_chunks(txt: str, target_chars: int = 1200, overlap: int = 150) -> List[str]:
    """
    Rough, fast chunker: split on paragraphs; then pack to ~target_chars with overlap.
    """
    paras = [p.strip() for p in txt.split("\n\n") if p.strip()]
    chunks = []
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

    # add simple overlaps
    if overlap > 0 and len(chunks) > 1:
        out = []
        for i, ch in enumerate(chunks):
            if i == 0:
                out.append(ch)
            else:
                tail = chunks[i-1][-overlap:]
                out.append(tail + "\n\n" + ch)
        chunks = out
    return chunks
