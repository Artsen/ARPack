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
    """Backward-compatible char-based chunker (keeps old behaviour)."""
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


# Token-aware chunking -----------------------------------------------------
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def count_tokens(s: str) -> int:
        return len(_ENC.encode(s))
except Exception:
    def count_tokens(s: str) -> int:
        # rough fallback: 4 chars ~= 1 token
        return max(1, len(s) // 4)

def chunk_by_tokens(txt: str, target_tokens: int = 800, overlap_tokens: int = 150) -> List[str]:
    """Token-aware chunker.

    If a tokenizer (tiktoken) is available we perform token-level slicing and
    return a list of dicts: {text, token_count, token_start, token_end} where
    token_start/token_end are token indices relative to the input `txt`.

    If tokenizer is not available, fall back to the char-based packing and
    return list of dicts with token_start/token_end set to None.
    """
    # Use token-level slicing when encoding is available for deterministic spans
    try:
        # _ENC is available in this module when tiktoken imported successfully
        enc = _ENC
    except NameError:
        enc = None

    out: List[dict] = []
    if enc is not None:
        toks = enc.encode(txt)
        n = len(toks)
        if target_tokens <= 0:
            target_tokens = max(1, n)
        stride = max(1, target_tokens - max(0, overlap_tokens))
        i = 0
        while i < n:
            chunk_tokens = toks[i : i + target_tokens]
            try:
                chunk_text = enc.decode(chunk_tokens)
            except Exception:
                # best-effort: decode individually and join
                chunk_text = "".join(enc.decode([t]) for t in chunk_tokens)
            token_start = i
            token_end = i + len(chunk_tokens) - 1
            out.append({
                "text": chunk_text.strip(),
                "token_count": len(chunk_tokens),
                "token_start": token_start,
                "token_end": token_end,
            })
            i += stride
        return out

    # fallback: use paragraph packing and approximate token counts
    paras = [p.strip() for p in txt.split("\n\n") if p.strip()]
    chunks: List[str] = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 2 <= target_tokens * 4:  # rough char -> token heuristic
            cur = (cur + "\n\n" + p) if cur else p
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)

    # overlap by simple tailing text
    if overlap_tokens > 0 and len(chunks) > 1:
        out_chunks: List[str] = []
        for i, ch in enumerate(chunks):
            if i == 0:
                out_chunks.append(ch)
            else:
                tail = chunks[i - 1][- (overlap_tokens * 4) :]
                out_chunks.append((tail + "\n\n" + ch).strip())
        chunks = out_chunks

    for ch in chunks:
        out.append({
            "text": ch,
            "token_count": count_tokens(ch),
            "token_start": None,
            "token_end": None,
        })
    return out
